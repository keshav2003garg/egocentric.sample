"""Writes one clip's viewer file: every available layer, an overview, and the layout."""
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import rerun as rr
import rerun.blueprint as rrb

from ego_rerun.clip import Clip, note
from ego_rerun.layers import LAYERS, TAB_ORDER, steps

APP_ID = "ego_sample"
# 3D view: looks at where the hands work, from behind-left and above (tuned on clip 004)
EYE_OFFSET = np.array([-0.42, -0.31, 0.58])


def look_target(clip: Clip) -> np.ndarray:
    """Middle of where the wrists go during the clip."""
    if not clip.has("hands.parquet"):
        return np.zeros(3)
    hands = pd.read_parquet(clip.dir / "hands.parquet", columns=["present", "wrist_pos_world"])
    wrists = np.stack(hands.loc[hands["present"], "wrist_pos_world"].to_numpy())
    return np.median(wrists, axis=0)


def overview(clip: Clip, summaries: dict):
    n = clip.json("camera_path_report.json")["frames_solved"].split()[-1] if clip.has("camera_path_report.json") else "–"
    layers = "\n".join(f"| {name} | {text} |" for name, text in summaries.items())
    legend = "\n".join(f"| {name} | {clip.label(k)} |" for k, (name, _) in clip.palette.items())
    task = clip.json("task.json") if clip.has("task.json") else {}
    note("notes/overview", f"""
## {clip.title}
{task.get('summary', clip.notes.get('task', ''))}

| | |
|---|---|
| Setting | {clip.notes.get('setting', '–')} |
| Captured | {clip.capture_date} |
| Length | {n} frames at 50 fps ({int(n) / 50:.1f} s) |

### Layers
| Layer | What it is |
|---|---|
{layers}

### Colours
| Colour | Item |
|---|---|
{legend}

### Using the viewer
- **Play / scrub** on the timeline at the bottom; every view follows the same frame.
- **3D view:** drag to orbit, scroll to zoom, double-click to reset. Click an item to see its values.
- **Tabs on the right** hold each layer's plots and its notes.
""")


def blueprint(clip: Clip, tabs: list, has_now: bool):
    target = look_target(clip)
    scene = rrb.Spatial3DView(
        origin="world", name="3D scene",
        eye_controls=rrb.EyeControls3D(position=(target + EYE_OFFSET).tolist(), look_target=target.tolist(), eye_up=[0, 0, 1]),
        line_grid=rrb.LineGrid3D(visible=True, spacing=0.1),
    )
    video = rrb.Spatial2DView(origin="video", name="Video · masks · hands")
    depth = rrb.Spatial2DView(origin="depth", name="Depth (red near · blue far)")
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(rrb.Horizontal(video, depth), scene, row_shares=[2, 3]),
            rrb.Vertical(
                *([rrb.TextDocumentView(origin="notes/now_step", name="Now: step"),
                   rrb.TextDocumentView(origin="notes/now_hands", name="Now: hands")] if has_now else []),
                rrb.Tabs(rrb.TextDocumentView(origin="notes/overview", name="Overview"), *tabs, active_tab="Overview"),
                row_shares=[0.9, 0.8, 3] if has_now else None,
            ),
            column_shares=[3, 2],
        ),
        rrb.TimePanel(timeline="time", state="collapsed", play_state="paused"),
        rrb.SelectionPanel(state="collapsed"),
        rrb.BlueprintPanel(state="collapsed"),
    )


def web_video(source: Path, out: Path) -> Path:
    """Smaller H.264 copy for the hosted viewer: same frames and timing, no B-frames, keyframe every second."""
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(source), "-an", "-c:v", "libx264", "-preset", "slow",
                    "-crf", "30", "-pix_fmt", "yuv420p", "-bf", "0", "-g", "50", "-movflags", "+faststart", str(out)],
                   check=True)
    return out


def build(clip_name: str, out: Path, web: bool = False) -> Path:
    clip = Clip(clip_name, web=web)
    if not clip.source_video.exists():
        raise SystemExit(f"no video at {clip.source_video}")
    out.parent.mkdir(parents=True, exist_ok=True)
    if web:
        clip.video_override = web_video(clip.source_video, out.with_suffix(".mp4"))
    rr.init(APP_ID, recording_id=clip.name)
    rr.save(out)
    rr.send_recording_name(clip.title)

    # Layout first, so the hosted viewer shows it while the rest of the file is still loading
    present = [layer for layer in LAYERS if clip.has(*layer.FILES)]
    tabs = [layer.tab(clip) for layer in TAB_ORDER if layer in present]
    rr.send_blueprint(blueprint(clip, tabs, has_now=steps in present), make_active=True, make_default=True)

    summaries = {}
    for layer in LAYERS:
        name = layer.__name__.rsplit(".", 1)[1]
        if layer not in present:
            print(f"  skip {name}: files not there")
            continue
        summaries.update(layer.log(clip))
        print(f"  {name}")
    overview(clip, summaries)
    rr.disconnect()
    return out
