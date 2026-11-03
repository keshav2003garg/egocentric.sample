"""Metric depth per frame, shown as a colour image beside the video."""
import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from ego_rerun.clip import Clip, note, times

NEAR_M, FAR_M = 0.2, 0.8  # red at 20 cm or closer, blue at 80 cm or farther
FAR_INDEX = 0.12  # where the far end starts on the TURBO map (below this it runs to near-black)
# depth_report.json "model" -> how the note describes it (the model itself isn't named to buyers);
# "fast" is the pipeline's 06f_depth_fast.py
MODELS = {"fast": "Metric depth, one pass per frame, aligned to the camera path",
          "Depth Anything 3 Metric-Large, one pass per frame": "Metric depth, one pass per frame, aligned to the camera path"}

FILES = ("depth.npz",)


def colourise(depth_mm: np.ndarray) -> np.ndarray:
    d = depth_mm.astype(np.float32) / 1000
    t = np.clip((FAR_M - d) / (FAR_M - NEAR_M), 0, 1)
    # the colour map's dark far end is skipped (it would read as "no depth"): far is plain blue
    rgb = cv2.applyColorMap(((FAR_INDEX + (1 - FAR_INDEX) * t) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    rgb[depth_mm == 0] = 0
    return rgb  # BGR, as cv2.imencode expects


def stats(clip: Clip) -> dict:
    """Depth check numbers: from depth_report.json when the clip has one, else from the clip notes."""
    if not clip.has("depth_report.json"):
        return clip.notes.get("depth", {})
    r = clip.json("depth_report.json")
    if r["model"] not in MODELS:
        raise SystemExit(f"{clip.name}: unknown depth model {r['model']!r} in depth_report.json; add it to depth.MODELS")
    extra = "The far fisheye corners have no depth."
    if "metric_scale_corrected_by_hand_size" in r:
        extra += (" The overall scale was checked against adult hand size and corrected by "
                  f"×{r['metric_scale_corrected_by_hand_size']['factor']:.2f}.")
    if "steady_scale" in r:
        extra += (" Jumps in the overall depth scale between neighbouring frames were removed by comparing the same "
                  f"still scene points (scale jump {r['steady_scale']['jump_rms_pct_before']}% → "
                  f"{r['steady_scale']['jump_rms_pct_after']}% per frame). The flicker figure in the table is a "
                  "different, per-pixel measure, and also counts hands and objects that really move.")
    bands = r["error_after_by_distance_from_centre_pct"]
    return {
        "model": MODELS[r["model"]],
        "coverage_pct": r["pixels_with_depth_pct"],
        "error_vs_sfm_points_pct": r["error_vs_colmap_points_after_pct"],
        "points_within_5pct": r["points_within_5pct_after"],
        "error_by_distance_from_centre_pct": {k.replace("-", "–").replace("px", " px"): v for k, v in bands.items()},
        "flicker_pct": r["flicker_after_pct"],
        "extra": "\n\n" + extra,
    }


def log(clip: Clip) -> dict | None:
    if not clip.has(*FILES):
        return None
    depth = np.load(clip.dir / "depth.npz")["depth_mm"]
    size, quality = ((depth.shape[2] // 2, depth.shape[1] // 2), 70) if clip.web else (None, 88)
    blobs = [cv2.imencode(".jpg", colourise(cv2.resize(d, size, interpolation=cv2.INTER_NEAREST) if size else d),
                          [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes() for d in depth]
    rr.send_columns("depth", indexes=times(np.arange(len(depth))),
                    columns=rr.EncodedImage.columns(blob=blobs, media_type=["image/jpeg"] * len(blobs)))

    s = stats(clip)
    bands = "\n".join(f"| {k} | {v}% |" for k, v in s.get("error_by_distance_from_centre_pct", {}).items())
    checks = f"""
| | |
|---|---|
| Pixels with depth | {s['coverage_pct']}% |
| Error vs structure-from-motion points | {s['error_vs_sfm_points_pct']}% (median) |
| Points within 5% | {s['points_within_5pct']}% |
| Frame-to-frame flicker (per pixel) | {s['flicker_pct']}% |

| Distance from image centre | Error |
|---|---|
{bands}
""" if s else ""
    note("notes/depth", f"""
### Depth
{s.get('model', 'Metric depth')}. Millimetres along the camera's forward axis, on the video's own fisheye pixels at {depth.shape[2]}×{depth.shape[1]}.{s.get('extra', '')}

**Colours:** red = 20 cm or nearer, blue = 80 cm or farther, black = no depth.
{checks}""")
    return {"Depth": f"Metric depth per frame, {s['error_vs_sfm_points_pct']}% median error" if s else "Metric depth per frame"}


def tab(clip: Clip):
    return rrb.Vertical(rrb.TextDocumentView(origin="notes/depth", name="About"), name="Depth")
