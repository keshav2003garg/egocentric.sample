"""Which hand touches or holds what, with which fingers and grip, and what the tool is doing.

Two forms of the data. Clip 004's was measured against 3D object models (not shown in the viewer), with thumb and
index contact points and the brush tip's state (painting / dipping). The other clips' is judged against each object's
visible surface from its mask and depth, and the tool's use is read from what it touches (held / in use on an object).
"""
import numpy as np
import pandas as pd
import rerun as rr
import rerun.blueprint as rrb

from ego_rerun.clip import FPS, HAND_COLOURS, Clip, log_view, note, plot, times

GRIPS = {"Pow-Circ": "power grip", "Pow-Pris": "power grip", "Pre-Pris": "precision grip", "Pre-Circ": "precision grip",
         "Lat": "lateral grip", "NP-Palm": "palm push", "NP-Fin": "fingertip push"}
COARSE_GRIPS = {"power": "power grip", "precision": "precision grip", "intermediate": "intermediate grip"}
STATE_LEVEL = {"none": 0, "touching": 1, "holding": 2}
# Clip 004's brush tip
TOOL_LEVEL = {"idle": 0, "approach": 1, "dip": 2, "stroke": 3}
TOOL_WORDS = {"idle": "not in use", "approach": "approaching", "dip": "dipping", "stroke": "painting"}
# Other clips' tools
USE_LEVEL = {"not held": 0, "held": 1, "resting": 1, "in use": 2}
CONTACT_DOT = [255, 70, 70]
FINGER_TIPS = {0: 4, 1: 9}  # finger_contact part -> fingertip in the 25 hand joints: thumb, index
STEADY_FRAMES = 10  # a fingertip dot appears or goes only after 0.2 s


def steady(on: np.ndarray) -> np.ndarray:
    """On/off with gaps and blips shorter than STEADY_FRAMES removed (gaps closed first, then blips dropped)."""
    out = on.copy()
    for value in (False, True):
        edges = np.flatnonzero(np.diff(np.r_[0, (out == value).astype(int), 0]))
        for a, b in zip(edges[::2], edges[1::2]):
            if b - a < STEADY_FRAMES and a > 0 and b < len(out):
                out[a:b] = not value
    return out
# One wording for contact events in every clip (004's data names them differently), in the order they happen
EVENT_WORDS = {"approach": "approach", "contact_start": "touch start", "touch_start": "touch start",
               "grasp_start": "grasp", "grasp": "grasp", "release": "release", "contact_end": "touch end",
               "touch_end": "touch end", "retreat": "retreat", "tool_start": "tool on", "tool_end": "tool off"}
EVENT_ORDER = {w: i for i, w in enumerate(dict.fromkeys(EVENT_WORDS.values()))}

FILES = ("contact.parquet", "contact_events.parquet", "contact_meta.json")


def detailed(clip: Clip) -> bool:
    return "per_hand" not in clip.json("contact_meta.json")["qc"]


def grip_words(row) -> str:
    feix = getattr(row, "grasp_feix", None)
    if isinstance(feix, str) and feix not in ("", "none"):
        return feix.lower()
    return GRIPS.get(getattr(row, "grasp_hands23", None), "") or COARSE_GRIPS.get(getattr(row, "grasp_rule_coarse", None), "")


def hand_parts(clip: Clip, row) -> tuple[str, str]:
    """What one hand does on one frame, and its grip: ('holding shoe 1', 'power grip'); grip '' when not holding."""
    if row.state == "out_of_view":
        return "at the frame edge or out of view", ""
    if row.state == "holding":
        also = getattr(row, "also_holding", "none")
        what = f"holding {clip.short(row.held_object)}" + (f" and {clip.short(also)}" if also not in ("none", None) else "")
        if not getattr(row, "hand_visible", True):
            return f"{what} (hand at the frame edge or out of view)", ""
        return what, grip_words(row)
    if row.state == "touching":
        return f"touching {clip.short(row.object)}", ""
    return "free", ""


def hand_status(clip: Clip, row) -> str:
    """Short phrase for one hand on one frame, e.g. 'holding shoe 1 · power grip'."""
    what, grip = hand_parts(clip, row)
    return what + (f" · {grip}" if grip else "")


def tool_phrase(clip: Clip, state: str) -> str:
    """'in use on primer_cup' -> 'in use on primer cup'."""
    for verb in ("in use on ", "resting on "):
        if state.startswith(verb):
            return verb + clip.short(state[len(verb):])
    return state


def tool_states(clip: Clip) -> dict:
    """Frame -> (tool name, what it's doing), for the "now" panel. Empty when the clip has no tool."""
    if not clip.has("tool_contact.parquet"):
        return {}
    tool = pd.read_parquet(clip.dir / "tool_contact.parquet")
    if "tool" not in tool.columns:
        return {int(f): ("Brush", TOOL_WORDS.get(s, "")) for f, s in zip(tool["frame"], tool["state"])}
    return {int(r.frame): (clip.label(r.tool), tool_phrase(clip, r.state)) for r in tool.itertuples()}


def use_level(state: str) -> int:
    return next((v for k, v in USE_LEVEL.items() if state.startswith(k)), 0)


def log(clip: Clip) -> dict | None:
    if not clip.has(*FILES):
        return None
    contact = pd.read_parquet(clip.dir / "contact.parquet")
    events = pd.read_parquet(clip.dir / "contact_events.parquet")
    meta = clip.json("contact_meta.json")
    full = detailed(clip)

    for hand in ("left", "right"):
        t = contact[contact["hand"] == hand].sort_values("frame")
        frames = t["frame"].to_numpy()
        colour = HAND_COLOURS[hand]

        # Thumb and index touching, as dots on the 3D hand's fingertips (the saved contact points are the nearest spot
        # on the object model, which jumps from frame to frame); only while that finger touches, without blips
        if full:
            joints = pd.read_parquet(clip.dir / "hands.parquet", columns=["frame", "hand", "joints25_world"])
            joints = joints[joints["hand"] == hand].set_index("frame")["joints25_world"]
            fingers = np.stack(t["finger_contact"].to_numpy())
            on = {tip: steady(fingers[:, part] == "touch") for part, tip in FINGER_TIPS.items()}
            positions, lengths = [], []
            for i, k in enumerate(frames):
                j = np.asarray(joints.get(k, np.full(75, np.nan)), np.float32).reshape(25, 3)
                frame_pts = [j[tip] for tip in FINGER_TIPS.values() if on[tip][i] and np.all(np.isfinite(j[tip]))]
                positions += frame_pts
                lengths.append(len(frame_pts))
            rr.log(f"world/hands/{hand}/contact_points", rr.Points3D.from_fields(colors=[CONTACT_DOT], radii=0.004),
                   static=True)
            rr.send_columns(f"world/hands/{hand}/contact_points", indexes=times(frames),
                            columns=rr.Points3D.columns(positions=np.array(positions or np.zeros((0, 3)), np.float32))
                            .partition(lengths))

        label = f"{hand} hand"
        plot(f"plots/contact/state/{hand}", frames, t["state"].map(STATE_LEVEL), colour, label)  # out of view: gap
        touching = np.array([sum(f == "touch" for f in fc) for fc in t["finger_contact"]], float)
        if "hand_visible" in t:
            touching[~t["hand_visible"].to_numpy(bool)] = np.nan
        plot(f"plots/contact/fingers/{hand}", frames, touching, colour, label)
        plot(f"plots/contact/confidence/{hand}", frames, t["contact_conf"], colour, label)

    if clip.has("tool_contact.parquet"):
        tool = pd.read_parquet(clip.dir / "tool_contact.parquet").sort_values("frame")
        if full:
            plot("plots/contact/tool/brush", tool["frame"], tool["state"].map(TOOL_LEVEL), clip.colour("brush"), "brush tip")
        else:
            for name, t in tool.groupby("tool"):
                plot(f"plots/contact/tool/{name}", t["frame"], t["state"].map(use_level), clip.colour(name),
                     clip.short(name))

    # Contact events as a log
    events["words"] = events["event"].map(lambda e: EVENT_WORDS.get(e, e.replace("_", " ")))
    events["order"] = events["words"].map(EVENT_ORDER).fillna(len(EVENT_ORDER))
    for e in events.sort_values(["frame", "order"]).itertuples():
        rr.set_time("frame", sequence=int(e.frame))
        rr.set_time("time", duration=e.frame / FPS)
        rr.log("logs/contact_events", rr.TextLog(f"{e.hand} hand · {e.words} · {clip.short(e.object)}"))
    rr.reset_time()

    note("notes/contact", (detailed_note if full else visible_surface_note)(clip, meta, contact))
    tools = " and ".join(clip.short(t) for t in meta.get("tool_summary", {})) if not full else "brush strokes and dips"
    return {"Contact": "Touch, hold, fingers, grip type and role per hand" + (f"; {tools}" if tools else "")}


def plural(n, word: str) -> str:
    return "–" if n is None else f"{n} {word}{'' if n == 1 else 's'}"


def detailed_note(clip: Clip, meta: dict, contact: pd.DataFrame) -> str:
    qc = meta["qc"]
    tool_summary = meta.get("tool_summary", {})
    rim = f", {plural(tool_summary['n_rim_touches'], 'touch')} on its rim" if tool_summary.get("n_rim_touches") else ""
    rim = rim.replace("touchs", "touches")
    return f"""
### Contact and grasp
For each hand on every frame: what it touches, what it holds, which fingers are in contact, the grip type and the hand's role.

**How it's found:** the 3D hand surface is measured against a 3D model of each object fitted to every frame, then checked against three independent cues: the hand and object outlines meet in the image, there is no depth gap between them, and the object moves with the hand. A smoothing step removes flicker, and a hand can hold only one object at a time.

| Check | Result |
|---|---|
| Objects moving with no hand touching them | {sum(qc['object_moving_without_contact_frames'].values())} frames |
| Held object swapped mid-hold | {qc['held_identity_swaps']} |
| Grip type: rules vs. visual labels | {qc['rule_vs_visual_grasp_agreement_hands23']:.0%} agree ({qc['rule_vs_visual_compared']} samples) |
| Hand surface inside an object | median {qc['penetration_mm_p50_p95'][0]} mm, 95th percentile {qc['penetration_mm_p50_p95'][1]} mm |

**Brush:** {plural(tool_summary.get('n_strokes'), 'painting stroke')} ({tool_summary.get('total_painting_time_s', '–')} s), {plural(tool_summary.get('n_dips'), 'dip')} in the paint pot{rim}.

**Grips:** left hand, power grip on the shoe; right hand, writing tripod (pen grip) on the brush.

**Fingers:** thumb, index, middle, ring, pinky and palm are each marked touch / no / unknown; "unknown" when the finger is hidden from the camera.

Force is not measured; how far the hand surface sits inside an object (above) is an uncalibrated estimate.
"""


def visible_surface_note(clip: Clip, meta: dict, contact: pd.DataFrame) -> str:
    qc = meta["qc"]
    n = contact["frame"].nunique()
    rows = []
    for hand in ("left", "right"):
        h = qc["per_hand"][hand]
        held = ", ".join(f"{clip.short(k)} {v}" for k, v in h["held"].items() if v) or "–"
        touched = ", ".join(f"{clip.short(k)} {v}" for k, v in h["frames_touching_by_object"].items() if v) or "–"
        also = ", ".join(f"{clip.short(k)} {v}" for k, v in h.get("also_held", {}).items() if v) or "–"
        rows.append(f"| {hand.capitalize()} | {h['frames_holding']} | {held} | {also} | {touched} | {h.get('frames_out_of_view', 0)} |")
    tools = []
    for name, t in meta.get("tool_summary", {}).items():
        uses = ", ".join(f"{clip.short(k)} {v / FPS:.1f} s" for k, v in t["frames_in_use_by_object"].items() if v >= 5)
        tools.append(f"**{clip.label(name)}:** held for {t['frames_held'] / FPS:.1f} s; in use on {uses or '–'}.")
    check = meta.get("hand_check", {})
    spot = (f"| Spot check of state and object | {check['state_and_object_correct']:.0%} right on "
            f"{check['hand_frames_scored']} hand-frames |" if check else "")
    roles = f"**In this clip:** {clip.notes['contact_roles']}\n\n" if clip.notes.get("contact_roles") else ""
    return f"""
### Contact and grasp
For each hand on every frame: what it touches, what it holds, which fingers and the palm are in contact, the grip type and the hand's role.

**How it's found:** the 3D hand surface is measured against each object's visible surface, taken from its mask and the depth. A hand touches an object when they come within 15 mm where their outlines meet in the image, or when it is partly behind the object (holding it from below). It holds a movable object when it grips it (thumb plus another part, or any three parts touching), or carries it from below as the object moves with the wrist. When the fingers are on a lace or tool, that is named rather than what it lies on. A smoothing step removes flicker. A hand holds one object at a time, except that a hand using a tool can also steady the object it works on ("holding marker and shoe"); that second hold is listed under "Also held". There are no 3D object models in this clip, so contact is judged against the surface the camera sees next to the fingers.

{roles}**Out of view:** where the 3D hand isn't drawn (out of the picture, or too far into its edge), contact isn't judged and the hand shows as out of view. The 3D hand is also left undrawn on short spans where it lands off the real hand; contact there is still judged from the hand data, so those frames aren't counted as out of view. A hold carries on through such a gap, or while a tool is hidden in the fist, when the same object is held on both sides and isn't seen apart from the hand.

| Hand | Frames holding (of {n}) | Held, frames | Also held, frames | Touching, frames | Out of view, frames |
|---|---|---|---|---|---|
{chr(10).join(rows)}

| Check | Result |
|---|---|
| Flicker runs under 3 frames | {sum(qc['flicker_runs_under_3_frames'].values())} |
| Hand-frames worth a second look (fingers hidden or borderline) | {qc['review_flag_frames']} |
{spot}

{chr(10).join(tools)}

**Fingers:** thumb, index, middle, ring, pinky and palm are each marked touch / no / unknown; "unknown" when the finger is hidden from the camera.

Force is not measured.
"""


def tab(clip: Clip):
    views = [
        rrb.TimeSeriesView(origin="plots/contact/state", name="Hand state: 0 free · 1 touching · 2 holding"),
        rrb.TimeSeriesView(origin="plots/contact/fingers", name="Fingers touching (of 6)"),
    ]
    if clip.has("tool_contact.parquet"):
        name = ("Brush: 0 idle · 1 approach · 2 dip · 3 painting" if detailed(clip)
                else "Tool: 0 not held · 1 held · 2 in use")
        views.append(rrb.TimeSeriesView(origin="plots/contact/tool", name=name))
    # the events log and the note side by side under the plots, so the log gets a full half-height
    below = rrb.Horizontal(log_view("logs/contact_events", "Contact events"),
                           rrb.TextDocumentView(origin="notes/contact", name="About"), column_shares=[1, 1])
    return rrb.Vertical(*views, below, row_shares=[1] * len(views) + [len(views) + 1], name="Contact")
