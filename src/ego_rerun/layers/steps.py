"""The task broken into timed action steps, and a live "now" panel: current step plus what each hand is doing."""
import pandas as pd
import rerun as rr
import rerun.blueprint as rrb

from ego_rerun.clip import FPS, Clip, log_view, note
from ego_rerun.layers.contact import detailed, hand_parts, tool_states

FILES = ("action_steps.parquet", "task.json")
STEADY_FRAMES = 10  # the hands part of the "Now" panel: a state shorter than 0.2 s is a blip...
PASSING_FRAMES = 5  # ...unless it sits between two different states and lasts this long (a let-go between two holds)


def settle(values: list) -> list:
    """Short states between two steady ones are smoothed away: where the line comes back to the same state they are a
    blip and the steady state stays; where it moves on to another state and they last PASSING_FRAMES, the longest of
    them is shown once in between (holding A, let go, touching B), so the change is never hidden."""
    edges = [0] + [k for k in range(1, len(values)) if values[k] != values[k - 1]] + [len(values)]
    runs = list(zip(edges[:-1], edges[1:]))
    steady = [i for i, (a, b) in enumerate(runs) if i == 0 or b - a >= STEADY_FRAMES or b == len(values)]
    shown = list(values)
    for i, j in zip(steady, steady[1:]):
        if j == i + 1:
            continue
        a, b = runs[i + 1][0], runs[j][0]
        before, after = values[runs[i][0]], values[runs[j][0]]
        if before == after or b - a < PASSING_FRAMES:
            shown[a:b] = [before] * (b - a)
        else:
            longest = max(runs[i + 1:j], key=lambda r: r[1] - r[0])
            shown[a:b] = [values[longest[0]]] * (b - a)
    return shown


def at_frame(frame: int):
    rr.set_time("frame", sequence=frame)
    rr.set_time("time", duration=frame / FPS)


def log(clip: Clip) -> dict | None:
    if not clip.has(*FILES):
        return None
    steps = pd.read_parquet(clip.dir / "action_steps.parquet")
    task = clip.json("task.json")
    main = steps[steps["level"] == "step"].sort_values("start_frame")
    sub = steps[steps["level"] == "substep"].sort_values("start_frame")
    n = int(main["end_frame"].max()) + 1
    # A "sub-step" spanning most of the clip isn't a finer step (024's one covers all of it)
    sub = sub[sub["end_frame"] - sub["start_frame"] < 0.8 * n]

    # Step list as a log: one entry as each step starts
    for s in pd.concat([main, sub]).sort_values(["start_frame", "level"]).itertuples():
        at_frame(int(s.start_frame))
        prefix = f"{int(s.step_id) + 1}. " if s.level == "step" else "  · "
        rr.log("logs/steps", rr.TextLog(f"{prefix}{s.name[0].upper()}{s.name[1:]}  ({s.start_s:.2f}–{s.end_s:.2f} s)"))

    # "Now" panel, in two parts. The step: rewritten only when the step changes
    for _, s in main.iterrows():
        at_frame(int(s.start_frame))
        rr.log("notes/now_step", rr.TextDocument(
            f"#### Step {int(s.step_id) + 1} of {len(main)} · {s['name'][0].upper()}{s['name'][1:]}\n{s['description']}\n",
            media_type=rr.MediaType.MARKDOWN))
    # The hands: the same lines every frame (nothing reflows), with short states smoothed as in settle() (the data is
    # already cleaned of blips; this keeps the panel calm where two parts change close together)
    contact = pd.read_parquet(clip.dir / "contact.parquet") if clip.has("contact.parquet") else None
    tools = tool_states(clip)
    by_hand = {h: contact[contact["hand"] == h].set_index("frame") for h in ("left", "right")} if contact is not None else {}
    # each line (each hand, each hand's grip, the tool) settles on its own, so one hand's change never holds back the other's
    cols = {key: [] for key in ("left", "right", "left_grip", "right_grip", "tool")}
    tool_name = next(iter(tools.values()))[0] if tools else ""
    for k in range(n):
        step = main[(main["start_frame"] <= k) & (main["end_frame"] >= k)]
        s = step.iloc[0] if not step.empty else main.iloc[-1]
        parts = {h: hand_parts(clip, row_of(t, k)) for h, t in by_hand.items() if k in t.index}
        for h in ("left", "right"):
            cols[h].append(parts[h][0] if h in parts else s[f"{h}_hand"])
            cols[f"{h}_grip"].append(parts[h][1] if h in parts else "")
        cols["tool"].append(tools.get(k, (tool_name, ""))[1])
    steady = {key: settle(v) for key, v in cols.items()}
    # grip words only where they come from the detailed per-frame grip labels (the coarse rule elsewhere reads too flat)
    show_grip = contact is not None and detailed(clip)
    shown = []
    for k in range(n):
        grips = [f"{h} {steady[f'{h}_grip'][k]}" for h in ("left", "right") if steady[f"{h}_grip"][k]]
        lines = [f"- **Left hand:** {steady['left'][k]}", f"- **Right hand:** {steady['right'][k]}"]
        if show_grip:
            lines.append(f"- **Grip:** {' · '.join(grips) if grips else '–'}")
        if tools:
            lines.append(f"- **{tool_name}:** {steady['tool'][k] or '–'}")
        shown.append("\n".join(lines) + "\n")
    for k in range(n):
        if k == 0 or shown[k] != shown[k - 1]:
            at_frame(k)
            rr.log("notes/now_hands", rr.TextDocument(shown[k], media_type=rr.MediaType.MARKDOWN))
    rr.reset_time()

    rows = "\n".join(f"| {int(s.step_id) + 1} | {s.start_s:.2f}–{s.end_s:.2f} s | {s['name'][0].upper()}{s['name'][1:]} |"
                     for _, s in main.iterrows())
    repeats = {k: v for k, v in task.get("counts", {}).items() if v > 1}
    counts = " **Counts:** " + ", ".join(f"{k.replace('_', ' ')} ×{v}" for k, v in repeats.items()) + "." if repeats else ""
    finer = (f"{len(sub)} sub-step{'s mark' if len(sub) != 1 else ' marks'} each bout of tool use within the steps."
             if len(sub) else "")
    lo, hi = (int(v) for v in pd.concat([main, sub])["boundary_uncertainty_frames"].agg(["min", "max"]))
    spread = f"±{lo}" if lo == hi else f"±{lo}–{hi}"
    spread_s = f"{lo / FPS:.2f}" if lo == hi else f"{lo / FPS:.2f}–{hi / FPS:.2f}"
    tool_words = ", tool" if clip.has("tool_contact.parquet") else ""
    note("notes/steps", f"""
### Action steps
**Instruction:** {task['instruction']}

**Hands:** left {task['hands']['left']}; right {task['hands']['right']}.

| # | Time | Step |
|---|---|---|
{rows}

{finer}{counts}

**Outcome:** {task.get('outcome', '–')}.

Step start and end times come from the contact{tool_words} and 3D hand data where they show the change, otherwise from frames checked by eye; each is accurate to within {hi / FPS:.2f} s. Step names and descriptions are written from frames of the video and checked against those timings.
""")
    and_sub = f" and {len(sub)} sub-step{'s' if len(sub) != 1 else ''}" if len(sub) else ""
    return {"Action steps": f"{len(main)} timed steps with descriptions{and_sub}"}


def row_of(table: pd.DataFrame, frame: int):
    """One frame's row as a named tuple, the shape contact.hand_parts expects."""
    return next(table.loc[[frame]].itertuples())


def tab(clip: Clip):
    return rrb.Vertical(
        log_view("logs/steps", "Steps"),
        rrb.TextDocumentView(origin="notes/steps", name="About"),
        row_shares=[2, 3], name="Steps")
