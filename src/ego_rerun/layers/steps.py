"""The task broken into timed action steps, and a live "now" panel: current step plus what each hand is doing."""
import pandas as pd
import rerun as rr
import rerun.blueprint as rrb

from ego_rerun.clip import FPS, Clip, log_view, note
from ego_rerun.layers.contact import hand_parts, tool_states

FILES = ("action_steps.parquet", "task.json")
STEADY_FRAMES = 10  # the hands part of the "Now" panel shows a change once it has lasted 0.2 s


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
    # The hands: the same lines every frame (nothing reflows), and a change shown only once it has lasted
    # STEADY_FRAMES (the data is already cleaned of blips; this keeps the panel calm where two parts change close together)
    contact = pd.read_parquet(clip.dir / "contact.parquet") if clip.has("contact.parquet") else None
    tools = tool_states(clip)
    by_hand = {h: contact[contact["hand"] == h].set_index("frame") for h in ("left", "right")} if contact is not None else {}
    texts = []
    for k in range(n):
        step = main[(main["start_frame"] <= k) & (main["end_frame"] >= k)]
        s = step.iloc[0] if not step.empty else main.iloc[-1]
        parts = {h: hand_parts(clip, row_of(t, k)) for h, t in by_hand.items() if k in t.index}
        what = {h: parts[h][0] if h in parts else s[f"{h}_hand"] for h in ("left", "right")}
        grips = [f"{h} {parts[h][1]}" for h in ("left", "right") if h in parts and parts[h][1]]
        lines = [f"- **Left hand:** {what['left']}", f"- **Right hand:** {what['right']}",
                 f"- **Grip:** {' · '.join(grips) if grips else '–'}"]
        if tools:
            name, phrase = tools.get(k, (next(iter(tools.values()))[0], ""))
            lines.append(f"- **{name}:** {phrase or '–'}")
        texts.append("\n".join(lines) + "\n")
    shown = texts[:1]
    for k in range(1, n):
        lasts = all(t == texts[k] for t in texts[k:k + STEADY_FRAMES]) or k + STEADY_FRAMES > n
        shown.append(texts[k] if texts[k] != shown[-1] and lasts else shown[-1])
    for k in range(n):
        if k == 0 or shown[k] != shown[k - 1]:
            at_frame(k)
            rr.log("notes/now_hands", rr.TextDocument(shown[k], media_type=rr.MediaType.MARKDOWN))
    rr.reset_time()

    rows = "\n".join(f"| {int(s.step_id) + 1} | {s.start_s:.2f}–{s.end_s:.2f} s | {s['name'][0].upper()}{s['name'][1:]} |"
                     for _, s in main.iterrows())
    repeats = {k: v for k, v in task.get("counts", {}).items() if v > 1}
    counts = " **Counts:** " + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in repeats.items()) + "." if repeats else ""
    finer = (f"{len(sub)} finer sub-step{'s mark' if len(sub) != 1 else ' marks'} repeated actions within the steps."
             if len(sub) else "")
    lo, hi = (int(v) for v in pd.concat([main, sub])["boundary_uncertainty_frames"].agg(["min", "max"]))
    spread = f"±{lo}" if lo == hi else f"±{lo}–{hi}"
    spread_s = f"{lo / FPS:.2f}" if lo == hi else f"{lo / FPS:.2f}–{hi / FPS:.2f}"
    note("notes/steps", f"""
### Action steps
**Instruction:** {task['instruction']}

**Hands:** left {task['hands']['left']}; right {task['hands']['right']}.

| # | Time | Step |
|---|---|---|
{rows}

{finer}{counts}

**Outcome:** {task.get('outcome', '–')}.

Step start and end frames come from the contact, tool and 3D hand data where they show the change, otherwise from keyframes; each is accurate to {spread} frames ({spread_s} s). Step names and descriptions are written by a vision-language model from keyframes and checked against those timings.
""")
    and_sub = f" and {len(sub)} sub-step{'s' if len(sub) != 1 else ''}" if len(sub) else ""
    return {"Action steps": f"{len(main)} timed steps{and_sub} with descriptions"}


def row_of(table: pd.DataFrame, frame: int):
    """One frame's row as a named tuple, the shape contact.hand_parts expects."""
    return next(table.loc[[frame]].itertuples())


def tab(clip: Clip):
    return rrb.Vertical(
        log_view("logs/steps", "Steps"),
        rrb.TextDocumentView(origin="notes/steps", name="About"),
        row_shares=[2, 3], name="Steps")
