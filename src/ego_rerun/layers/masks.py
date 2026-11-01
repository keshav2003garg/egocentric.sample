"""Per-pixel masks for the hands and each object, laid over the video."""
import numpy as np
import pandas as pd
import rerun as rr
import rerun.blueprint as rrb

from ego_rerun.clip import FPS, Clip, note, plot

FILES = ("masks.npz", "masks.parquet")


def log(clip: Clip) -> dict | None:
    if not clip.has(*FILES):
        return None
    z = np.load(clip.dir / "masks.npz")
    labels, order = z["labels"], [str(n) for n in z["names"]]
    table = pd.read_parquet(clip.dir / "masks.parquet")
    scale = clip.lens["width"] / labels.shape[2]

    # Id 0 stays see-through; the mask grid is scaled up onto the full-size video
    rr.log("video", rr.AnnotationContext(
        [rr.AnnotationInfo(id=0, label="none", color=[0, 0, 0, 0])]
        + [rr.AnnotationInfo(id=i, label=clip.label(n), color=clip.colour(n)) for i, n in enumerate(order, 1)]),
        static=True)
    rr.log("video/masks", rr.Transform3D(scale=scale), static=True)
    for k in range(labels.shape[0]):
        rr.set_time("frame", sequence=k)
        rr.set_time("time", duration=k / FPS)
        rr.log("video/masks", rr.SegmentationImage(labels[k], opacity=0.4, draw_order=10.0))

    rows = []
    for n in order:
        t = table[table["object"] == n]
        plot(f"plots/masks/confidence/{n}", t["frame"], t["score"], clip.colour(n), clip.label(n))
        seen = int(t["visible"].sum())
        rows.append(f"| {clip.label(n)} | {seen} | {t.loc[t['visible'], 'score'].median():.2f} |")

    if clip.has("masks_report.json"):
        how = ("SAM 3.1, found by a text prompt per object (picks checked by eye), then followed forward and backward "
               "through the clip by its video memory; an object that is lost is looked for again every 10 frames.")
    else:
        how = ("SAM 3.1, prompted once per object by text, then followed through the clip by its video memory. Objects "
               "are never re-detected, so the two identical shoes keep their identities.")
    note("notes/masks", f"""
### Masks
{how}

| Object | Frames visible (of {labels.shape[0]}) | Median confidence |
|---|---|---|
""" + "\n".join(rows))
    objects = len([n for n in order if not n.endswith("_hand")])
    return {"Masks": f"Per-pixel masks for both hands and {objects} object{'s' if objects != 1 else ''}"}


def tab(clip: Clip):
    return rrb.Vertical(
        rrb.TimeSeriesView(origin="plots/masks/confidence", name="Mask confidence"),
        rrb.TextDocumentView(origin="notes/masks", name="About"),
        row_shares=[2, 3], name="Masks")
