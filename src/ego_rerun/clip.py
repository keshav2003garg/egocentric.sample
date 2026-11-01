"""One clip: where its finished files live, its notes, and the helpers every layer shares."""
import json
import re
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import rerun.blueprint.encodings as rrbe

DATA_ROOT = Path.home() / "Developer/ego_sample"
NOTES_ROOT = Path(__file__).resolve().parents[2] / "clips"
FPS = 50

HAND_COLOURS = {"left": [60, 200, 90], "right": [40, 140, 255]}
# Objects take these in the order the masks list them, so each clip's legend stays short and distinct
PALETTE = [("Yellow", [255, 210, 0]), ("Red", [240, 60, 60]), ("Purple", [200, 90, 220]), ("Orange", [255, 140, 0]),
           ("Cyan", [0, 220, 220]), ("Pink", [255, 120, 180]), ("White", [235, 235, 235])]
GREY = [200, 200, 200]
HIDDEN = [170, 170, 170]


@dataclass
class Clip:
    name: str
    web: bool = False  # smaller file for the hosted site: lighter video, depth and hand surfaces
    video_override: Path | None = None
    dir: Path = field(init=False)
    notes: dict = field(init=False)

    def __post_init__(self):
        self.dir = DATA_ROOT / self.name
        notes = NOTES_ROOT / f"{self.name}.json"
        self.notes = json.loads(notes.read_text()) if notes.exists() else {}

    @property
    def title(self) -> str:
        return self.notes.get("title", self.name)

    @property
    def source_video(self) -> Path:
        return self.dir / f"{self.name}.mp4"

    @property
    def video(self) -> Path:
        return self.video_override or self.source_video

    def has(self, *names: str) -> bool:
        return all((self.dir / n).exists() for n in names)

    def json(self, name: str) -> dict:
        return json.loads((self.dir / name).read_text())

    def label(self, key: str) -> str:
        return self.notes.get("objects", {}).get(key, key.replace("_", " ").capitalize())

    @cached_property
    def lens(self) -> dict:
        return self.json("camera_info.json")["lens"]

    @cached_property
    def objects(self) -> list[str]:
        """Hands first, then the objects in mask order."""
        if self.has("masks.npz"):
            return [str(n) for n in np.load(self.dir / "masks.npz")["names"]]
        return ["left_hand", "right_hand"]

    @cached_property
    def palette(self) -> dict:
        """Object key -> (colour name, RGB)."""
        found = {"left_hand": ("Green", HAND_COLOURS["left"]), "right_hand": ("Blue", HAND_COLOURS["right"])}
        others = [n for n in self.objects if n not in found]
        return found | {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(others)}

    def colour(self, key: str) -> list:
        return self.palette.get(key, ("Grey", GREY))[1]

    def short(self, key: str) -> str:
        """Lower-case name without the bracketed note, e.g. 'shoe 1'."""
        return self.label(key).split(" (")[0].lower()

    @property
    def surface(self) -> str:
        """What the work happens on, for 'height above the …' wording."""
        return self.notes.get("surface", "table")

    @property
    def capture_date(self) -> str:
        if "capture_date" in self.notes:
            return self.notes["capture_date"]
        match = re.search(r"_(\d{4})(\d{2})(\d{2})\d{6}_", self.json("camera_info.json").get("source_recording", ""))
        return "-".join(match.groups()) if match else "–"


def times(frames) -> list:
    """Frame index and seconds, the two timelines every layer uses."""
    frames = np.asarray(frames)
    return [rr.TimeColumn("frame", sequence=frames), rr.TimeColumn("time", duration=frames / FPS)]


def plot(path: str, frames, values, colour, name: str):
    """One line in a time-series view; NaN samples are left out."""
    values = np.asarray(values, dtype=float)
    keep = np.isfinite(values)
    if not keep.any():
        return
    rr.log(path, rr.SeriesLines(colors=[colour], names=[name]), static=True)
    rr.send_columns(path, indexes=times(np.asarray(frames)[keep]), columns=rr.Scalars.columns(scalars=values[keep]))


def log_view(origin: str, name: str) -> rrb.TextLogView:
    """A text log showing only the time and the text (no entity path or level columns)."""
    return rrb.TextLogView(origin=origin, name=name, columns=rrb.TextLogColumns(
        timeline_columns=[rrbe.TimelineColumn("time"), rrbe.TimelineColumn("frame", visible=False)],
        text_log_columns=[rrbe.TextLogColumn("EntityPath", visible=False), rrbe.TextLogColumn("LogLevel", visible=False),
                          rrbe.TextLogColumn("Body")]))


def note(path: str, markdown: str):
    rr.log(path, rr.TextDocument(markdown.strip() + "\n", media_type=rr.MediaType.MARKDOWN), static=True)
