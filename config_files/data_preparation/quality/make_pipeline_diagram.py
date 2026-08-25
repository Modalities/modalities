#!/usr/bin/env python3
"""Generates the Data Preprocessing Pipeline diagram as an Excalidraw file.

Matches the visual language of the existing training/pruning diagram: hand-drawn rounded
boxes, orange stage labels, coloured owner badges, grouped containers with titles, and a
section header underlined with '=' characters.

Layout is two converging lanes -- documents along the top, annotations below -- meeting at
the join. That is not decoration: a single row needs arrows that skip past boxes, and those
arrows cross other boxes wherever they go.

Kept as a generator rather than hand-written JSON so the layout can be nudged in one place.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(
    "/home/richard.rutmann/repos/modalities/config_files/data_preparation/quality/"
    "data_preprocessing_pipeline.excalidraw"
)

INK = "#1e1e1e"
STAGE_TEXT = "#e8590c"
ARTIFACT_TEXT = "#0c8599"
NOTE_TEXT = "#868e96"
GUARD_TEXT = "#2f9e44"

# Owner badge palette, keyed by the initial used in the existing diagram.
BADGES = {
    "R": ("#1971c2", "#a5d8ff"),
    "T": ("#2f9e44", "#b2f2bb"),
    "S": ("#f08c00", "#ffec99"),
    "H": ("#e03131", "#ffc9c9"),
    "M": ("#099268", "#96f2d7"),
}

elements: list[dict] = []
_counter = [0]


def _next_id() -> str:
    _counter[0] += 1
    return f"el{_counter[0]:04d}"


def _base(kind: str, x: float, y: float, w: float, h: float, **over) -> dict:
    element = {
        "id": _next_id(),
        "type": kind,
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "angle": 0,
        "strokeColor": INK,
        "backgroundColor": "transparent",
        "fillStyle": "solid",
        "strokeWidth": 1,
        "strokeStyle": "solid",
        "roughness": 1,
        "opacity": 100,
        "groupIds": [],
        "frameId": None,
        "roundness": {"type": 3},
        "seed": 100_000 + _counter[0] * 7919,
        "version": 1,
        "versionNonce": 200_000 + _counter[0] * 104_729,
        "isDeleted": False,
        "boundElements": None,
        "updated": 1,
        "link": None,
        "locked": False,
    }
    element.update(over)
    return element


def text(content: str, x: float, y: float, size: int = 14, color: str = INK) -> dict:
    lines = content.split("\n")
    element = _base(
        "text", x, y,
        max(len(line) for line in lines) * size * 0.55, len(lines) * size * 1.25,
        strokeColor=color, roundness=None,
    )
    element.update({
        "text": content, "originalText": content, "fontSize": size, "fontFamily": 1,
        "textAlign": "left", "verticalAlign": "top", "containerId": None,
        "lineHeight": 1.25, "autoResize": True,
    })
    elements.append(element)
    return element


def box(label: str, x: float, y: float, w: float, h: float, color: str = STAGE_TEXT,
        dashed: bool = False, size: int = 12) -> dict:
    """A rounded rectangle with centred bound text."""
    rect = _base("rectangle", x, y, w, h, strokeStyle="dashed" if dashed else "solid")
    lines = label.split("\n")
    label_element = _base(
        "text", x + 6, y + (h - len(lines) * size * 1.25) / 2, w - 12, len(lines) * size * 1.25,
        strokeColor=color, roundness=None,
    )
    label_element.update({
        "text": label, "originalText": label, "fontSize": size, "fontFamily": 1,
        "textAlign": "center", "verticalAlign": "middle", "containerId": rect["id"],
        "lineHeight": 1.25, "autoResize": False,
    })
    rect["boundElements"] = [{"type": "text", "id": label_element["id"]}]
    elements.append(rect)
    elements.append(label_element)
    return rect


def container(title: str, x: float, y: float, w: float, h: float) -> dict:
    rect = _base("rectangle", x, y, w, h)
    elements.append(rect)
    text(title, x + 14, y + 10, size=13)
    return rect


def badge(letter: str, cx: float, cy: float, r: float = 11) -> None:
    stroke, fill = BADGES[letter]
    ellipse = _base("ellipse", cx - r, cy - r, r * 2, r * 2,
                    strokeColor=stroke, backgroundColor=fill, roundness=None)
    elements.append(ellipse)
    label = _base("text", cx - r + 2, cy - 7, r * 2 - 4, 14, strokeColor=stroke, roundness=None)
    label.update({
        "text": letter, "originalText": letter, "fontSize": 11, "fontFamily": 1,
        "textAlign": "center", "verticalAlign": "middle", "containerId": ellipse["id"],
        "lineHeight": 1.25, "autoResize": False,
    })
    ellipse["boundElements"] = [{"type": "text", "id": label["id"]}]
    elements.append(label)


def arrow(*waypoints: tuple[float, float], dashed: bool = False, color: str = INK) -> None:
    """An arrow through a sequence of absolute points, so routes can dodge boxes."""
    x0, y0 = waypoints[0]
    points = [[x - x0, y - y0] for x, y in waypoints]
    element = _base(
        "arrow", x0, y0,
        max(abs(p[0]) for p in points), max(abs(p[1]) for p in points),
        strokeColor=color, strokeStyle="dashed" if dashed else "solid",
        roundness={"type": 2},
    )
    element.update({
        "points": points, "lastCommittedPoint": None, "startBinding": None,
        "endBinding": None, "startArrowhead": None, "endArrowhead": "arrow",
        "elbowed": False,
    })
    elements.append(element)


# --------------------------------------------------------------------------- header
text("Data Preprocessing Pipeline", 60, 46, size=26)
text("=" * 34, 60, 82, size=18)

# --------------------------------------------------------------------------- inputs
container("Inputs", 60, 130, 275, 320)
box("Source Corpora\n/data/annealing  (19 subsets)", 82, 172, 230, 54)
badge("R", 312, 172)
box("Corpus Registry\nannealing_registry.yaml", 82, 240, 230, 50)
badge("R", 312, 240)
box("Propella Annotations\n(external parquet cache)", 82, 304, 230, 50)
badge("R", 312, 304)
box("Tokenizer Config\n(for token estimates only)", 82, 368, 230, 50)
badge("R", 312, 368)

# ------------------------------------------------- lane 1: documents (top), y = 168
box("1. calibrate", 400, 168, 160, 58)
badge("R", 552, 164)
text("~15 s / dataset", 402, 232, size=10, color=NOTE_TEXT)
box("calibration.yaml\nstratified ratios", 400, 262, 160, 46, color=ARTIFACT_TEXT, dashed=True, size=11)

box("2. build-sidecar", 590, 168, 160, 58)
badge("R", 742, 164)
text("~15 h  (SLURM array)", 592, 232, size=10, color=NOTE_TEXT)
box("sidecar/*.parquet\n+ _files.json", 590, 262, 160, 46, color=ARTIFACT_TEXT, dashed=True, size=11)

# ------------------------------------------------- lane 2: annotations (below), y = 340
box("3. bucket-annotations", 590, 340, 170, 58)
badge("R", 752, 336)
text("~4.5 h  (SLURM array)", 592, 404, size=10, color=NOTE_TEXT)
box("buckets/  (key-hashed)", 590, 418, 170, 46, color=ARTIFACT_TEXT, dashed=True, size=11)

# ------------------------------------------------- lanes converge
box("4. join-annotations", 860, 250, 160, 58)
badge("R", 1012, 246)
text("hours;  --resume", 862, 314, size=10, color=NOTE_TEXT)
box("labelled sidecar\n100% coverage", 860, 330, 160, 46, color=ARTIFACT_TEXT, dashed=True, size=11)

box("5. build-cube", 1080, 250, 160, 58)
badge("R", 1232, 246)
text("~20 min", 1082, 314, size=10, color=NOTE_TEXT)
box("cube/*.parquet\ncontingency table", 1080, 330, 160, 46, color=ARTIFACT_TEXT, dashed=True, size=11)

# ------------------------------------------------- selection loop
container("Selection Loop   (cheap -- iterate freely)", 400, 500, 420, 150)
box("annealing_selection.yaml\nthresholds + ratios", 424, 542, 175, 56)
badge("R", 599, 538)
box("6. preview\n--exact / --allow_fallback", 640, 542, 160, 56)
badge("R", 800, 538)
text("~14 s per iteration over 19 cubes", 424, 612, size=10, color=NOTE_TEXT)
arrow((601, 558), (638, 558))
arrow((638, 584), (601, 584))

# ------------------------------------------------- tail
for x, name, artifact in (
    (880, "7. apply", "filtered *.idx\n+ mix_manifest.yaml"),
    (1070, "8. export-jsonl", "*.jsonl per dataset\n(sampling in the bytes)"),
):
    box(name, x, 530, 160, 58)
    badge("R", x + 152, 526)
    box(artifact, x, 608, 160, 46, color=ARTIFACT_TEXT, dashed=True, size=11)

box("cat out/*/*.jsonl", 1260, 530, 175, 58)
badge("R", 1432, 526)
box("-> Trainings Pipeline\n(Trainings Loop)", 1260, 608, 175, 46, color=GUARD_TEXT, dashed=True, size=11)

# ------------------------------------------------- validation
container("Validation & Guards", 60, 560, 275, 265)
box("verify-sidecar\nbyte offsets vs source", 82, 602, 230, 50, color=GUARD_TEXT)
badge("R", 312, 602)
box("join coverage report\nper dataset", 82, 666, 230, 46, color=GUARD_TEXT)
badge("R", 312, 666)
box("smoke snapshot\n+ check_token_estimates", 82, 726, 230, 50, color=GUARD_TEXT)
badge("R", 312, 726)
text("run after any transfer,\nand before  apply", 82, 786, size=10, color=NOTE_TEXT)

# ------------------------------------------------- flow arrows (routed to avoid boxes)
arrow((314, 198), (396, 192))                                  # corpora -> calibrate
arrow((562, 197), (586, 197))                                  # calibrate -> build-sidecar
arrow((315, 330), (586, 366))                                  # propella -> bucket-annotations
arrow((752, 200), (856, 262))                                  # sidecar -> join
arrow((762, 366), (856, 298))                                  # buckets -> join
arrow((1022, 279), (1076, 279))                                # join -> cube
arrow((1160, 378), (1160, 476), (800, 476), (800, 498))        # cube -> selection loop
arrow((802, 570), (876, 556))                                  # preview -> apply
arrow((1042, 559), (1068, 559))                                # apply -> export-jsonl
arrow((1232, 559), (1258, 559))                                # export-jsonl -> concatenation

# ------------------------------------------------- legend
container("Owner", 1680, 130, 120, 200)
for row, letter in enumerate("RTSHM"):
    badge(letter, 1712, 178 + row * 30)
text("initials only", 1680, 340, size=9, color=NOTE_TEXT)

document = {
    "type": "excalidraw",
    "version": 2,
    "source": "https://excalidraw.com",
    "elements": elements,
    "appState": {"gridSize": None, "viewBackgroundColor": "#ffffff"},
    "files": {},
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(document, indent=2))
print(f"wrote {OUT}  ({len(elements)} elements)")
