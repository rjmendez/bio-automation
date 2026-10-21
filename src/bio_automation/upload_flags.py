"""Flag problem frames/observations in an iNat upload set's observations.csv. Nothing is removed or reordered (the owner never culls): two columns
are appended, `flags` (semicolon-separated tokens) and `flag_notes` (readable text), and a re-run replaces them instead of duplicating them.

Automatic flag:  black_frame -- a JPEG whose median luma is below BLACK_LUMA (a dead/unlit exposure; verified on the NEFs for the first set).
Manual flags:    a JSON file {"<group>": {"flags": ["electronics"], "note": "circuit board"}}, for things only a person (or a reviewer looking at
                 the pictures) can tell: not wildlife, people visible, signs, pets, ...
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from .stock_select import median_luma

BLACK_LUMA = 2.0
FLAG_COLUMNS = ("flags", "flag_notes")


def jpeg_median_luma(path: Path) -> float:
    with Image.open(path) as im:
        im.draft("RGB", (1024, 1024))
        im = im.convert("RGB")
        im.thumbnail((512, 512))
        return median_luma(np.asarray(im))


def load_manual_notes(path: Path | None) -> dict[int, dict]:
    if path is None:
        return {}
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for group, v in raw.items():
        if not str(group).isdigit():
            raise ValueError(f"manual notes: group key {group!r} is not a number")
        flags = v.get("flags")
        if not isinstance(flags, list) or not flags or not all(isinstance(f, str) and f for f in flags):
            raise ValueError(f"manual notes: group {group} needs a non-empty list of flag names")
        out[int(group)] = {"flags": flags, "note": str(v.get("note", ""))}
    return out


def flag_observations_csv(folder: Path, manual: dict[int, dict] | None = None, *, black_luma: float = BLACK_LUMA) -> dict:
    """Add `flags`/`flag_notes` to folder/observations.csv from measured luma of the folder's JPEGs and the manual notes. Returns counts."""
    folder = Path(folder)
    csv_path = folder / "observations.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"{csv_path} not found")
    manual = manual or {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = [c for c in reader.fieldnames if c not in FLAG_COLUMNS]
        rows = list(reader)
    unknown = sorted(set(manual) - {int(r["group"]) for r in rows})
    if unknown:
        raise ValueError(f"manual notes name groups that are not in observations.csv: {unknown}")
    black_total = flagged_groups = 0
    for r in rows:
        files = [x for x in r["files"].split(";") if x]
        black = []
        for name in files:
            if jpeg_median_luma(folder / name) < black_luma:
                black.append(name)
        flags, notes = [], []
        if black:
            flags.append("black_frame")
            notes.append(f"{len(black)} of {len(files)} frames are black: {', '.join(black)}")
        m = manual.get(int(r["group"]))
        if m:
            flags.extend(m["flags"])
            if m["note"]:
                notes.append(m["note"])
        r["flags"] = ";".join(flags)
        r["flag_notes"] = " | ".join(notes)
        black_total += len(black)
        flagged_groups += bool(flags)
    tmp = csv_path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fields + list(FLAG_COLUMNS), lineterminator="\r\n")   # same as the file build_upload_set wrote
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, csv_path)
    return {"groups": len(rows), "flagged_groups": flagged_groups, "black_frames": black_total}
