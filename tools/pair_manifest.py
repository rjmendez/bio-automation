"""Read-only sizing scan: how many real (plain-raw, Lightroom-enhanced) pairs exist
in the owner's own library, for training custom detail/denoise and upscale weights.

Two pairings, both keyed on bio_automation.stock_select.frame_stem so a NEF and its
Lightroom-made Enhanced-NR variant are recognised as the same camera frame:

1. raw/raw  : a .nef next to a sibling "*-Enhanced-NR*.dng" in the same day folder
              under settings.LIBRARY (F:\\Bio). Strongest pair (both are raw
              decodes), but DNG "Enhanced-NR" variants are rare.
2. raw/jpeg : a .nef under settings.LIBRARY whose frame_stem matches a
              "*-Enhanced-NR*.jpg" under settings.STOCK_DIR/ready (the already
              Lightroom-exported Stock set). Weaker target (q95 JPEG, not raw) but
              far more pairs, since most Enhanced-NR edits never got re-exported to
              a standalone DNG.

Nothing is read except directory listings and filenames: no pixels decoded, no
writes. Prints counts and writes the manifest rows to --out (CSV) for later use.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bio_automation import settings  # noqa: E402
from bio_automation.stock_select import ENHANCED_SUFFIX, frame_stem  # noqa: E402


def scan_raw_library(library: Path) -> dict[tuple[str, str], list[Path]]:
    """{(day, frame_stem): [paths]} for every raw file under library/<year>/<day>/."""
    groups: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for year_dir in sorted(p for p in library.iterdir() if p.is_dir() and p.name.isdigit()):
        for day_dir in sorted(p for p in year_dir.iterdir() if p.is_dir()):
            for f in day_dir.iterdir():
                if f.suffix.lower() in settings.RAW_EXTENSIONS:
                    groups[(day_dir.name, frame_stem(f.name))].append(f)
    return groups


def scan_stock_jpegs(stock_ready: Path) -> dict[str, list[Path]]:
    """{frame_stem: [paths]} for every Enhanced-NR-named JPEG under stock_ready."""
    groups: dict[str, list[Path]] = defaultdict(list)
    if not stock_ready.exists():
        return groups
    for f in stock_ready.rglob("*.jpg"):
        if ENHANCED_SUFFIX in f.stem:
            groups[frame_stem(f.name)].append(f)
    return groups


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("pair_manifest.csv"))
    args = ap.parse_args()

    library = settings.LIBRARY
    stock_ready = settings.STOCK_DIR / "ready"

    raw_groups = scan_raw_library(library)
    stock_groups = scan_stock_jpegs(stock_ready)

    raw_raw_pairs = []   # (plain_nef, enhanced_dng)
    raw_jpeg_pairs = []  # (plain_nef, enhanced_jpeg)

    for (day, stem), paths in raw_groups.items():
        plain = [p for p in paths if ENHANCED_SUFFIX not in p.stem]
        enhanced_dng = [p for p in paths if ENHANCED_SUFFIX in p.stem]
        if plain and enhanced_dng:
            raw_raw_pairs.append((plain[0], enhanced_dng[0]))
        if plain and stem in stock_groups:
            raw_jpeg_pairs.append((plain[0], stock_groups[stem][0]))

    print(f"raw library:        {sum(len(v) for v in raw_groups.values())} raw files, "
          f"{len(raw_groups)} (day, frame) groups")
    print(f"stock Enhanced-NR:  {sum(len(v) for v in stock_groups.values())} JPEGs, "
          f"{len(stock_groups)} distinct frames")
    print(f"raw/raw pairs:      {len(raw_raw_pairs)}  (NEF <-> sibling Enhanced-NR DNG)")
    print(f"raw/jpeg pairs:     {len(raw_jpeg_pairs)}  (NEF <-> Enhanced-NR Stock JPEG)")

    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["kind", "plain_raw", "enhanced_target"])
        for plain, enhanced in raw_raw_pairs:
            w.writerow(["raw_raw", str(plain), str(enhanced)])
        for plain, enhanced in raw_jpeg_pairs:
            w.writerow(["raw_jpeg", str(plain), str(enhanced)])
    print(f"manifest written: {args.out} ({len(raw_raw_pairs) + len(raw_jpeg_pairs)} rows)")


if __name__ == "__main__":
    main()
