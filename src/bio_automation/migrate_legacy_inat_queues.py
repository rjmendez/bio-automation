"""One-time migration: the two legacy per-pipeline iNat review queues into
ONE unified queue + decision ledger.

Background
----------
Legacy pipeline/stages/build_inat_queue.py produced inat_queue.json (fed by
raw-folder exports, group numbers 1..N) and its own inat_decisions.jsonl.
Legacy pipeline/stages/lc_queue_from_raws.py produced the LightCraft queue
inat_queue_lc.json (group numbers *also* 1..N, restarting from 1 -- these
two sequences collide) with its own inat_decisions_lc.jsonl. The review
console (pipeline/stages/inat_console.py) picked between them with the
BIO_QUEUE_TAG env var, so there have always been two queue files, two
ledger files and, in effect, two console deployments.

Unified schema
--------------
Every migrated item (and every migrated decision row) gains two new,
always-present fields:

  provenance    "files" | "lightcraft" -- which legacy pipeline produced it.
  legacy_group  the group number it had in ITS OWN legacy queue file, kept
                purely for audit/traceability back to the source data.

and `group` is replaced with a NEW, globally-unique integer: every legacy
"files" item keeps its legacy_group as `group` unchanged, and every legacy
"lightcraft" item's `group` becomes legacy_group + <files-queue's own max
group number> (the `lightcraft_group_offset` on MigrationReport). Since the
"files" queue's own group numbers already run 1..max contiguously, this
offset is exactly big enough that the two ranges can never collide, and it
requires no re-computation of burst grouping -- the proposed-observation
boundaries a human already reviewed are never touched by this migration,
only renumbered into one shared space.

All other per-item fields (rel, day, camera, lat/lon, loc_source, release,
copy, lc_id, kind, keywords, title, ...) are carried over unchanged, so the
unified schema is simply the union of the two legacy per-item shapes plus
the three fields above -- one list, one JSON file, one decision ledger, one
console can read both provenances without a TAG switch.

The decisions ledger is similarly unified: each legacy ledger is replayed
(last-write-wins per legacy group, via inat_ledger.load_decisions -- the
exact semantics the review console already relies on) into one row per
decided group, group-remapped and provenance/legacy_group-tagged the same
way, and written out as a single fresh append-only ledger.

Safety
------
Every path (both legacy queues, both legacy decision ledgers, and both
unified outputs) is a caller-supplied argument -- nothing in this module
hardcodes a legacy path. Sources are only ever read (read_json/read_jsonl,
both plain open-for-read); this module never opens a source path for
writing. Destinations are fully rebuilt from the sources on every run, so
re-running with the same inputs reproduces byte-identical output
(idempotent) -- there is no append-to-existing-destination code path that
could double up records.

Run as a module:

    python -m bio_automation.migrate_legacy_inat_queues \\
        --files-queue /path/inat_queue.json \\
        --lightcraft-queue /path/inat_queue_lc.json \\
        --files-decisions /path/inat_decisions.jsonl \\
        --lightcraft-decisions /path/inat_decisions_lc.jsonl \\
        --out-queue /path/inat_queue_unified.json \\
        --out-decisions /path/inat_decisions_unified.jsonl \\
        [--dry-run]
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from .inat_ledger import load_decisions
from .io import read_json, write_json, write_jsonl

PROVENANCE_FILES = "files"
PROVENANCE_LIGHTCRAFT = "lightcraft"
PROVENANCES = (PROVENANCE_FILES, PROVENANCE_LIGHTCRAFT)


@dataclass(frozen=True)
class MigrationReport:
    files_items: int
    lightcraft_items: int
    total_items: int
    files_decided: int
    lightcraft_decided: int
    total_decided: int
    lightcraft_group_offset: int


def _max_group(items: list[dict]) -> int:
    """The highest `group` number in a legacy queue's items, 0 for an empty
    queue. Legacy group numbers are a contiguous 1..max running sequence
    (see inat_queue.group_photos()), so this is also the count of distinct
    groups -- but we take the max directly rather than assume that, in
    case a legacy queue is ever hand-edited down to a sparser set."""
    return max((item["group"] for item in items), default=0)


def migrate_queue_items(files_items: list[dict], lightcraft_items: list[dict]) -> tuple[list[dict], int]:
    """Union the two legacy per-item lists into one unified list (see
    module docstring for the schema). Returns (unified_items,
    lightcraft_group_offset) -- the offset is also what migrate_decisions()
    needs to remap the lightcraft ledger the same way.
    """
    offset = _max_group(files_items)
    unified: list[dict] = []
    for item in files_items:
        row = dict(item)
        legacy_group = row.pop("group")
        row["legacy_group"] = legacy_group
        row["group"] = legacy_group
        row["provenance"] = PROVENANCE_FILES
        unified.append(row)
    for item in lightcraft_items:
        row = dict(item)
        legacy_group = row.pop("group")
        row["legacy_group"] = legacy_group
        row["group"] = legacy_group + offset
        row["provenance"] = PROVENANCE_LIGHTCRAFT
        unified.append(row)
    return unified, offset


def migrate_decisions(files_decisions: dict[int, dict], lightcraft_decisions: dict[int, dict],
                       lightcraft_group_offset: int) -> list[dict]:
    """Build one fresh unified decisions ledger (a list of rows, one per
    decided group) from the two already-replayed legacy decision maps
    (as returned by inat_ledger.load_decisions -- latest row per legacy
    group). Every field on the original row (status, taxon_id, lat, lon,
    photo_order, ts, ...) is carried over unchanged; only `group` is
    remapped (see migrate_queue_items) and `legacy_group`/`provenance` are
    added. Rows are emitted files-groups-ascending then
    lightcraft-groups-ascending, for a deterministic, readable ledger (this
    is a fresh snapshot ledger, not a replay of every historical row --
    that is the point of collapsing two append-only logs into one).
    """
    rows: list[dict] = []
    for legacy_group, row in sorted(files_decisions.items()):
        new_row = dict(row)
        new_row["legacy_group"] = legacy_group
        new_row["group"] = legacy_group
        new_row["provenance"] = PROVENANCE_FILES
        rows.append(new_row)
    for legacy_group, row in sorted(lightcraft_decisions.items()):
        new_row = dict(row)
        new_row["legacy_group"] = legacy_group
        new_row["group"] = legacy_group + lightcraft_group_offset
        new_row["provenance"] = PROVENANCE_LIGHTCRAFT
        rows.append(new_row)
    return rows


def migrate(files_queue_path, lightcraft_queue_path, files_decisions_path, lightcraft_decisions_path,
            out_queue_path, out_decisions_path, *, dry_run: bool = False) -> MigrationReport:
    """Read the four legacy source files (never written to) and write the
    two unified destination files (fully rebuilt each call -- never
    appended to), unless *dry_run* is True, in which case nothing is
    written and only the report is computed. Every argument is an explicit
    path; none is defaulted to a real legacy location.
    """
    files_items = read_json(Path(files_queue_path))
    lightcraft_items = read_json(Path(lightcraft_queue_path))
    files_decisions = load_decisions(Path(files_decisions_path))
    lightcraft_decisions = load_decisions(Path(lightcraft_decisions_path))

    unified_items, offset = migrate_queue_items(files_items, lightcraft_items)
    unified_decisions = migrate_decisions(files_decisions, lightcraft_decisions, offset)

    if not dry_run:
        write_json(Path(out_queue_path), unified_items)
        write_jsonl(Path(out_decisions_path), unified_decisions)

    return MigrationReport(
        files_items=len(files_items),
        lightcraft_items=len(lightcraft_items),
        total_items=len(unified_items),
        files_decided=len(files_decisions),
        lightcraft_decided=len(lightcraft_decisions),
        total_decided=len(unified_decisions),
        lightcraft_group_offset=offset,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="One-time migration of the two legacy iNat review queues into one unified queue + ledger.")
    parser.add_argument("--files-queue", required=True, help="legacy inat_queue.json path")
    parser.add_argument("--lightcraft-queue", required=True, help="legacy inat_queue_lc.json path")
    parser.add_argument("--files-decisions", required=True, help="legacy inat_decisions.jsonl path")
    parser.add_argument("--lightcraft-decisions", required=True, help="legacy inat_decisions_lc.jsonl path")
    parser.add_argument("--out-queue", required=True, help="unified queue JSON to write")
    parser.add_argument("--out-decisions", required=True, help="unified decisions JSONL to write")
    parser.add_argument("--dry-run", action="store_true",
                         help="compute and print the report only; write nothing")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = migrate(args.files_queue, args.lightcraft_queue, args.files_decisions, args.lightcraft_decisions,
                      args.out_queue, args.out_decisions, dry_run=args.dry_run)
    mode = "DRY RUN -- wrote nothing" if args.dry_run else "wrote unified queue + ledger"
    print(f"{mode}")
    print(f"  files:      {report.files_items} items, {report.files_decided} decided")
    print(f"  lightcraft: {report.lightcraft_items} items, {report.lightcraft_decided} decided "
          f"(group offset +{report.lightcraft_group_offset})")
    print(f"  total:      {report.total_items} items, {report.total_decided} decided")


if __name__ == "__main__":
    main()
