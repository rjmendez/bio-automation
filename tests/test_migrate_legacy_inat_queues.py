"""Tests for bio_automation.migrate_legacy_inat_queues (the one-time legacy
queue migration) against a committed, representative fixture slice -- see
tests/fixtures/inat_migration/. These are real JSON/JSONL files the module
reads with its real read_json/load_decisions machinery (never stubbed);
only the source *paths* are fixtures, not the migration logic.
"""
from __future__ import annotations

from bio_automation.inat_queue import QueueItem, presort
from bio_automation.io import read_json, read_jsonl
from bio_automation.migrate_legacy_inat_queues import (
    MigrationReport,
    migrate,
)

FIXTURES = "tests/fixtures/inat_migration"
FILES_QUEUE = f"{FIXTURES}/legacy_queue_files.json"
LIGHTCRAFT_QUEUE = f"{FIXTURES}/legacy_queue_lightcraft.json"
FILES_DECISIONS = f"{FIXTURES}/legacy_decisions_files.jsonl"
LIGHTCRAFT_DECISIONS = f"{FIXTURES}/legacy_decisions_lightcraft.jsonl"


def _run(tmp_path, dry_run: bool = False) -> tuple[MigrationReport, object, object]:
    out_queue = tmp_path / "inat_queue_unified.json"
    out_decisions = tmp_path / "inat_decisions_unified.jsonl"
    report = migrate(FILES_QUEUE, LIGHTCRAFT_QUEUE, FILES_DECISIONS, LIGHTCRAFT_DECISIONS,
                      out_queue, out_decisions, dry_run=dry_run)
    return report, out_queue, out_decisions


class TestMigrationCounts:
    def test_counts_match_the_fixture_exactly(self, tmp_path):
        """5 files items (3 groups, 2 decided) + 4 lightcraft items (2
        groups, both decided) -> 9 items in, 4 decided in, offset +3 (the
        files queue's own max group number)."""
        report, _, _ = _run(tmp_path)

        assert report == MigrationReport(
            files_items=5, lightcraft_items=4, total_items=9,
            files_decided=2, lightcraft_decided=2, total_decided=4,
            lightcraft_group_offset=3,
        )

    def test_dry_run_computes_the_same_report_but_writes_nothing(self, tmp_path):
        """Positive/negative twin of test_writes_the_unified_files: same
        inputs, dry_run=True must report identically but touch neither
        output path."""
        report, out_queue, out_decisions = _run(tmp_path, dry_run=True)

        assert report == MigrationReport(
            files_items=5, lightcraft_items=4, total_items=9,
            files_decided=2, lightcraft_decided=2, total_decided=4,
            lightcraft_group_offset=3,
        )
        assert not out_queue.exists()
        assert not out_decisions.exists()

    def test_writes_the_unified_files(self, tmp_path):
        _, out_queue, out_decisions = _run(tmp_path)

        assert out_queue.exists()
        assert out_decisions.exists()


class TestUnifiedQueueSchema:
    def test_files_items_keep_their_legacy_group_number_as_the_unified_group(self, tmp_path):
        _, out_queue, _ = _run(tmp_path)
        items = {item["rel"]: item for item in read_json(out_queue)}

        burst = items["2026/2026-09-01/inat-LR/DSC_0001.jpg"]
        assert burst["provenance"] == "files"
        assert burst["legacy_group"] == 1
        assert burst["group"] == 1

        solo = items["2026/2026-09-02/inat-LR/DSC_0010.jpg"]
        assert solo["group"] == 2
        assert solo["legacy_group"] == 2

    def test_lightcraft_items_are_renumbered_by_the_files_queues_max_group(self, tmp_path):
        """Positive twin of the files-items test on the same migration run:
        lightcraft legacy group 1 -> unified group 4 (3 + 1), legacy group
        2 -> unified group 5 (3 + 2) -- never colliding with files groups
        1-3."""
        _, out_queue, _ = _run(tmp_path)
        items = {item["lc_id"]: item for item in read_json(out_queue) if item["provenance"] == "lightcraft"}

        assert items[9001]["group"] == 4
        assert items[9001]["legacy_group"] == 1
        assert items[9010]["group"] == 5
        assert items[9010]["legacy_group"] == 2

    def test_no_unified_group_number_is_shared_across_provenances(self, tmp_path):
        _, out_queue, _ = _run(tmp_path)
        items = read_json(out_queue)

        by_group = {}
        for item in items:
            by_group.setdefault(item["group"], set()).add(item["provenance"])
        assert all(len(provenances) == 1 for provenances in by_group.values())
        assert sorted(by_group) == [1, 2, 3, 4, 5]

    def test_non_schema_fields_are_carried_over_unchanged(self, tmp_path):
        """The unification only touches group/legacy_group/provenance --
        every other legacy field (lat/lon, loc_source, lc_id, kind, ...)
        passes through untouched."""
        _, out_queue, _ = _run(tmp_path)
        items = {item["lc_id"]: item for item in read_json(out_queue) if item["provenance"] == "lightcraft"}

        assert items[9010]["lat"] == 39.9
        assert items[9010]["lon"] == -77.4
        assert items[9010]["loc_source"] == "gpx_myasshurts"
        assert items[9010]["kind"] == "backlog"


class TestUnifiedDecisions:
    def test_every_legacy_decision_survives_with_its_exact_fields_and_new_group(self, tmp_path):
        """The core safety property: migrate a copy of legacy data, then
        every decided item in the source has a corresponding decided item
        in the output with the SAME decision (exact fields, not just
        status), at its remapped group number."""
        _, _, out_decisions = _run(tmp_path)
        rows = {row["group"]: row for row in read_jsonl(out_decisions)}

        assert set(rows) == {1, 2, 4, 5}

        approved = rows[1]
        assert approved["status"] == "approved"
        assert approved["provenance"] == "files"
        assert approved["legacy_group"] == 1
        assert approved["taxon_name"] == "Papilio glaucus"
        assert approved["lat"] == 40.0
        assert approved["lon"] == -76.0
        assert approved["photo_order"] == [0, 1]

        files_skip = rows[2]
        assert files_skip["status"] == "skipped"
        assert files_skip["provenance"] == "files"
        assert files_skip["legacy_group"] == 2

        lc_skip = rows[4]
        assert lc_skip["status"] == "skipped"
        assert lc_skip["provenance"] == "lightcraft"
        assert lc_skip["legacy_group"] == 1
        assert lc_skip["taxon_name"] == "Opiliones"

    def test_the_latest_legacy_row_wins_not_an_earlier_one(self, tmp_path):
        """Positive twin: group 1 (files) has an earlier 'hold' row that a
        later 'approved' row supersedes; group 2 (lightcraft) has an
        earlier 'pending' row a later 'hold' row supersedes. Both must
        reflect the LATEST legacy status, proving load_decisions' replay
        -- not just the first or the last line in the raw file -- is what
        migrate_decisions consumes."""
        _, _, out_decisions = _run(tmp_path)
        rows = {row["group"]: row for row in read_jsonl(out_decisions)}

        assert rows[1]["status"] == "approved"
        assert rows[5]["status"] == "hold"

    def test_an_undecided_legacy_group_has_no_row_in_the_unified_ledger(self, tmp_path):
        """Negative twin of the above: files legacy group 3 was never
        decided and must not appear at all (not as 'pending', not as
        anything)."""
        _, _, out_decisions = _run(tmp_path)
        groups = {row["group"] for row in read_jsonl(out_decisions)}

        assert 3 not in groups


class TestIdempotentAndSafe:
    def test_running_twice_produces_byte_identical_output(self, tmp_path):
        out_queue = tmp_path / "inat_queue_unified.json"
        out_decisions = tmp_path / "inat_decisions_unified.jsonl"

        migrate(FILES_QUEUE, LIGHTCRAFT_QUEUE, FILES_DECISIONS, LIGHTCRAFT_DECISIONS, out_queue, out_decisions)
        first_queue_bytes = out_queue.read_bytes()
        first_decisions_bytes = out_decisions.read_bytes()

        migrate(FILES_QUEUE, LIGHTCRAFT_QUEUE, FILES_DECISIONS, LIGHTCRAFT_DECISIONS, out_queue, out_decisions)
        second_queue_bytes = out_queue.read_bytes()
        second_decisions_bytes = out_decisions.read_bytes()

        assert first_queue_bytes == second_queue_bytes
        assert first_decisions_bytes == second_decisions_bytes

    def test_source_files_are_never_written_to(self, tmp_path):
        import pathlib

        sources = [pathlib.Path(p) for p in
                   (FILES_QUEUE, LIGHTCRAFT_QUEUE, FILES_DECISIONS, LIGHTCRAFT_DECISIONS)]
        before = {p: p.read_bytes() for p in sources}

        _run(tmp_path)

        after = {p: p.read_bytes() for p in sources}
        assert before == after


class TestUnifiedQueueFeedsTheExistingPresort:
    """Ties the migration to the already-ported "pre-sort a keep list and a
    cull list, don't cull yourself" contract (inat_queue.presort): rebuild
    QueueItems from the unified output and prove presort buckets them
    correctly using nothing but the migrated decisions -- one schema, one
    set of pure functions, no per-provenance branching needed downstream.
    """

    def test_presort_over_the_unified_output_matches_the_migrated_decisions(self, tmp_path):
        _, out_queue, out_decisions = _run(tmp_path)
        queue_items = read_json(out_queue)
        decisions = {row["group"]: row for row in read_jsonl(out_decisions)}

        by_group: dict[int, list[str]] = {}
        meta: dict[int, dict] = {}
        for item in queue_items:
            by_group.setdefault(item["group"], []).append(item["rel"])
            meta[item["group"]] = item
        items = [
            QueueItem(group=g, day=meta[g]["day"], camera=meta[g]["camera"], src=meta[g]["provenance"],
                      photos=photos)
            for g, photos in by_group.items()
        ]

        buckets = presort(items, decisions)

        assert buckets == {"keep": [1], "cull": [2, 4], "hold": [5], "undecided": [3]}
