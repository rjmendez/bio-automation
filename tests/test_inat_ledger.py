"""Unit tests for bio_automation.inat_ledger (hermetic -- real JSONL I/O
against tmp_path, no stubbing of the unit under test)."""
from __future__ import annotations

import pytest

from bio_automation.inat_ledger import (
    is_done,
    load_decisions,
    load_posted,
    record_decision,
    record_posted,
)
from bio_automation.io import read_jsonl


def _approval_fields(**overrides):
    base = {"observed_on_string": "2026-09-01 08:00:00", "time_zone": "Pacific Time (US & Canada)",
            "lat": 37.9, "lon": -122.5, "geoprivacy": "open"}
    base.update(overrides)
    return base


class TestRecordDecision:
    def test_appends_a_hold_row_with_no_approval_fields_required(self, tmp_path):
        path = tmp_path / "inat_decisions.jsonl"

        row = record_decision(path, 7, "hold", now_fn=lambda: 1700000000.0)

        assert row == {"group": 7, "status": "hold", "ts": 1700000000.0}
        assert read_jsonl(path) == [row]

    def test_appends_a_skipped_row(self, tmp_path):
        path = tmp_path / "inat_decisions.jsonl"

        row = record_decision(path, 7, "skipped", now_fn=lambda: 2.0)

        assert row == {"group": 7, "status": "skipped", "ts": 2.0}

    def test_rejects_an_unknown_status_value(self, tmp_path):
        path = tmp_path / "inat_decisions.jsonl"

        with pytest.raises(ValueError, match="status must be one of"):
            record_decision(path, 7, "maybe-later")

        assert read_jsonl(path) == []

    def test_approve_without_required_fields_raises_and_writes_nothing(self, tmp_path):
        path = tmp_path / "inat_decisions.jsonl"

        with pytest.raises(ValueError, match="approve needs: observed_on_string, time_zone, lat, lon"):
            record_decision(path, 7, "approved")

        assert read_jsonl(path) == []

    def test_approve_with_all_fields_but_bad_geoprivacy_raises(self, tmp_path):
        path = tmp_path / "inat_decisions.jsonl"

        with pytest.raises(ValueError, match="geoprivacy must be one of"):
            record_decision(path, 7, "approved", fields=_approval_fields(geoprivacy="public"))

        assert read_jsonl(path) == []

    def test_approve_with_valid_fields_succeeds_and_keeps_them_on_the_row(self, tmp_path):
        """Positive twin of the two approve-rejection tests above, on the
        same fixture shape."""
        path = tmp_path / "inat_decisions.jsonl"

        row = record_decision(path, 7, "approved", fields=_approval_fields(), now_fn=lambda: 3.0)

        assert row == {"group": 7, "status": "approved", "ts": 3.0, **_approval_fields()}
        assert read_jsonl(path) == [row]

    def test_second_call_appends_rather_than_overwriting(self, tmp_path):
        path = tmp_path / "inat_decisions.jsonl"
        record_decision(path, 1, "hold", now_fn=lambda: 1.0)
        record_decision(path, 2, "skipped", now_fn=lambda: 2.0)

        rows = read_jsonl(path)

        assert [r["group"] for r in rows] == [1, 2]


class TestLoadDecisions:
    def test_latest_row_wins_when_a_group_has_two_decisions(self, tmp_path):
        path = tmp_path / "inat_decisions.jsonl"
        record_decision(path, 1, "hold", now_fn=lambda: 1.0)
        record_decision(path, 1, "approved", fields=_approval_fields(), now_fn=lambda: 2.0)

        decisions = load_decisions(path)

        assert decisions[1]["status"] == "approved"
        assert decisions[1]["ts"] == 2.0

    def test_missing_ledger_file_loads_as_empty(self, tmp_path):
        assert load_decisions(tmp_path / "does_not_exist.jsonl") == {}

    def test_distinct_groups_each_keep_their_own_decision(self, tmp_path):
        path = tmp_path / "inat_decisions.jsonl"
        record_decision(path, 1, "approved", fields=_approval_fields(), now_fn=lambda: 1.0)
        record_decision(path, 2, "skipped", now_fn=lambda: 2.0)

        decisions = load_decisions(path)

        assert decisions[1]["status"] == "approved"
        assert decisions[2]["status"] == "skipped"


class TestPostedLedger:
    def test_is_done_false_before_any_record(self, tmp_path):
        path = tmp_path / "inat_posted.jsonl"
        assert is_done(path, 1) is False

    def test_is_done_true_after_a_posted_record(self, tmp_path):
        path = tmp_path / "inat_posted.jsonl"
        record_posted(path, 1, "posted", observation_id=4242, now_fn=lambda: 5.0)

        assert is_done(path, 1) is True
        assert is_done(path, 2) is False

    def test_is_done_true_for_a_partial_record(self, tmp_path):
        """A partial post (observation created, photo attach failed) still
        counts as done -- it must never be re-sent (that would create a
        duplicate observation)."""
        path = tmp_path / "inat_posted.jsonl"
        record_posted(path, 1, "partial", observation_id=4242, photos=0,
                       error="HTTPError: 500", now_fn=lambda: 5.0)

        assert is_done(path, 1) is True

    def test_is_done_false_for_a_failed_record(self, tmp_path):
        """A failed post (no observation was ever created) is NOT done --
        it is safe, and intended, to retry."""
        path = tmp_path / "inat_posted.jsonl"
        record_posted(path, 1, "failed", error="InatAuthError: 401", now_fn=lambda: 5.0)

        assert is_done(path, 1) is False

    def test_rejects_an_unknown_posted_status(self, tmp_path):
        path = tmp_path / "inat_posted.jsonl"

        with pytest.raises(ValueError, match="status must be one of"):
            record_posted(path, 1, "sort-of-posted")

        assert read_jsonl(path) == []

    def test_load_posted_returns_observation_id_per_group(self, tmp_path):
        path = tmp_path / "inat_posted.jsonl"
        record_posted(path, 1, "posted", observation_id=4242, photos=3, now_fn=lambda: 5.0)
        record_posted(path, 2, "posted", observation_id=9001, photos=1, now_fn=lambda: 6.0)

        posted = load_posted(path)

        assert posted[1]["observation_id"] == 4242
        assert posted[2]["observation_id"] == 9001
