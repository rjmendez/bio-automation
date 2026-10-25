"""Qdrant freshness: the sync marker, the stale-index alert and selfcheck warning, and the recording wrapper around `bio qdrant-sync` and
`bio qdrant-inat-sync`. Qdrant itself is never reached: the heavy collaborators are fakes that record their calls."""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import pytest
from test_social import NOW as SELFCHECK_NOW
from test_social_alerts import NOW
from test_social_alerts import Env

from bio_automation import cli, settings, social, social_alerts, social_selfcheck, sync_marker
from bio_automation.social_rules import Audit
from bio_automation.social_selfcheck import OK, WARN

DAY = dt.timedelta(days=1)


class TestMarker:
    def test_record_then_load_round_trips_the_time_and_the_counts(self, tmp_path):
        path = tmp_path / "m.json"
        got = sync_marker.record("qdrant-sync", {"images": 7}, path=path, now=NOW)
        assert got == {"at": NOW.isoformat(timespec="seconds"), "images": 7}
        assert sync_marker.load(path) == {"qdrant-sync": got}

    def test_a_second_sync_keeps_the_first_ones_entry(self, tmp_path):
        path = tmp_path / "m.json"
        sync_marker.record("qdrant-sync", path=path, now=NOW)
        sync_marker.record("qdrant-inat-sync", path=path, now=NOW + DAY)
        assert {k: v["at"] for k, v in sync_marker.load(path).items()} == {
            "qdrant-sync": NOW.isoformat(timespec="seconds"), "qdrant-inat-sync": (NOW + DAY).isoformat(timespec="seconds")}
        assert list(path.parent.glob("*.tmp-*")) == []

    @pytest.mark.parametrize("content", [None, "not json", "[1]", '{"qdrant-sync": 3}'])
    def test_a_missing_or_unreadable_marker_is_empty(self, tmp_path, content):
        path = tmp_path / "m.json"
        if content is not None:
            path.write_text(content, encoding="utf-8")
        assert sync_marker.load(path) == {}


class TestStale:
    def marks(self, tmp_path, qdrant_age, inat_age):
        path = tmp_path / "m.json"
        sync_marker.record("qdrant-sync", path=path, now=NOW - qdrant_age)
        sync_marker.record("qdrant-inat-sync", path=path, now=NOW - inat_age)
        return path

    def test_fresh_syncs_are_not_stale(self, tmp_path):
        assert sync_marker.stale(NOW, path=self.marks(tmp_path, DAY, 2 * DAY), max_days=3) == []

    def test_only_the_old_sync_is_named_with_its_age(self, tmp_path):
        got = sync_marker.stale(NOW, path=self.marks(tmp_path, DAY, 5 * DAY), max_days=3)
        assert got == [("qdrant-inat-sync", f"qdrant-inat-sync last wrote 5.0 days ago ({(NOW - 5 * DAY).isoformat(timespec='seconds')}), "
                                            "more than 3: the Qdrant index is stale")]

    def test_exactly_the_limit_is_not_stale_a_moment_past_it_is(self, tmp_path):
        path = self.marks(tmp_path, 3 * DAY, 3 * DAY)
        assert sync_marker.stale(NOW, path=path, max_days=3) == []
        assert [n for n, _ in sync_marker.stale(NOW + dt.timedelta(seconds=1), path=path, max_days=3)] == ["qdrant-sync", "qdrant-inat-sync"]

    def test_a_sync_that_never_ran_is_reported_as_such(self, tmp_path):
        assert sync_marker.stale(NOW, path=tmp_path / "none.json", max_days=3) == [
            (n, f"{n} has no recorded successful run (the Qdrant index may be empty or out of date)") for n in sync_marker.NAMES]

    def test_the_default_limit_is_the_setting(self, tmp_path, monkeypatch):
        path = self.marks(tmp_path, 2 * DAY, 2 * DAY)
        monkeypatch.setattr(settings, "QDRANT_STALE_DAYS", 1.5)
        assert [n for n, _ in sync_marker.stale(NOW, path=path)] == list(sync_marker.NAMES)
        monkeypatch.setattr(settings, "QDRANT_STALE_DAYS", 2.5)
        assert sync_marker.stale(NOW, path=path) == []


class TestAlert:
    def test_a_stale_sync_is_a_warning_alert_naming_it(self, tmp_path):
        env = Env(tmp_path)
        sync_marker.record("qdrant-inat-sync", path=env.marker, now=NOW - 4 * DAY)
        alerts, _ = env.scan()
        assert [(a.key, a.kind, a.level) for a in alerts] == [("qdrant_stale:qdrant-inat-sync:2026-10-23", "qdrant_stale", "warning")]
        assert alerts[0].message.startswith("qdrant-inat-sync last wrote 4.0 days ago")

    def test_fresh_syncs_raise_nothing(self, tmp_path):
        assert Env(tmp_path).scan()[0] == []

    def test_a_missing_marker_alerts_for_both_syncs(self, tmp_path):
        env = Env(tmp_path)
        env.marker.unlink()
        assert [a.key for a in env.scan()[0]] == ["qdrant_stale:qdrant-inat-sync:2026-10-23", "qdrant_stale:qdrant-sync:2026-10-23"]

    def test_the_limit_can_be_widened(self, tmp_path):
        env = Env(tmp_path)
        sync_marker.record("qdrant-sync", path=env.marker, now=NOW - 4 * DAY)
        assert env.scan(qdrant_stale_days=5)[0] == []
        assert [a.kind for a in env.scan(qdrant_stale_days=3)[0]] == ["qdrant_stale"]

    @pytest.mark.parametrize("event", ["qdrant_sync_failed", "backup_offsite_failed", "restore_check_failed"])
    def test_the_new_failure_events_are_warning_alerts(self, tmp_path, event):
        env = Env(tmp_path)
        env.add_audit({"at": social_alerts_at(13), "event": event, "error": "boom"})
        alerts, _ = env.scan()
        assert [(a.kind, a.level) for a in alerts] == [(event, "warning")]
        assert alerts[0].message == f"{event}: boom ({social_alerts_at(13)})"


def social_alerts_at(h):
    return dt.datetime(2026, 10, 23, h, 0, tzinfo=settings.LOCAL_TZ).isoformat(timespec="seconds")


class TestSelfcheck:
    def test_stale_is_a_warning_that_names_the_commands_and_fresh_is_ok(self, world):
        assert world.run()["qdrant-sync"].level == OK
        sync_marker.record("qdrant-sync", now=SELFCHECK_NOW - 9 * DAY)
        c = world.run()["qdrant-sync"]
        assert c.level == WARN
        assert c.message.startswith("qdrant-sync last wrote 9.0 days ago") and c.message.endswith("or the Bio-Qdrant-Sync task)")

    def test_never_synced_is_a_warning(self, world):
        settings.QDRANT_SYNC_MARKER.unlink()
        assert world.run()["qdrant-sync"].level == WARN


@pytest.fixture
def world(tmp_path, monkeypatch):
    from test_social_selfcheck import World as W
    return W(tmp_path, monkeypatch)


class Calls:
    """Records what the fake sync was asked; returns the summary the real one would."""

    def __init__(self, written=True):
        self.calls, self.written = [], written

    def __call__(self, **kw):
        self.calls.append(kw)
        return {"images": 3, "written": self.written}


def patch_heavy(monkeypatch, tmp_path, calls):
    from bio_automation import qdrant_inat, qdrant_obs

    monkeypatch.setattr(qdrant_obs, "qdrant_config", lambda: ("http://qdrant.invalid", "k"))
    monkeypatch.setattr(qdrant_obs, "Qdrant", lambda url, key: ("client", url))
    monkeypatch.setattr(qdrant_obs, "sync", calls)
    monkeypatch.setattr(qdrant_inat, "sync", calls)
    monkeypatch.setattr(social, "embedder", lambda: (None, lambda texts: []))
    monkeypatch.setattr(social, "load_index", lambda path: {"files": {}})
    monkeypatch.setattr(social, "load_prints", lambda: {})


class TestSyncCommands:
    def run(self, monkeypatch, *argv):
        monkeypatch.setattr(sys, "argv", ["bio", *argv])
        return cli.main()

    def events(self):
        return [(r["event"], r["command"]) for r in Audit().records() if r["event"].startswith("qdrant_sync")]

    @pytest.mark.parametrize("command", ["qdrant-sync", "qdrant-inat-sync"])
    def test_a_real_run_records_the_marker_and_audits(self, monkeypatch, tmp_path, capsys, command):
        calls = Calls()
        patch_heavy(monkeypatch, tmp_path, calls)
        assert self.run(monkeypatch, command) == 0
        assert len(calls.calls) == 1 and calls.calls[0]["write"] is True and calls.calls[0]["client"] == ("client", "http://qdrant.invalid")
        assert list(sync_marker.load()) == [command]
        assert self.events() == [("qdrant_sync", command)]

    @pytest.mark.parametrize("command", ["qdrant-sync", "qdrant-inat-sync"])
    def test_a_dry_run_leaves_no_marker_and_no_audit(self, monkeypatch, tmp_path, command):
        calls = Calls(written=False)
        patch_heavy(monkeypatch, tmp_path, calls)
        assert self.run(monkeypatch, command, "--dry-run") == 0
        assert calls.calls[0]["write"] is False
        assert sync_marker.load() == {} and self.events() == []

    def test_a_trial_run_with_a_limit_is_not_a_fresh_index(self, monkeypatch, tmp_path):
        patch_heavy(monkeypatch, tmp_path, Calls())
        assert self.run(monkeypatch, "qdrant-sync", "--limit", "5") == 0
        assert sync_marker.load() == {}

    def test_a_crash_is_audited_as_failed_with_no_marker_and_exits_one(self, monkeypatch, tmp_path, capsys):
        from bio_automation import qdrant_obs

        def boom(**kw):
            raise qdrant_obs.QdrantError("qdrant down")

        patch_heavy(monkeypatch, tmp_path, Calls())
        monkeypatch.setattr(qdrant_obs, "sync", boom)
        assert self.run(monkeypatch, "qdrant-sync") == 1
        assert "bio qdrant-sync: qdrant down" in capsys.readouterr().err
        assert sync_marker.load() == {}
        assert self.events() == [("qdrant_sync_failed", "qdrant-sync")]
        assert Audit().records()[-1]["error"] == "QdrantError: qdrant down"

    def test_a_non_zero_exit_code_is_audited_as_failed(self, monkeypatch, tmp_path):
        seen = []
        wrapped = cli._recorded("qdrant-sync", lambda args: seen.append(args) or 3)
        ns = type("A", (), {"dry_run": False, "limit": None})()
        assert wrapped(ns) == 3 and seen == [ns]
        assert self.events() == [("qdrant_sync_failed", "qdrant-sync")] and sync_marker.load() == {}
        assert [r["error"] for r in Audit().records() if r["event"] == "qdrant_sync_failed"] == ["exit code 3"]


class TestWrapperWithTheRealParser:
    def test_the_inat_sync_parser_runs_through_the_wrapper_and_writes_the_marker(self, monkeypatch):
        seen = []
        monkeypatch.setitem(cli.SOCIAL_COMMANDS, "qdrant-inat-sync", cli._recorded("qdrant-inat-sync", lambda args: seen.append(args) or 0))
        monkeypatch.setattr(sys, "argv", ["bio", "qdrant-inat-sync"])
        assert cli.main() == 0
        assert [(a.command, a.dry_run, a.limit) for a in seen] == [("qdrant-inat-sync", False, None)]
        assert list(sync_marker.load()) == ["qdrant-inat-sync"]

    def test_a_namespace_without_dry_run_or_limit_is_a_real_full_run(self):
        wrapped = cli._recorded("qdrant-sync", lambda args: 0)
        assert wrapped(type("A", (), {})()) == 0
        assert list(sync_marker.load()) == ["qdrant-sync"]

    def test_a_namespace_without_dry_run_that_crashes_is_audited_and_the_crash_propagates(self):
        def boom(args):
            raise RuntimeError("qdrant down")
        with pytest.raises(RuntimeError, match="qdrant down"):
            cli._recorded("qdrant-sync", boom)(type("A", (), {})())
        assert [r["error"] for r in Audit().records() if r["event"] == "qdrant_sync_failed"] == ["RuntimeError: qdrant down"]


class TestMarkerLock:
    def test_a_marker_locked_by_another_holder_is_waited_for_then_reported_and_the_entry_is_not_lost(self, tmp_path, monkeypatch):
        from bio_automation.social import SocialError, byte_range_lock
        path = tmp_path / "m.json"
        sync_marker.record("qdrant-sync", path=path, now=NOW)
        monkeypatch.setattr(sync_marker, "LOCK_WAIT_S", 0.2)
        with byte_range_lock(Path(str(path) + ".lock"), "held"):
            with pytest.raises(SocialError, match="locked by another sync"):
                sync_marker.record("qdrant-inat-sync", path=path, now=NOW)
        assert list(sync_marker.load(path)) == ["qdrant-sync"]
        sync_marker.record("qdrant-inat-sync", path=path, now=NOW)                    # positive twin: free lock, both entries
        assert sorted(sync_marker.load(path)) == ["qdrant-inat-sync", "qdrant-sync"]

    def test_two_syncs_recording_at_once_keep_both_entries(self, tmp_path):
        import threading
        path = tmp_path / "m.json"
        threads = [threading.Thread(target=sync_marker.record, args=(n,), kwargs={"path": path, "now": NOW}) for n in ("qdrant-sync", "qdrant-inat-sync") * 4]
        [t.start() for t in threads]
        [t.join() for t in threads]
        assert sorted(sync_marker.load(path)) == ["qdrant-inat-sync", "qdrant-sync"]
