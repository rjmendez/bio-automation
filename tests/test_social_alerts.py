"""social_alerts: the pull-model alert scanner and the daily digest. Synthetic audit, ledger, taxonomy and log files under tmp_path; the alert
command is a recording runner, never a real process."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
from pathlib import Path

import pytest

from bio_automation import settings, social_alerts, social_hooks, themes
from bio_automation.social_alerts import Alert, AlertsError
from test_social import rules_with

TZ = settings.LOCAL_TZ
NOW = dt.datetime(2026, 10, 23, 14, 0, tzinfo=TZ)            # a Friday, 14:00: the 08:00 and 12:00 slots are 60+ minutes old
SLOTS = [dt.time(8), dt.time(12), dt.time(20)]
GB = 1024 ** 3


def at(h: int, m: int = 0, day: int = 23) -> str:
    return dt.datetime(2026, 10, day, h, m, tzinfo=TZ).isoformat(timespec="seconds")


def write_jsonl(path: Path, recs: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    return path


def lentry(order="Agaricales", grade="research", kingdom="Fungi"):
    return {"taxon": {"id": 1, "name": "Amanita muscaria", "rank": "species", "iconic": "Fungi",
                      "lineage": [{"rank": "kingdom", "name": kingdom}, {"rank": "order", "name": order}]}, "grade": grade, "fetched": 1.0}


class Env:
    """The files a scan reads, under tmp_path, with a quiet default: two posts today, a draft waiting, runs at 08:xx and 12:xx."""

    def __init__(self, tmp_path: Path):
        self.dir = tmp_path / "alerts-env"
        self.audit = self.dir / "audit.jsonl"
        self.ledger = self.dir / "ledger.jsonl"
        self.pause = self.dir / "PAUSED"
        self.log = self.dir / "social.log"
        self.taxonomy = self.dir / "obs_taxonomy.json"
        self.alerts = self.dir / "alerts.jsonl"
        self.set_audit([{"at": at(8, 16), "event": "posted", "draft_id": "a"}, {"at": at(12, 5), "event": "posted", "draft_id": "b"}])
        self.set_ledger([{"at": at(8, 16), "event": "posted", "draft_id": "a", "obs_id": 1},
                         {"at": at(12, 5), "event": "posted", "draft_id": "b", "obs_id": 2},
                         {"at": at(7, 0), "event": "drafted", "draft_id": "c", "obs_id": 3, "month": "2026-10"}])

    def set_audit(self, recs):
        write_jsonl(self.audit, recs)

    def set_ledger(self, recs):
        write_jsonl(self.ledger, recs)

    def add_audit(self, rec):
        write_jsonl(self.audit, social_alerts.read_jsonl(self.audit) + [rec])

    def scan(self, now=NOW, rules=None, free=None, **kw):
        free = free or {"C:\\": 100 * GB, "D:\\": 100 * GB}
        return social_alerts.scan(now, audit_path=self.audit, ledger_path=self.ledger, rules=rules or rules_with(), slots=SLOTS,
                                  pause_file=self.pause, log_file=self.log, taxonomy_file=self.taxonomy, disk_free=lambda root: free[root], **kw)

    def kinds(self, **kw):
        return [a.kind for a in self.scan(**kw)[0]]


@pytest.fixture
def env(tmp_path):
    return Env(tmp_path)


# --------------------------------------------------------------------------------------------------- scan: events

def test_quiet_state_has_no_alert_and_says_the_taxonomy_was_not_checked(env):
    alerts, notes = env.scan()
    assert alerts == []
    assert notes == [f"low stock not checked: taxonomy cache {env.taxonomy} is missing"]
    assert social_alerts.exit_code(alerts) == 0


@pytest.mark.parametrize("event,level", [("post_failed", "error"), ("post_verify_failed", "error"), ("circuit_open", "error"),
                                         ("ledger_corrupt", "error"), ("post_blocked", "warning"), ("hook_error", "warning"),
                                         ("paused", "warning"), ("backup_failed", "warning")])
def test_each_event_kind_alerts_with_its_level(env, event, level):
    env.add_audit({"at": at(12, 6), "event": event, "draft_id": "b", "reason": "because"})
    alerts, _ = env.scan()
    assert [(a.kind, a.level) for a in alerts] == [(event, level)]
    assert alerts[0].message.startswith(event) and "because" in alerts[0].message
    assert social_alerts.exit_code(alerts) == (2 if level == "error" else 1)


@pytest.mark.parametrize("event", ["post_failed", "post_verify_failed", "circuit_open", "ledger_corrupt", "post_blocked", "hook_error",
                                   "paused", "backup_failed"])
def test_an_event_older_than_since_does_not_alert(env, event):
    env.add_audit({"at": at(13, 59, day=22), "event": event, "draft_id": "b"})            # 24 h 1 min before NOW
    assert env.kinds() == []
    env.add_audit({"at": at(14, 0, day=22), "event": event, "draft_id": "b"})             # exactly 24 h: inside
    assert env.kinds() == [event]


def test_events_that_are_not_trouble_do_not_alert(env):
    for e in ("posted", "post_decision", "approved", "hook", "gate_rejected", "post_verified"):
        env.add_audit({"at": at(12, 6), "event": e, "draft_id": "b"})
    assert env.kinds() == []


def test_the_pause_and_breaker_events_repeat_every_run_but_alert_once_a_day(env):
    for h in (8, 12, 13):
        env.add_audit({"at": at(h, 30), "event": "paused", "reason": "PAUSED file"})
    alerts, _ = env.scan()
    assert [a.key for a in alerts] == ["paused:2026-10-23"]


def test_a_since_window_shorter_than_the_event_age_hides_it(env):
    env.add_audit({"at": at(12, 6), "event": "post_failed", "draft_id": "b"})
    assert env.kinds(since=dt.timedelta(hours=1)) == []
    assert env.kinds(since=dt.timedelta(hours=2)) == ["post_failed"]


# --------------------------------------------------------------------------------------------------- scan: expected post missing

def missing_env(env):
    env.set_ledger([{"at": at(7, 0), "event": "drafted", "draft_id": "c", "obs_id": 3, "month": "2026-10"}])
    env.set_audit([{"at": at(8, 1), "event": "post_skipped"}, {"at": at(12, 1), "event": "post_skipped"}])      # runs happened, nothing posted


def test_expected_post_missing_when_two_slots_passed_and_nothing_posted(env):
    missing_env(env)
    alerts, _ = env.scan()
    assert [(a.kind, a.level, a.key) for a in alerts] == [("expected_post_missing", "error", "expected_post_missing:2026-10-23:2")]
    assert alerts[0].message == ("0 scheduled post(s) today, 2 slot(s) passed by 60+ minutes (last 12:00), cap not reached, an eligible draft is waiting")


def test_expected_post_missing_slot_edge_is_sixty_minutes(env):
    missing_env(env)
    env.add_audit({"at": at(8, 2), "event": "post_skipped"})
    assert env.kinds(now=dt.datetime(2026, 10, 23, 8, 59, tzinfo=TZ)) == []
    assert env.kinds(now=dt.datetime(2026, 10, 23, 9, 0, tzinfo=TZ)) == ["expected_post_missing"]


def test_expected_post_missing_twin_no_eligible_draft(env):
    missing_env(env)
    env.set_ledger([{"at": at(7, 0), "event": "drafted", "draft_id": "c", "obs_id": 3, "month": "2026-10"},
                    {"at": at(7, 1), "event": "withdrawn", "draft_id": "c"}])
    kinds = env.kinds()
    assert "expected_post_missing" not in kinds and kinds == ["queue_empty"]


def test_expected_post_missing_twin_paused(env):
    missing_env(env)
    env.pause.write_text("stop", encoding="utf-8")
    assert env.kinds() == []
    assert env.kinds(rules=rules_with(pause_switch=False)) == ["expected_post_missing"]       # a PAUSED file means nothing while the rule is off


def test_expected_post_missing_twin_cap_reached(env):
    missing_env(env)
    env.set_ledger([{"at": at(7, 0), "event": "drafted", "draft_id": "c", "obs_id": 3, "month": "2026-10"},
                    {"at": at(8, 16), "event": "posted", "draft_id": "a"}])
    assert env.kinds(rules=rules_with(daily_cap=1)) == []
    assert env.kinds(rules=rules_with(daily_cap=2)) == ["expected_post_missing"]


def test_expected_post_missing_twin_enough_posted(env):
    missing_env(env)
    env.set_ledger([{"at": at(7, 0), "event": "drafted", "draft_id": "c", "obs_id": 3, "month": "2026-10"},
                    {"at": at(8, 16), "event": "posted", "draft_id": "a"}, {"at": at(12, 5), "event": "posted", "draft_id": "b"}])
    assert env.kinds() == []


def test_a_manual_post_does_not_count_as_a_scheduled_one(env):
    missing_env(env)
    env.set_ledger([{"at": at(7, 0), "event": "drafted", "draft_id": "c", "obs_id": 3, "month": "2026-10"},
                    {"at": at(8, 16), "event": "posted", "draft_id": "a", "manual": True}, {"at": at(12, 5), "event": "posted", "draft_id": "b"}])
    assert env.kinds() == ["expected_post_missing"]            # one scheduled post, two slots due
    env.set_ledger([{"at": at(7, 0), "event": "drafted", "draft_id": "c", "obs_id": 3, "month": "2026-10"},
                    {"at": at(8, 16), "event": "posted", "draft_id": "a"}, {"at": at(12, 5), "event": "posted", "draft_id": "b"}])
    assert env.kinds() == []


def test_a_themed_draft_for_another_day_is_not_eligible_while_theme_day_only_is_on(env):
    missing_env(env)
    env.set_ledger([{"at": at(7, 0), "event": "drafted", "draft_id": "t", "obs_ids": [3, 4], "month": "2026-10", "for_date": "2026-10-24", "theme": "x"}])
    assert env.kinds() == []
    assert env.kinds(rules=rules_with(theme_day_only=False)) == ["expected_post_missing"]


# --------------------------------------------------------------------------------------------------- scan: silent runs

def silent_env(env):
    env.set_audit([])
    env.set_ledger([{"at": at(7, 0), "event": "drafted", "draft_id": "c", "obs_id": 3, "month": "2026-10"}])
    return dt.datetime(2026, 10, 23, 8, 31, tzinfo=TZ)


def test_a_run_with_no_audit_record_and_no_log_write_is_silent(env):
    now = silent_env(env)
    alerts, _ = env.scan(now=now)
    assert [(a.kind, a.level, a.key) for a in alerts] == [("run_silent", "error", "run_silent:2026-10-23:08:00")]
    assert alerts[0].message == "the 08:00 run left no audit record and no social.log write by 08:31"


def test_silent_run_edge_is_thirty_minutes(env):
    silent_env(env)
    assert env.kinds(now=dt.datetime(2026, 10, 23, 8, 29, tzinfo=TZ)) == []
    assert env.kinds(now=dt.datetime(2026, 10, 23, 8, 30, tzinfo=TZ)) == ["run_silent"]


def test_silent_run_twin_an_audit_record_in_the_slot(env):
    now = silent_env(env)
    env.add_audit({"at": at(8, 5), "event": "draft_run"})
    assert env.kinds(now=now) == []


def test_silent_run_twin_a_record_before_the_slot_does_not_count(env):
    now = silent_env(env)
    env.add_audit({"at": at(7, 59), "event": "draft_run"})
    assert env.kinds(now=now) == ["run_silent"]


def test_silent_run_twin_social_log_written_in_the_slot(env):
    now = silent_env(env)
    env.log.write_text("run", encoding="utf-8")
    stamp = dt.datetime(2026, 10, 23, 8, 10, tzinfo=TZ).timestamp()
    os.utime(env.log, (stamp, stamp))
    assert env.kinds(now=now) == []
    old = dt.datetime(2026, 10, 22, 20, 10, tzinfo=TZ).timestamp()
    os.utime(env.log, (old, old))
    assert env.kinds(now=now) == ["run_silent"]


def test_a_later_slots_run_does_not_hide_an_earlier_silent_one(env):
    silent_env(env)
    env.add_audit({"at": at(12, 3), "event": "draft_run"})
    alerts, _ = env.scan(now=dt.datetime(2026, 10, 23, 12, 40, tzinfo=TZ))
    assert [a.key for a in alerts if a.kind == "run_silent"] == ["run_silent:2026-10-23:08:00"]


# --------------------------------------------------------------------------------------------------- scan: stock, queue, disk

def pool_file(env, mushrooms: int, claimed_extra=None):
    env.taxonomy.write_text(json.dumps({str(i): lentry() for i in range(1000, 1000 + mushrooms)}), encoding="utf-8")


def stock(env, **kw):
    return {a.key: a for a in env.scan(**kw)[0] if a.kind == "low_stock"}


def test_low_stock_edge_is_ten_posts_of_four_observations(env):
    pool_file(env, 39)
    got = stock(env)
    assert got["low_stock:mushroom-monday"].message == "Mushroom Monday: 9 post(s) left (39 unclaimed observations), under 10"
    assert got["low_stock:mushroom-monday"].level == "warning"
    pool_file(env, 40)
    assert "low_stock:mushroom-monday" not in stock(env)


def test_low_stock_skips_a_theme_with_no_alert_when_it_is_disabled(env, tmp_path):
    pool_file(env, 3)
    toml = tmp_path / "themes.toml"
    toml.write_text(themes.themes_toml([t._replace(enabled=False) if t.key == "mushroom-monday" else t for t in themes.DEFAULT_THEMES]), encoding="utf-8")
    assert "low_stock:mushroom-monday" not in {a.key for a in env.scan(themes_file=toml)[0]}
    assert "low_stock:mushroom-monday" in {a.key for a in env.scan()[0]}


def test_claimed_observations_leave_the_pool_and_withdrawn_ones_come_back(env):
    pool_file(env, 40)
    env.set_ledger([{"at": at(7, 0), "event": "drafted", "draft_id": "c", "obs_ids": [1000, 1001, 1002, 1003], "month": "2026-10"}])
    assert stock(env)["low_stock:mushroom-monday"].message.startswith("Mushroom Monday: 9 post(s) left (36 unclaimed")
    env.set_ledger([{"at": at(7, 0), "event": "drafted", "draft_id": "c", "obs_ids": [1000, 1001, 1002, 1003], "month": "2026-10"},
                    {"at": at(7, 1), "event": "withdrawn", "draft_id": "c"}])
    assert "low_stock:mushroom-monday" not in stock(env)


def test_casual_observations_are_not_stock(env):
    env.taxonomy.write_text(json.dumps({str(i): lentry(grade="casual") for i in range(2000, 2040)}), encoding="utf-8")
    assert stock(env)["low_stock:mushroom-monday"].message.startswith("Mushroom Monday: 0 post(s) left (0 unclaimed")


def test_queue_empty_and_its_twin(env):
    env.set_ledger([{"at": at(8, 16), "event": "posted", "draft_id": "a"}, {"at": at(12, 5), "event": "posted", "draft_id": "b"}])
    alerts, _ = env.scan()
    assert [(a.kind, a.level, a.key) for a in alerts] == [("queue_empty", "warning", "queue_empty:2026-10-23")]
    env.set_ledger([{"at": at(8, 16), "event": "posted", "draft_id": "a"}, {"at": at(12, 5), "event": "posted", "draft_id": "b"},
                    {"at": at(9, 0), "event": "drafted", "draft_id": "n"}, {"at": at(9, 1), "event": "approved", "draft_id": "n"}])
    assert env.kinds() == []


def test_disk_edges_and_levels(env):
    low = {"C:\\": 15 * GB - 1, "D:\\": 20 * GB - 1}
    alerts, _ = env.scan(free=low)
    assert [(a.key, a.level) for a in alerts] == [("disk:C", "error"), ("disk:D", "warning")]
    assert alerts[0].message == "C:\\ has 15.0 GB free, under 15 GB"
    assert env.scan(free={"C:\\": 15 * GB, "D:\\": 20 * GB})[0] == []


def test_an_unreadable_disk_is_a_note_not_an_alert(env):
    def free(root):
        raise OSError("no such drive")

    alerts, notes = social_alerts.scan(NOW, audit_path=env.audit, ledger_path=env.ledger, rules=rules_with(), slots=SLOTS, pause_file=env.pause,
                                       log_file=env.log, taxonomy_file=env.taxonomy, disk_free=free)
    assert alerts == []
    assert notes[1:] == ["disk C:\\ not checked: no such drive", "disk D:\\ not checked: no such drive"]


# --------------------------------------------------------------------------------------------------- de-duplication

def log_rec(key, when, record="alert"):
    return {"at": when.isoformat(), "record": record, "key": key}


def test_dedup_window_edges():
    a = Alert("k", "disk", "error", "m")
    just_inside = log_rec("k", NOW - dt.timedelta(hours=12) + dt.timedelta(seconds=1))
    exactly = log_rec("k", NOW - dt.timedelta(hours=12))
    assert social_alerts.new_alerts([a], [just_inside], NOW) == []
    assert social_alerts.new_alerts([a], [exactly], NOW) == [a]
    assert social_alerts.new_alerts([a], [], NOW) == [a]
    assert social_alerts.new_alerts([a], [log_rec("other", NOW)], NOW) == [a]


def test_the_latest_log_entry_of_a_key_decides():
    a = Alert("k", "disk", "error", "m")
    log = [log_rec("k", NOW - dt.timedelta(hours=30)), log_rec("k", NOW - dt.timedelta(hours=1))]
    assert social_alerts.new_alerts([a], log, NOW) == []


def test_a_sticky_alert_is_announced_once_ever_and_notify_records_do_not_count_as_announcements():
    event = Alert("post_failed:a:t", "post_failed", "error", "m", sticky=True)
    old = log_rec(event.key, NOW - dt.timedelta(hours=30))
    assert social_alerts.new_alerts([event], [old], NOW) == []
    assert social_alerts.new_alerts([event], [log_rec(event.key, NOW, record="notify_failed")], NOW) == [event]
    assert social_alerts.new_alerts([event._replace(sticky=False)], [old], NOW) == [event._replace(sticky=False)]


# --------------------------------------------------------------------------------------------------- run_alerts: log and notify

class RecordingRunner:
    def __init__(self, returncode=0, stderr="", raises=None):
        self.calls, self.returncode, self.stderr, self.raises = [], returncode, stderr, raises

    def __call__(self, argv, **kw):
        self.calls.append((argv, kw))
        if self.raises:
            raise self.raises
        return subprocess.CompletedProcess(argv, self.returncode, "", self.stderr)


def two_alerts(env):
    env.add_audit({"at": at(12, 6), "event": "post_failed", "draft_id": "b", "error": "boom"})
    env.add_audit({"at": at(12, 7), "event": "backup_failed", "reason": "disk full"})


def run(env, runner, **kw):
    return social_alerts.run_alerts(NOW, rules=kw.pop("rules", rules_with()), alerts_log=env.alerts, audit_path=env.audit, ledger_path=env.ledger,
                                    slots=SLOTS, pause_file=env.pause, log_file=env.log, taxonomy_file=env.taxonomy,
                                    disk_free=lambda root: 100 * GB, runner=runner, **kw)


def test_new_alerts_are_logged_once_and_not_again_on_the_next_scan(env):
    two_alerts(env)
    first = run(env, RecordingRunner(), command="")
    rows = social_alerts.read_jsonl(env.alerts)
    assert [(r["record"], r["kind"], r["level"], r["at"]) for r in rows] == [("alert", "post_failed", "error", NOW.isoformat(timespec="seconds")),
                                                                              ("alert", "backup_failed", "warning", NOW.isoformat(timespec="seconds"))]
    assert first["exit"] == 2 and len(first["new"]) == 2
    second = run(env, RecordingRunner(), command="")
    assert second["new"] == [] and len(second["alerts"]) == 2 and second["exit"] == 2
    assert len(social_alerts.read_jsonl(env.alerts)) == 2


def test_notify_runs_the_command_once_per_new_alert_with_the_message_on_stdin(env):
    two_alerts(env)
    runner = RecordingRunner()
    rep = run(env, runner, notify=True, command='notify.exe --to "me"')
    assert [c[0] for c in runner.calls] == [["notify.exe", "--to", "me"]] * 2
    assert [c[1]["input"] for c in runner.calls] == [f"[error] post_failed: {rep['new'][0].message}\n", f"[warning] backup_failed: {rep['new'][1].message}\n"]
    assert all(c[1]["timeout"] == 30 and c[1]["shell"] is False for c in runner.calls)
    assert [a.kind for a in rep["notified"]] == ["post_failed", "backup_failed"] and rep["notify_failed"] == []
    assert [r["record"] for r in social_alerts.read_jsonl(env.alerts)] == ["alert", "alert", "notified", "notified"]
    run(env, runner, notify=True, command="notify.exe")
    assert len(runner.calls) == 2                                                         # nothing new, nothing sent


def test_a_configured_command_notifies_without_the_flag_and_without_a_command_the_flag_only_notes(env, monkeypatch):
    two_alerts(env)
    runner = RecordingRunner()
    run(env, runner, command="notify.exe")
    assert len(runner.calls) == 2
    env.alerts.unlink()
    runner2 = RecordingRunner()
    rep = run(env, runner2, notify=True, command="")
    assert runner2.calls == [] and rep["notes"][-1] == "--notify given but no alert command is set (BIO_SOCIAL_ALERT_COMMAND)"


def test_the_command_defaults_to_the_setting(env, monkeypatch):
    two_alerts(env)
    monkeypatch.setattr(settings, "SOCIAL_ALERT_COMMAND", "from-settings --x")
    runner = RecordingRunner()
    run(env, runner)
    assert runner.calls[0][0] == ["from-settings", "--x"]


def test_without_notify_or_command_nothing_is_run(env):
    two_alerts(env)
    runner = RecordingRunner()
    rep = run(env, runner)
    assert runner.calls == [] and len(rep["new"]) == 2


def test_rule_alerts_off_reports_and_logs_but_does_not_notify(env):
    two_alerts(env)
    runner = RecordingRunner()
    rep = run(env, runner, notify=True, command="notify.exe", rules=rules_with(alerts=False))
    assert runner.calls == [] and rep["exit"] == 2 and len(rep["alerts"]) == 2
    assert rep["notes"][-1] == "rule alerts is off: nothing was sent to the alert command"
    assert [r["record"] for r in social_alerts.read_jsonl(env.alerts)] == ["alert", "alert"]


@pytest.mark.parametrize("runner,expected", [
    (RecordingRunner(returncode=3, stderr="bad token\n"), "exit code 3: bad token"),
    (RecordingRunner(raises=FileNotFoundError("no notify.exe")), "FileNotFoundError: no notify.exe"),
    (RecordingRunner(raises=subprocess.TimeoutExpired("notify.exe", 30)), "TimeoutExpired: Command 'notify.exe' timed out after 30 seconds"),
])
def test_a_failing_command_is_recorded_and_never_raised(env, runner, expected):
    env.add_audit({"at": at(12, 6), "event": "post_failed", "draft_id": "b"})
    rep = run(env, runner, notify=True, command="notify.exe")
    assert [a.kind for a in rep["notify_failed"]] == ["post_failed"] and rep["notified"] == []
    rows = social_alerts.read_jsonl(env.alerts)
    assert [(r["record"], r.get("error")) for r in rows] == [("alert", None), ("notify_failed", expected)]
    assert rows[1]["key"] == rows[0]["key"]


def test_a_failed_notification_is_not_retried_for_an_already_logged_alert(env):
    env.add_audit({"at": at(12, 6), "event": "post_failed", "draft_id": "b"})
    run(env, RecordingRunner(returncode=1), notify=True, command="notify.exe")
    runner = RecordingRunner()
    run(env, runner, notify=True, command="notify.exe")
    assert runner.calls == []


def test_split_command_keeps_windows_backslashes():
    assert social_alerts.split_command(r'"C:\Program Files\n\a.exe" --flag C:\x\y') == [r"C:\Program Files\n\a.exe", "--flag", r"C:\x\y"]


def test_parse_since_and_slots():
    assert social_alerts.parse_since("30m") == dt.timedelta(minutes=30)
    assert social_alerts.parse_since("24h") == dt.timedelta(hours=24)
    assert social_alerts.parse_since("2d") == dt.timedelta(days=2)
    for bad in ("", "24", "h", "0h", "1w"):
        with pytest.raises(AlertsError, match="expected a number and m, h or d"):
            social_alerts.parse_since(bad)
    assert social_alerts.parse_slots("20:00, 08:00") == [dt.time(8), dt.time(20)]
    with pytest.raises(AlertsError, match="'noon' is not HH:MM"):
        social_alerts.parse_slots("08:00,noon")


def test_read_jsonl_skips_a_damaged_line(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"a": 1}\n{broken\n[1]\n{"b": 2}\n', encoding="utf-8")
    assert social_alerts.read_jsonl(p) == [{"a": 1}, {"b": 2}]
    assert social_alerts.read_jsonl(tmp_path / "missing.jsonl") == []


# --------------------------------------------------------------------------------------------------- digest

def fixture_digest(tmp_path, engagement=None):
    rules_then, rules_now = rules_with(variety=False).snapshot(), rules_with().snapshot()
    audit = [
        {"at": at(8, 0, day=22), "event": "post_decision", "rules": rules_then, "draft_id": "old"},
        {"at": at(8, 15), "event": "post_decision", "rules": rules_now, "draft_id": "2026-10-23-fungi-friday", "images": 4},
        {"at": at(12, 0), "event": "post_skipped", "status": "done for today"},
        {"at": at(14, 1), "event": "hook", "name": "image_floor", "stage": "pre_send", "action": "withdraw", "reason": "photo 2 is 640 px", "draft_id": "2026-10-24-wildflower-saturday"},
        {"at": at(20, 0), "event": "post_blocked", "reason": "outcome unknown", "draft_id": "x"},
        {"at": at(9, 0, day=24), "event": "post_failed", "draft_id": "tomorrow"},
    ]
    ledger = [
        {"at": at(7, 0, day=21), "event": "drafted", "draft_id": "2026-10-23-fungi-friday", "obs_ids": [1, 2, 3, 4], "month": "2026-10", "theme": "fungi-friday",
         "for_date": "2026-10-23", "images": 4},
        {"at": at(8, 16), "event": "posted", "draft_id": "2026-10-23-fungi-friday", "uri": "at://did:plc:x/app.bsky.feed.post/3aaa"},
        {"at": at(7, 0, day=22), "event": "drafted", "draft_id": "2026-10-24-wildflower-saturday", "obs_ids": [5, 6], "month": "2026-10",
         "theme": "wildflower-saturday", "for_date": "2026-10-24", "images": 2},
        {"at": at(12, 5), "event": "posted", "draft_id": "2026-10-obs9", "uri": "at://did:plc:x/app.bsky.feed.post/3bbb"},
    ]
    alerts = [{"at": at(9, 0), "record": "alert", "key": "disk:D", "kind": "disk", "level": "warning", "message": "D:\\ has 12.0 GB free, under 20 GB"},
              {"at": at(9, 0), "record": "notified", "key": "disk:D"},
              {"at": at(9, 0), "record": "alert", "key": "post_failed:x:t", "kind": "post_failed", "level": "error", "message": "post_failed x"},
              {"at": at(9, 0, day=22), "record": "alert", "key": "old", "kind": "disk", "level": "error", "message": "yesterday"}]
    pool = {"mushroom-monday": {"name": "Mushroom Monday", "observations": 52, "posts": 13}, "insect-thursday": {"name": "Insect Thursday", "observations": 7, "posts": 1}}
    return social_alerts.digest_text(dt.date(2026, 10, 23), audit=audit, ledger=ledger, alerts_log=alerts, rules=rules_with(), pool=pool,
                                     themes_list=themes.DEFAULT_THEMES, engagement_file=engagement, tz=TZ)


GOLDEN = """# Bluesky digest 2026-10-23

## Posted

- 08:16 2026-10-23-fungi-friday | theme: fungi-friday | images: 4 | at://did:plc:x/app.bsky.feed.post/3aaa
- 12:05 2026-10-obs9 | theme: none | images: unknown | at://did:plc:x/app.bsky.feed.post/3bbb

## Not posted

- 12:00 post_skipped: done for today
- 14:01 hook 2026-10-24-wildflower-saturday: photo 2 is 640 px
- 20:00 post_blocked x: outcome unknown

## Rules in force

- on: alerts, alt_text_privacy, auto_approve, circuit_breaker(3), daily_cap(3), image_floor, ledger_integrity, live_recheck, lower_quality_photos(1000), manual_exempt_from_cap, pause_switch, photo_cache_cap(10), prints_first, skip_casual, strip_review_claim, theme_day_only, theme_species_first, themes, variety(3), verify_posts, year_rotation
- off: month_candidates, month_posting, seasonal_boost

Changes since yesterday:
- variety: off -> on (3)

## Pool remaining

- Insect Thursday: 1 posts (7 observations)
- Mushroom Monday: 13 posts (52 observations)

## Next three days

- 2026-10-24 Sat: Wildflower Saturday (draft 2026-10-24-wildflower-saturday: drafted)
- 2026-10-25 Sun: Slime Mold Sunday (no draft yet)
- 2026-10-26 Mon: Mushroom Monday (no draft yet)

## Active alerts

- error post_failed: post_failed x
- warning disk: D:\\ has 12.0 GB free, under 20 GB
"""


def test_digest_golden_text(tmp_path):
    assert fixture_digest(tmp_path) == GOLDEN


def test_digest_empty_day_says_so():
    text = social_alerts.digest_text(dt.date(2026, 10, 23), audit=[], ledger=[], alerts_log=[], rules=rules_with(), pool=None,
                                     pool_note="taxonomy cache x is missing", themes_list=themes.DEFAULT_THEMES, tz=TZ)
    assert "## Posted\n\n- nothing was posted\n" in text
    assert "## Not posted\n\n- nothing was skipped, withdrawn or blocked\n" in text
    assert "Changes since yesterday: no earlier audit record to compare with." in text
    assert "## Pool remaining\n\n- not counted: taxonomy cache x is missing\n" in text
    assert "## Active alerts\n\n- none\n" in text
    assert "## Engagement" not in text


def test_digest_engagement_section_only_with_the_file_and_reads_it_tolerantly(tmp_path):
    assert "## Engagement" not in fixture_digest(tmp_path, engagement=tmp_path / "missing.jsonl")
    log = tmp_path / "eng.jsonl"
    log.write_text("\n".join([
        json.dumps({"uri": "at://did:plc:x/app.bsky.feed.post/3aaa", "likes": 1, "reposts": 0}),
        "not json at all",
        json.dumps({"uri": "at://did:plc:x/app.bsky.feed.post/3aaa", "likeCount": 5, "repostCount": 2, "replyCount": 1, "quoteCount": "x"}),
        json.dumps({"uri": "at://other/3zzz", "likes": 99}),
        json.dumps(["not", "an object"])]) + "\n", encoding="utf-8")
    text = fixture_digest(tmp_path, engagement=log)
    assert text.endswith("\n## Engagement\n\n- 2026-10-23-fungi-friday: 5 likes, 2 reposts, 1 replies, 0 quotes\n")
    assert text.startswith(GOLDEN)


def test_digest_engagement_with_no_matching_record_says_so(tmp_path):
    log = tmp_path / "eng.jsonl"
    log.write_text(json.dumps({"uri": "at://other/3zzz", "likes": 99}) + "\n", encoding="utf-8")
    assert fixture_digest(tmp_path, engagement=log).endswith("## Engagement\n\n- no engagement records for today's posts yet\n")


def test_write_digest_writes_the_dated_file_in_the_digest_dir(tmp_path):
    ledger = write_jsonl(tmp_path / "l.jsonl", [{"at": at(8, 16), "event": "posted", "draft_id": "d1", "uri": "at://u/1"}])
    path = social_alerts.write_digest(dt.date(2026, 10, 23), rules=rules_with(), digest_dir=tmp_path / "out", audit_path=tmp_path / "a.jsonl",
                                      ledger_path=ledger, alerts_log=tmp_path / "al.jsonl", taxonomy_file=tmp_path / "none.json")
    assert path == tmp_path / "out" / "2026-10-23.md"
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# Bluesky digest 2026-10-23\n\n## Posted\n\n- 08:16 d1 | theme: none | images: unknown | at://u/1\n")
    assert "- not counted: taxonomy cache" in text and "## Engagement" not in text


# --------------------------------------------------------------------------------------------------- commands and hook

def test_digest_command_prints_the_path(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(settings, "SOCIAL_DIGEST_DIR", tmp_path / "dg")
    assert social_alerts.cmd_social_digest(argparse.Namespace(date="2026-10-23")) == 0
    assert capsys.readouterr().out.strip() == str(tmp_path / "dg" / "2026-10-23.md")
    assert (tmp_path / "dg" / "2026-10-23.md").is_file()


def test_digest_command_rejects_a_bad_date():
    with pytest.raises(AlertsError, match=r"--date '23/10': expected YYYY-MM-DD"):
        social_alerts.cmd_social_digest(argparse.Namespace(date="23/10"))


def test_alerts_command_exit_code_and_output(env, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(settings, "SOCIAL_AUDIT_LOG", env.audit)
    monkeypatch.setattr(settings, "SOCIAL_LEDGER", env.ledger)
    monkeypatch.setattr(settings, "SOCIAL_ALERTS_LOG", env.alerts)
    monkeypatch.setattr(social_alerts, "social_log_path", lambda: env.log)
    monkeypatch.setattr(social_alerts.shutil, "disk_usage", lambda p: type("U", (), {"free": 500 * GB})())
    now = dt.datetime.now(TZ)
    env.set_audit([{"at": now.isoformat(), "event": "post_failed", "draft_id": "b", "error": "boom"}])
    env.set_ledger([{"at": now.isoformat(), "event": "drafted", "draft_id": "c", "obs_id": 3, "month": "2026-10"}])
    code = social_alerts.cmd_social_alerts(argparse.Namespace(since="1h", notify=False))
    out = capsys.readouterr().out
    assert code == 2
    assert f"ERROR   post_failed: post_failed b: boom ({now.isoformat()})" in out
    code = social_alerts.cmd_social_alerts(argparse.Namespace(since="1h", notify=False))
    assert code == 2 and "(already announced)" in capsys.readouterr().out
    with pytest.raises(AlertsError, match="expected a number"):
        social_alerts.cmd_social_alerts(argparse.Namespace(since="soon", notify=False))


def test_the_parsers_and_commands_are_exposed_to_the_cli():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    social_alerts.add_parsers(sub)
    a = parser.parse_args(["social-alerts", "--since", "2d", "--notify"])
    assert (a.command, a.since, a.notify) == ("social-alerts", "2d", True)
    assert parser.parse_args(["social-alerts"]).since == "24h"
    d = parser.parse_args(["social-digest", "--date", "2026-10-23"])
    assert (d.command, d.date) == ("social-digest", "2026-10-23")
    assert set(social_alerts.COMMANDS) == {"social-alerts", "social-digest"}
    assert social_alerts.ERRORS == (AlertsError,)


def test_the_maintenance_hook_records_alerts_while_the_rule_is_on(tmp_path, monkeypatch):
    from bio_automation.social import Ledger
    from bio_automation.social_rules import Audit

    monkeypatch.setattr(social_alerts.shutil, "disk_usage", lambda p: type("U", (), {"free": 500 * GB})())
    social_alerts.register_hooks()
    assert [(h.name, h.stage, h.rule) for h in social_hooks.hooks("maintenance")] == [("alerts_scan", "maintenance", "alerts")]
    ctx = {"drafts_dir": tmp_path, "ledger": Ledger(), "today": dt.date(2026, 10, 23)}
    social_hooks.run("maintenance", {**ctx, "rules": rules_with(alerts=False)}, rules=rules_with(alerts=False), audit=Audit())
    assert social_alerts.read_jsonl(settings.SOCIAL_ALERTS_LOG) == []
    social_hooks.run("maintenance", {**ctx, "rules": rules_with()}, rules=rules_with(), audit=Audit())
    assert [r["kind"] for r in social_alerts.read_jsonl(settings.SOCIAL_ALERTS_LOG) if r["record"] == "alert"].count("queue_empty") == 1
