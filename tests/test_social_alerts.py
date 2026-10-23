"""social_alerts: the pull-model alert scanner and the daily digest. Synthetic audit, ledger, taxonomy and log files under tmp_path; the alert
command is a recording runner, never a real process."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
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


YESTERDAY = [{"at": at(h, 5, day=22), "event": "draft_run"} for h in (8, 12, 20)]       # yesterday's three runs left a trace: only today's slots are under test


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
        self.set_audit([*YESTERDAY, {"at": at(8, 16), "event": "posted", "draft_id": "a"}, {"at": at(12, 5), "event": "posted", "draft_id": "b"}])
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
                                         ("paused", "warning"), ("backup_failed", "warning"), ("live_recheck_unavailable", "warning"),
                                         ("draft_stalled", "error")])
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
    env.set_audit([*YESTERDAY, {"at": at(8, 1), "event": "post_skipped"}, {"at": at(12, 1), "event": "post_skipped"}])      # runs happened, nothing posted


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
    env.set_audit(YESTERDAY)
    env.set_ledger([{"at": at(7, 0), "event": "drafted", "draft_id": "c", "obs_id": 3, "month": "2026-10"}])
    return dt.datetime(2026, 10, 23, 8, 31, tzinfo=TZ)


def test_a_run_with_no_audit_record_and_no_log_write_is_silent(env):
    now = silent_env(env)
    alerts, _ = env.scan(now=now)
    assert [(a.kind, a.level, a.key) for a in alerts] == [("run_silent", "error", "run_silent:2026-10-23:08:00")]
    assert alerts[0].message == "the 08:00 run left no audit record and no [post]/[prep] line in social.log by 08:31"


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


def test_silent_run_twin_a_post_line_in_social_log_in_the_slot(env):
    now = silent_env(env)
    env.log.write_text("2026-10-23T08:00:03 [post] run start (pid 1, user x)\n", encoding="utf-8")
    assert env.kinds(now=now) == []
    env.log.write_text("2026-10-22T20:00:03 [post] run start (pid 1, user x)\n", encoding="utf-8")
    assert env.kinds(now=now) == ["run_silent"]


def test_a_prep_line_counts_too(env):
    now = silent_env(env)
    env.log.write_text("2026-10-23T08:02:00 [prep] run start (pid 1, user x)\n", encoding="utf-8")
    assert env.kinds(now=now) == []


def test_the_watch_scripts_own_log_lines_and_the_files_mtime_do_not_hide_a_silent_run(env):
    now = silent_env(env)
    env.log.write_text("2026-10-23T08:20:00 [watch] run start, step alerts (pid 1, user x)\nsome python output line\n", encoding="utf-8")
    stamp = dt.datetime(2026, 10, 23, 8, 20, tzinfo=TZ).timestamp()
    os.utime(env.log, (stamp, stamp))
    assert env.kinds(now=now) == ["run_silent"]


def test_a_silent_late_slot_of_yesterday_is_found_after_midnight(env):
    env.set_audit([*YESTERDAY, {"at": at(8, 5, day=23), "event": "draft_run"}, {"at": at(12, 5, day=23), "event": "draft_run"}])
    env.set_ledger([{"at": at(7, 0, day=23), "event": "drafted", "draft_id": "c", "obs_id": 3, "month": "2026-10"}])
    alerts, _ = env.scan(now=dt.datetime(2026, 10, 24, 0, 10, tzinfo=TZ))
    assert [(a.kind, a.key) for a in alerts if a.kind == "run_silent"] == [("run_silent", "run_silent:2026-10-23:20:00")]


def test_yesterdays_late_slot_is_quiet_when_it_ran(env):
    env.set_audit([{"at": at(8, 5, day=23), "event": "draft_run"}, {"at": at(12, 5, day=23), "event": "draft_run"},
                   {"at": at(20, 3, day=23), "event": "draft_run"}])
    env.set_ledger([{"at": at(7, 0, day=23), "event": "drafted", "draft_id": "c", "obs_id": 3, "month": "2026-10"}])
    assert "run_silent" not in env.kinds(now=dt.datetime(2026, 10, 24, 0, 10, tzinfo=TZ))


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
    assert [c[1]["input"] for c in runner.calls] == [f"[error] post_failed: {rep['new'][0].message}\n".encode("utf-8"),
                                                         f"[warning] backup_failed: {rep['new'][1].message}\n".encode("utf-8")]
    assert [c[1]["timeout"] for c in runner.calls] == [30, 30]
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
    (RecordingRunner(raises=subprocess.TimeoutExpired("notify.exe", 30)), "timed out after 30 s; the command and its child processes were killed"),
])
def test_a_failing_command_is_recorded_and_never_raised(env, runner, expected):
    env.add_audit({"at": at(12, 6), "event": "post_failed", "draft_id": "b"})
    rep = run(env, runner, notify=True, command="notify.exe")
    assert [a.kind for a in rep["notify_failed"]] == ["post_failed"] and rep["notified"] == []
    rows = social_alerts.read_jsonl(env.alerts)
    assert [(r["record"], r.get("error")) for r in rows] == [("alert", None), ("notify_failed", expected)]
    assert rows[1]["key"] == rows[0]["key"]


def three_alerts(env):
    two_alerts(env)
    env.add_audit({"at": at(12, 8), "event": "hook_error", "name": "h", "error": "x"})


def test_notifying_stops_after_the_first_failure_and_the_rest_wait_for_the_next_scan(env):
    three_alerts(env)
    failing = RecordingRunner(returncode=1, stderr="down")
    rep = run(env, failing, notify=True, command="notify.exe")
    assert len(failing.calls) == 1 and len(rep["notify_failed"]) == 1 and rep["notified"] == []
    assert rep["notes"][-1].endswith("; 2 alert(s) wait for the next scan") and rep["notes"][-1].startswith("notification stopped after the first failure")
    assert [r["event"] for r in social_alerts.read_jsonl(env.audit) if r["event"] in ("alert_sent", "alert_failed")] == ["alert_failed"]
    healthy = RecordingRunner()
    rep2 = run(env, healthy, notify=True, command="notify.exe")
    assert len(healthy.calls) == 3 and len(rep2["notified"]) == 3                      # all three are still pending and now go out


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class SlowRunner(RecordingRunner):
    def __init__(self, clock, seconds):
        super().__init__()
        self.clock, self.seconds = clock, seconds

    def __call__(self, argv, **kw):
        self.clock.t += self.seconds
        return super().__call__(argv, **kw)


def test_the_notify_budget_is_ninety_seconds_in_total_and_each_command_gets_what_is_left(env):
    three_alerts(env)
    clock = FakeClock()
    runner = SlowRunner(clock, 40)
    rep = run(env, runner, notify=True, command="notify.exe", clock=clock)
    assert [c[1]["timeout"] for c in runner.calls] == [30, 30, 10.0] and len(rep["notified"]) == 3      # 0, 40, 80 s used: the third gets the 10 s left


def test_a_scan_that_used_its_budget_leaves_the_rest_pending(env):
    three_alerts(env)
    clock = FakeClock()
    runner = SlowRunner(clock, 50)
    rep = run(env, runner, notify=True, command="notify.exe", clock=clock)
    assert [c[1]["timeout"] for c in runner.calls] == [30, 30] and len(rep["notified"]) == 2
    assert rep["notes"][-1] == "notification stopped: the 90 s budget is used up; 1 alert(s) wait for the next scan"
    again = RecordingRunner()
    assert len(run(env, again, notify=True, command="notify.exe")["notified"]) == 1


def test_a_failed_notification_is_retried_by_the_next_scan_and_then_not_again(env):
    env.add_audit({"at": at(12, 6), "event": "post_failed", "draft_id": "b"})
    run(env, RecordingRunner(returncode=1), notify=True, command="notify.exe")
    retry = RecordingRunner()
    rep = run(env, retry, notify=True, command="notify.exe")
    assert len(retry.calls) == 1 and [a.kind for a in rep["notified"]] == ["post_failed"] and rep["new"] == []
    again = RecordingRunner()
    run(env, again, notify=True, command="notify.exe")
    assert again.calls == []
    assert [r["record"] for r in social_alerts.read_jsonl(env.alerts)] == ["alert", "notify_failed", "notified"]


def test_an_alert_logged_without_a_command_is_sent_by_the_next_notify_run(env):
    """The 03:00 maintenance scan logs alerts with no command; the watch task's --notify scan must still deliver them."""
    env.add_audit({"at": at(12, 6), "event": "post_failed", "draft_id": "b"})
    silent = RecordingRunner()
    run(env, silent, command="")
    assert silent.calls == [] and [r["record"] for r in social_alerts.read_jsonl(env.alerts)] == ["alert"]
    runner = RecordingRunner()
    rep = run(env, runner, notify=True, command="notify.exe")
    assert len(runner.calls) == 1 and rep["new"] == [] and [a.kind for a in rep["notified"]] == ["post_failed"]


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


# --------------------------------------------------------------------------------------------------- new alert kinds

def prep_rec(h, m, **kw):
    return {"at": at(h, m), "event": "prep_step", "step": "social-draft", "seconds": 5.0, **kw}


def test_a_failed_prep_step_is_an_error_alert_and_a_good_one_is_not(env):
    env.add_audit(prep_rec(3, 5, ok=True))
    assert env.kinds() == []
    env.add_audit(prep_rec(3, 9, ok=False, error="Ollama unreachable"))
    alerts, _ = env.scan()
    assert [(a.kind, a.level, a.key) for a in alerts] == [("prep_failed", "error", f"prep_failed:social-draft:{at(3, 9)}")]
    assert alerts[0].message == f"prep step social-draft failed: Ollama unreachable ({at(3, 9)})"


def test_a_selfcheck_with_level_fail_is_an_error_alert_and_warn_or_ok_are_not(env):
    for n, level in enumerate(("ok", "warn")):
        env.add_audit({"at": at(3, n), "event": "selfcheck", "level": level, "problems": []})
    assert env.kinds() == []
    env.add_audit({"at": at(3, 9), "event": "selfcheck", "level": "fail", "problems": ["rules: broken", "ledger: torn"]})
    alerts, _ = env.scan()
    assert [(a.kind, a.level) for a in alerts] == [("selfcheck_failed", "error")]
    assert alerts[0].message == f"selfcheck failed: rules: broken; ledger: torn ({at(3, 9)})"


def withdrawals(env, n, event="withdrawn", start_hour=9):
    for i in range(n):
        env.add_audit({"at": at(start_hour, i), "event": event, "draft_id": f"w{start_hour}-{i}", "reason": "casual now"})


def test_withdrawal_spike_edge_is_more_than_three_in_24_hours(env):
    withdrawals(env, 3)
    assert env.kinds() == []
    withdrawals(env, 1, start_hour=10)
    alerts, _ = env.scan()
    assert [(a.kind, a.level, a.key) for a in alerts] == [("withdrawal_spike", "warning", "withdrawal_spike:2026-10-23")]
    assert alerts[0].message == "4 drafts withdrawn (or would be, in a dry run) in the last 24 h, more than 3"


def test_would_withdraw_counts_toward_the_spike_and_a_mix_adds_up(env):
    withdrawals(env, 2, "withdrawn")
    withdrawals(env, 2, "would_withdraw", start_hour=10)
    assert env.kinds() == ["withdrawal_spike"]
    # twin: the same four, but one is an unrelated event
    env.set_audit([r for r in social_alerts.read_jsonl(env.audit) if r["event"] != "would_withdraw"])
    assert env.kinds() == []


def test_withdrawals_older_than_24_hours_do_not_count(env):
    for i in range(4):
        env.add_audit({"at": at(13, 59, day=22), "event": "withdrawn", "draft_id": f"w{i}"})
    assert env.kinds(since=dt.timedelta(days=3)) == []
    for i in range(4):
        env.add_audit({"at": at(14, 0, day=22), "event": "withdrawn", "draft_id": f"x{i}"})
    assert env.kinds(since=dt.timedelta(days=3)) == ["withdrawal_spike"]


def rejects(env, run_id, n, reason="observation {i}: image is 800 px, under the floor", start=0):
    for i in range(start, start + n):
        env.add_audit({"at": at(3, i % 60), "run": run_id, "event": "draft_skipped", "obs_id": i, "reason": reason.format(i=i)})


def test_draft_reject_spike_edge_is_more_than_ten_hook_rejects_in_one_run(env):
    rejects(env, "r1", 10)
    assert env.kinds() == []
    rejects(env, "r1", 1, start=10)
    alerts, _ = env.scan()
    assert [(a.kind, a.level, a.key) for a in alerts] == [("draft_reject_spike", "warning", "draft_reject_spike:r1")]
    assert alerts[0].message == "run r1: 11 candidates rejected by draft hooks, more than 10"


def test_draft_rejects_are_counted_per_run_and_only_hook_rejects_count(env):
    rejects(env, "r1", 6)
    rejects(env, "r2", 6, start=100)
    rejects(env, "r3", 20, reason="observation {i} is casual")                    # not a hook reject: no colon after the number
    assert env.kinds() == []
    rejects(env, "r1", 5, start=200)
    assert env.kinds() == ["draft_reject_spike"]


def test_a_torn_ledger_line_is_an_error_alert_even_without_an_audit_record(env):
    assert env.kinds() == []
    with open(env.ledger, "a", encoding="utf-8") as f:
        f.write('{"at": "2026-10-23T14:00:00-04:00", "event": "post')               # a cut-off write, no newline
    alerts, _ = env.scan()
    assert [(a.kind, a.level, a.key) for a in alerts] == [("ledger_corrupt", "error", "ledger_corrupt:file:1")]
    assert "1 unreadable line(s)" in alerts[0].message
    assert social_alerts.exit_code(alerts) == 2


def test_count_unreadable_lines_ignores_blank_lines_and_a_missing_file(tmp_path):
    p = tmp_path / "l.jsonl"
    assert social_alerts.count_unreadable_lines(p) == 0
    p.write_text('{"a": 1}\n\n[1, 2]\nnot json\n{"b": 2}\n', encoding="utf-8")
    assert social_alerts.count_unreadable_lines(p) == 2                            # a JSON array is not a record, "not json" is not JSON


def test_an_unverified_post_is_a_warning_and_a_wrong_one_an_error(env):
    env.add_audit({"at": at(12, 6), "event": "post_verify_failed", "draft_id": "b", "level": "warn",
                   "problems": ["the AppView does not have the post after 10 attempts"]})
    alerts, _ = env.scan()
    assert [(a.kind, a.level) for a in alerts] == [("post_verify_failed", "warning")]
    assert "the AppView does not have the post after 10 attempts" in alerts[0].message
    env.add_audit({"at": at(12, 7), "event": "post_verify_failed", "draft_id": "b", "level": "error", "problems": ["text differs"]})
    env.add_audit({"at": at(12, 8), "event": "post_verify_failed", "draft_id": "b", "problems": ["old record without a level"]})
    assert sorted(a.level for a in env.scan()[0]) == ["error", "error", "warning"]


def test_the_digest_lists_would_withdraw_among_the_skips():
    text = social_alerts.digest_text(dt.date(2026, 10, 23), audit=[{"at": at(9, 0), "event": "would_withdraw", "draft_id": "d1", "reason": "casual now"}],
                                     ledger=[], alerts_log=[], rules=rules_with(), pool=None, pool_note="x")
    assert "- 09:00 would_withdraw d1: casual now" in text


# --------------------------------------------------------------------------------------------------- run_notifier, for real

def pid_alive(pid: int) -> bool:
    if sys.platform == "win32":
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, pid)                                    # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        k32.GetExitCodeProcess(h, ctypes.byref(code))
        k32.CloseHandle(h)
        return code.value == 259                                                   # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def child_script(tmp_path: Path, body: str) -> list[str]:
    script = tmp_path / "notifier.py"
    script.write_text(body, encoding="utf-8")
    return [sys.executable, str(script)]


def test_run_notifier_feeds_stdin_as_utf8_bytes_and_reports_exit_code_and_stderr(tmp_path):
    out = tmp_path / "got.bin"
    argv = child_script(tmp_path, "import sys\nopen(sys.argv[1], 'wb').write(sys.stdin.buffer.read())\nsys.stderr.write('bad token')\nsys.exit(3)\n") + [str(out)]
    msg = "[error] post_failed: café 日本 \U0001F344\n"
    res = social_alerts.run_notifier(argv, input=msg.encode("utf-8"), timeout=30)
    assert out.read_bytes() == msg.encode("utf-8")
    assert (res.returncode, res.stderr) == (3, "bad token")


def test_run_notifier_success_has_exit_code_zero(tmp_path):
    res = social_alerts.run_notifier(child_script(tmp_path, "import sys\nsys.stdin.buffer.read()\n"), input=b"x\n", timeout=30)
    assert (res.returncode, res.stderr) == (0, "")


def test_run_notifier_timeout_is_a_real_bound_and_kills_the_child_and_its_grandchild(tmp_path):
    import time
    pidfile = tmp_path / "grandchild.pid"
    body = ("import subprocess, sys, time\n"
            "g = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"        # inherits stdout and stderr
            "open(sys.argv[1], 'w').write(str(g.pid))\n"
            "time.sleep(120)\n")
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        social_alerts.run_notifier(child_script(tmp_path, body) + [str(pidfile)], input=b"x\n", timeout=4)
    assert time.monotonic() - started < 25
    grandchild = int(pidfile.read_text())
    deadline = time.monotonic() + 10
    while pid_alive(grandchild) and time.monotonic() < deadline:
        time.sleep(0.2)
    assert not pid_alive(grandchild)


def test_run_notifier_returns_at_once_when_a_grandchild_outlives_a_finished_command(tmp_path):
    import time
    pidfile = tmp_path / "late.pid"
    body = ("import subprocess, sys\n"
            "g = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(8)'])\n"
            "open(sys.argv[1], 'w').write(str(g.pid))\n")                                            # exits at once, grandchild keeps running
    started = time.monotonic()
    res = social_alerts.run_notifier(child_script(tmp_path, body) + [str(pidfile)], input=b"x\n", timeout=30)
    assert res.returncode == 0 and time.monotonic() - started < 6
    deadline = time.monotonic() + 15
    while pid_alive(int(pidfile.read_text())) and time.monotonic() < deadline:
        time.sleep(0.2)
    assert not pid_alive(int(pidfile.read_text()))


def test_alert_sent_and_alert_failed_are_audited(env):
    env.add_audit({"at": at(12, 6), "event": "post_failed", "draft_id": "b"})
    run(env, RecordingRunner(returncode=2, stderr="no inbox\n"), notify=True, command="notify.exe")
    run(env, RecordingRunner(), notify=True, command="notify.exe")
    rows = [r for r in social_alerts.read_jsonl(env.audit) if r["event"] in ("alert_sent", "alert_failed")]
    assert [(r["event"], r["kind"], r["level"], r.get("reason")) for r in rows] == [
        ("alert_failed", "post_failed", "error", "exit code 2: no inbox"), ("alert_sent", "post_failed", "error", None)]


def test_a_message_with_characters_outside_the_locale_code_page_reaches_the_command_as_utf8(env):
    env.add_audit({"at": at(12, 6), "event": "post_failed", "draft_id": "b", "error": "caf\u00e9 \u65e5\u672c \U0001F344"})
    runner = RecordingRunner()
    rep = run(env, runner, notify=True, command="notify.exe")
    assert runner.calls[0][1]["input"] == f"[error] post_failed: {rep['new'][0].message}\n".encode("utf-8")
    assert "caf\u00e9 \u65e5\u672c \U0001F344" in runner.calls[0][1]["input"].decode("utf-8")
    assert [a.kind for a in rep["notified"]] == ["post_failed"] and rep["notify_failed"] == []


def test_expected_post_missing_twin_remote_pause_marker(env, tmp_path):
    missing_env(env)
    remote = tmp_path / "synced" / "PAUSE-BIO"
    assert env.kinds(pause_sync_file=remote) == ["expected_post_missing"]            # twin: configured, but no marker (folder missing too)
    remote.parent.mkdir()
    remote.write_text("", encoding="utf-8")
    assert env.kinds(pause_sync_file=remote) == []                                   # the remote marker pauses like the local file
    assert env.kinds(pause_sync_file=remote, rules=rules_with(pause_switch=False)) == ["expected_post_missing"]
