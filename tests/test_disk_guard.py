"""disk_guard: the pause/resume mechanism on its own (events, hysteresis, timeout), the floors from settings, and the callers that pause
(social-prep, the live post run, the disk-guard command). Only the disk reading, the clock and the sleep are replaced, by recorders."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from bio_automation import cli, disk_guard, settings, social_prep
from bio_automation.disk_guard import GB, Floor
from bio_automation.social_rules import Audit

SYS, DATA = Path("C:/sys"), Path("D:/data")
SYS_S = str(SYS)
FLOORS = [Floor("system", SYS, 15.0), Floor("data", DATA, 20.0)]


class Disks:
    """free(path) from a script of readings per drive; every sleep advances the clock and moves to the next reading."""

    def __init__(self, c, d):
        self.c, self.d, self.i, self.t, self.sleeps = c, d, 0, 0.0, []

    def free(self, path):
        series = self.c if Path(path) == SYS else self.d
        return int(series[min(self.i, len(series) - 1)] * GB)

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s
        self.i += 1

    def clock(self):
        return self.t


def run(disks, events, *, max_wait_s=0, margin_gb=1):
    said = []
    got = disk_guard.wait_for_room(FLOORS, who="w", audit=lambda event, **f: events.append({"event": event, **f}), free=disks.free, sleep=disks.sleep,
                                   clock=disks.clock, poll_s=60, max_wait_s=max_wait_s, margin_gb=margin_gb, say=said.append)
    return got, said


def test_room_enough_returns_at_once_and_writes_nothing():
    disks, events = Disks([15.0], [20.0]), []
    got, said = run(disks, events)
    assert (got, events, said, disks.sleeps) == (True, [], [], [])


def test_a_short_drive_pauses_once_then_resumes_once_with_the_wait():
    disks, events = Disks([14.0, 14.0, 15.5, 16.0], [30.0]), []
    got, said = run(disks, events)
    assert got is True
    assert events == [
        {"event": "disk_pause", "who": "w", "short": [{"drive": "system", "path": SYS_S, "free_gb": 14.0, "need_gb": 15.0}]},
        {"event": "disk_resume", "who": "w", "waited_s": 180.0}]
    assert disks.sleeps == [60, 60, 60]
    assert said == [f"w: paused, low disk space (system {SYS_S} has 14.0 GB free, needs 15); resumes by itself at +1 GB",
                    "w: disk space is back, resuming after 180s"]


def test_resume_needs_the_floor_plus_the_margin_so_a_drive_at_the_edge_does_not_flap():
    disks, events = Disks([14.0, 15.0, 15.9, 16.0], [30.0]), []
    run(disks, events)
    assert [e["waited_s"] for e in events if e["event"] == "disk_resume"] == [180.0]      # 15.0 and 15.9 are above the floor but under floor+1


def test_with_no_margin_the_floor_itself_is_enough_to_resume():
    disks, events = Disks([14.0, 15.0], [30.0]), []
    run(disks, events, margin_gb=0)
    assert [e["waited_s"] for e in events if e["event"] == "disk_resume"] == [60.0]


def test_both_drives_are_reported_and_both_must_recover():
    disks, events = Disks([10.0, 40.0, 40.0], [5.0, 5.0, 25.0]), []
    run(disks, events)
    assert [s["drive"] for s in events[0]["short"]] == ["system", "data"]
    assert events[1] == {"event": "disk_resume", "who": "w", "waited_s": 120.0}


def test_a_wait_limit_that_runs_out_gives_up_with_a_timeout_event_and_false():
    disks, events = Disks([10.0], [30.0]), []
    got, said = run(disks, events, max_wait_s=100)
    assert got is False
    assert [e["event"] for e in events] == ["disk_pause", "disk_timeout"]
    assert events[1] == {"event": "disk_timeout", "who": "w", "waited_s": 100.0,
                         "short": [{"drive": "system", "path": SYS_S, "free_gb": 10.0, "need_gb": 16.0}]}
    assert disks.sleeps == [60, 40]                                                         # the last sleep is cut to the time left
    assert said[-1] == f"w: gave up after 100s, still low on disk space (system {SYS_S} has 10.0 GB free, needs 16)"


def test_a_wait_limit_does_not_stop_a_drive_that_recovers_in_time():
    disks, events = Disks([10.0, 20.0], [30.0]), []
    got, _ = run(disks, events, max_wait_s=100)
    assert (got, [e["event"] for e in events]) == (True, ["disk_pause", "disk_resume"])


def test_the_log_event_line_goes_to_the_guard_log(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DISK_GUARD_LOG", tmp_path / "state" / "dg.jsonl")
    rec = disk_guard.log_event("disk_pause", who="nas_pipeline", short=[])
    lines = (tmp_path / "state" / "dg.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(x) for x in lines] == [rec]
    assert (rec["event"], rec["who"], rec["short"]) == ("disk_pause", "nas_pipeline", [])


def test_default_floors_come_from_settings_and_extra_paths_take_the_data_floor(monkeypatch):
    monkeypatch.setattr(settings, "DISK_GUARD_SYSTEM_MIN_GB", 11.0)
    monkeypatch.setattr(settings, "DISK_GUARD_DATA_MIN_GB", 33.0)
    floors = disk_guard.default_floors(extra={"out": Path("F:/out")})
    assert [(f.label, f.min_gb) for f in floors] == [("system", 11.0), ("data", 33.0), ("out", 33.0)]
    assert floors[1].path == Path(settings.CACHE_DIR)
    assert disk_guard.output_floors("F:/o", 35) == [floors[0], Floor("output", Path("F:/o"), 35)]


def test_free_bytes_measures_the_nearest_existing_parent(tmp_path):
    assert disk_guard.free_bytes(tmp_path / "not" / "there") > 0


def test_the_command_exits_3_when_short_and_0_when_there_is_room(monkeypatch, capsys):
    args = argparse.Namespace(who="t", wait=False, max_wait=None, path=[])
    monkeypatch.setattr(settings, "DISK_GUARD_SYSTEM_MIN_GB", 10 ** 9)
    assert disk_guard.cmd_disk_guard(args) == 3
    assert capsys.readouterr().err.startswith("t: low disk space (system ")
    monkeypatch.setattr(settings, "DISK_GUARD_SYSTEM_MIN_GB", 0.0)
    monkeypatch.setattr(settings, "DISK_GUARD_DATA_MIN_GB", 0.0)
    assert disk_guard.cmd_disk_guard(args) == 0
    assert capsys.readouterr().out == "t: disk space ok\n"


def test_the_command_with_wait_gives_up_with_3_after_its_limit(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "DISK_GUARD_SYSTEM_MIN_GB", 10 ** 9)
    monkeypatch.setattr(settings, "DISK_GUARD_LOG", tmp_path / "dg.jsonl")
    monkeypatch.setattr(settings, "DISK_GUARD_POLL_S", 0.01)
    code = disk_guard.cmd_disk_guard(argparse.Namespace(who="t", wait=True, max_wait=0.05, path=[]))
    assert code == 3
    events = [json.loads(x)["event"] for x in (tmp_path / "dg.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events == ["disk_pause", "disk_timeout"]


# ---------------------------------------------------------------------------------------------------------------- social-prep pauses

class Steps:
    def __init__(self):
        self.calls = []

    def runners(self):
        return {n: (lambda n=n: self.calls.append(n) or 0) for n in social_prep.STEPS}


def prep(tmp_path, readings, *, wait_s):
    """readings: [(c_gb, d_gb), ...] one per sleep; the first is the state at the start."""
    audit, said, state = Audit(tmp_path / "audit.jsonl"), [], {"i": 0, "t": 0.0}
    steps = Steps()

    def free(drive):
        c, d = readings[min(state["i"], len(readings) - 1)]
        return c if drive.startswith("C") else d

    def sleep(s):
        state["i"] += 1
        state["t"] += s

    code = social_prep.run_prep(steps.runners(), audit=audit, free=free, clock=lambda: state["t"], say=said.append, wait_s=wait_s, sleep=sleep)
    return code, steps.calls, audit.records(), said


def test_prep_waits_for_room_and_then_runs_every_step(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DISK_GUARD_POLL_S", 60)
    code, calls, records, _ = prep(tmp_path, [(12.0, 50.0), (12.0, 50.0), (16.0, 50.0)], wait_s=3600)
    assert (code, calls) == (0, ["social-index", "darkroom-sync", "social-draft"])
    assert [(r["event"], r.get("step") or r.get("who")) for r in records] == [
        ("disk_pause", "social-prep"), ("disk_resume", "social-prep"), ("prep_step", "preflight"), ("prep_step", "social-index"),
        ("prep_step", "darkroom-sync"), ("prep_step", "social-draft")]
    assert records[0]["short"] == [{"drive": "C:", "path": "C:\\", "free_gb": 12.0, "need_gb": 15.0}]
    assert records[1]["waited_s"] == 120.0
    assert [r["ok"] for r in records if r["event"] == "prep_step"] == [True] * 4


def test_prep_stops_with_the_reason_when_the_wait_runs_out(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DISK_GUARD_POLL_S", 60)
    code, calls, records, said = prep(tmp_path, [(12.0, 50.0)], wait_s=100)
    assert (code, calls) == (1, [])
    assert [r["event"] for r in records] == ["disk_pause", "disk_timeout", "prep_step"]
    assert records[2]["reason"] == "C: has 12.0 GB free, social-prep needs 15 GB"
    assert said[-1] == "bio social-prep: stopped, C: has 12.0 GB free, social-prep needs 15 GB"


def test_prep_with_room_never_pauses(tmp_path):
    code, calls, records, _ = prep(tmp_path, [(15.0, 20.0)], wait_s=3600)
    assert (code, calls, [r["event"] for r in records]) == (0, ["social-index", "darkroom-sync", "social-draft"], ["prep_step"] * 4)


# ---------------------------------------------------------------------------------------------------------------- the live post run pauses

def test_the_post_run_pauses_under_its_own_low_floor_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "SOCIAL_AUDIT_LOG", tmp_path / "audit.jsonl")
    monkeypatch.setattr(settings, "SOCIAL_DIR", tmp_path / "social")
    monkeypatch.setattr(settings, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(settings, "SOCIAL_POST_MIN_FREE_GB", 2.0)
    monkeypatch.setattr(settings, "SOCIAL_POST_MAX_WAIT_S", 0.0)
    readings, t = [1.0, 1.0, 3.5], {"i": 0}
    got = cli.wait_for_post_room(free=lambda p: int(readings[min(t["i"], 2)] * GB), sleep=lambda s: t.update(i=t["i"] + 1),
                                 clock=lambda: float(t["i"] * 60), poll_s=60, margin_gb=1.0, say=lambda m: None)
    assert got is True
    events = Audit(tmp_path / "audit.jsonl").records()
    assert [(e["event"], e["who"]) for e in events] == [("disk_pause", "social-post"), ("disk_resume", "social-post")]
    assert [s["drive"] for s in events[0]["short"]] == ["system", "data", "drafts"]
    assert {s["need_gb"] for s in events[0]["short"]} == {2.0}
    assert events[1]["waited_s"] == 120.0


def test_the_post_run_gives_up_with_false_and_a_timeout_event(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "SOCIAL_AUDIT_LOG", tmp_path / "audit.jsonl")
    monkeypatch.setattr(settings, "SOCIAL_DIR", tmp_path / "social")
    monkeypatch.setattr(settings, "SOCIAL_POST_MAX_WAIT_S", 90.0)
    t = {"s": 0.0}
    got = cli.wait_for_post_room(free=lambda p: int(0.5 * GB), sleep=lambda s: t.update(s=t["s"] + s), clock=lambda: t["s"], poll_s=60, say=lambda m: None)
    assert got is False
    assert [e["event"] for e in Audit(tmp_path / "audit.jsonl").records()] == ["disk_pause", "disk_timeout"]


def test_the_post_run_with_room_writes_no_event(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "SOCIAL_AUDIT_LOG", tmp_path / "audit.jsonl")
    monkeypatch.setattr(settings, "SOCIAL_DIR", tmp_path / "social")
    assert cli.wait_for_post_room(free=lambda p: 50 * GB) is True
    assert not (tmp_path / "audit.jsonl").exists()


# ---------------------------------------------------------------------------------------------------------------- scratch space on the data drive

def test_scratch_dir_is_created_where_settings_say(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "SCRATCH_DIR", tmp_path / "x" / "scratch")
    assert settings.scratch_dir() == tmp_path / "x" / "scratch"
    assert (tmp_path / "x" / "scratch").is_dir()


def test_the_default_scratch_dir_is_under_the_cache_dir_on_the_data_drive():
    env = {k: v for k, v in os.environ.items() if k not in ("BIO_SCRATCH_DIR", "BIO_CACHE_DIR")}
    out = subprocess.run([sys.executable, "-c", "from bio_automation import settings as s; print(s.SCRATCH_DIR); print(s.CACHE_DIR)"],
                         env=env, capture_output=True, text=True, check=True).stdout.splitlines()
    assert Path(out[0]) == Path(out[1]) / "tmp"
    assert Path(out[1]) == Path("D:/bio-cache")


def test_exiftool_argfiles_are_made_in_the_scratch_dir_not_in_temp(tmp_path, monkeypatch):
    from bio_automation import social

    seen = []

    def fake_run(cmd, **kw):                      # records the argfile's place; exiftool itself is not under test
        seen.append(Path(cmd[-1]))
        return type("R", (), {"stdout": b"[]", "stderr": b""})()

    monkeypatch.setattr(social.subprocess, "run", fake_run)
    monkeypatch.setattr(social, "_exiftool", lambda: "exiftool")
    assert social.read_titles([tmp_path / "a.jpg"]) == {}
    assert [p.parent for p in seen] == [settings.SCRATCH_DIR]
    assert not seen[0].exists()                   # and it is removed again


# ---------------------------------------------------------------------------------------------------------------- the post command stops before it writes

def post_args():
    return argparse.Namespace(live=True, manual=False, draft=None)


def test_a_live_post_with_no_room_returns_1_and_posts_nothing(monkeypatch, capsys):
    from bio_automation import social

    calls = []
    monkeypatch.setattr(cli, "wait_for_post_room", lambda: False)
    monkeypatch.setattr(social, "post_next", lambda *a, **k: calls.append(1) or {"status": "dry-run"})
    assert cli.cmd_social_post(post_args()) == 1
    assert calls == []


def test_a_live_post_with_room_goes_on_to_post(monkeypatch, capsys):
    from bio_automation import social

    calls = []
    monkeypatch.setattr(cli, "wait_for_post_room", lambda: True)
    monkeypatch.setattr(social, "post_next", lambda *a, **k: calls.append(1) or {"status": "nothing approved"})
    monkeypatch.setattr(cli, "run_maintenance", lambda *a, **k: None)
    assert cli.cmd_social_post(post_args()) == 0
    assert calls == [1]
