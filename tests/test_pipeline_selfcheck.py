"""bio_automation.pipeline_selfcheck: the read-only checks behind `bio pipeline-selfcheck` and the deploy rollback gate."""
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from argparse import Namespace

import pytest

from bio_automation import pipeline_selfcheck as sc
from bio_automation import topics, worker_runtime
from bio_automation.queue import MIGRATIONS, Queue, QueueError

NOW = 100000.0


class Clock:
    def __init__(self, now=NOW):
        self.now = now

    def __call__(self):
        return self.now


def by_name(checks):
    return {c.name: c for c in checks}


def done(stdout="", stderr="", rc=0):
    return subprocess.CompletedProcess([], rc, stdout, stderr)


class TestReadonlyQueue:
    def test_a_readonly_handle_reads_and_refuses_to_write(self, tmp_path):
        path = tmp_path / "q.sqlite3"
        with Queue(path, clock=Clock()) as q:
            q.publish("t", {"n": 1})
        with Queue(path, readonly=True) as ro:
            assert ro.counts() == {"pending": 1}
            with pytest.raises(sqlite3.OperationalError, match="readonly database"):
                ro.publish("t", {"n": 2})

    def test_a_missing_database_is_not_created(self, tmp_path):
        with pytest.raises(QueueError, match="does not exist"):
            Queue(tmp_path / "sub" / "none.sqlite3", readonly=True)
        assert not (tmp_path / "sub").exists()

    def test_an_unmigrated_database_is_read_without_being_migrated(self, tmp_path):
        path = tmp_path / "v3.sqlite3"
        conn = sqlite3.connect(str(path))
        for m in MIGRATIONS[:3]:
            conn.executescript(m)
        conn.execute("INSERT INTO messages (topic, payload, status, created) VALUES ('t', '{}', 'pending', 1.0)")
        conn.execute("PRAGMA user_version=3")
        conn.commit()
        conn.close()
        with Queue(path, clock=Clock(), readonly=True) as ro:
            assert ro.schema_version() == 3
            assert ro.health() == {"counts": {"pending": 1}, "dead": 0, "retrying": 0, "expired_claims": 0,
                                   "oldest_claimed_age_s": None, "oldest_pending_age_s": NOW - 1.0}
        assert sqlite3.connect(str(path)).execute("PRAGMA user_version").fetchone()[0] == 3


class TestMigrationRollback:
    def test_a_database_rolled_back_to_version_3_migrates_again(self, tmp_path):
        path = tmp_path / "q.sqlite3"
        with Queue(path, clock=Clock()) as q:
            mid = q.publish("t", {"n": 1})
            q.claim(["t"], worker_id="w")
            q.retry(mid, "x", worker_id="w", delay_s=5.0)
        conn = sqlite3.connect(str(path))
        conn.execute("PRAGMA user_version=3")                      # the documented rollback for older code
        conn.close()
        with Queue(path, clock=Clock()) as q:
            assert q.schema_version() == 5
            assert q.counts() == {"pending": 1}


class TestQueueAndDepth:
    def test_a_missing_database_is_a_warning_not_a_crash(self, tmp_path):
        q, check = sc.open_queue(tmp_path / "none.sqlite3")
        assert q is None and check.level == "warn" and "does not exist" in check.message

    def test_a_database_newer_than_the_code_fails_and_says_how_to_roll_back(self, tmp_path):
        path = tmp_path / "q.sqlite3"
        Queue(path).close()
        conn = sqlite3.connect(str(path))
        conn.execute(f"PRAGMA user_version={len(MIGRATIONS) + 1}")
        conn.close()
        q, check = sc.open_queue(path)
        assert q is None and check.level == "fail"
        assert f"schema version {len(MIGRATIONS) + 1}, newer than this code understands ({len(MIGRATIONS)})" in check.message
        assert "PRAGMA user_version=3" in check.message

    def test_depth_is_ok_with_pending_work_and_warns_on_a_dead_letter_with_its_reason(self, tmp_path):
        clock = Clock()
        with Queue(tmp_path / "q.sqlite3", clock=clock) as q:
            mid = q.publish("t", {"n": 1})
            assert sc.check_depth(q) == sc.Check("depth", "ok", "pending=1; dead-lettered=0, retrying=0")
            q.claim(["t"], worker_id="w")
            q.retry(mid, "CUDA error 999", worker_id="w", delay_s=0.0, max_attempts=1)
            check = sc.check_depth(q)
            assert check.level == "warn"
            assert check.message.startswith("empty; dead-lettered=1, retrying=0; 1 failed or dead-lettered (newest: 1 attempts used up; last error: CUDA error 999)")


class TestStuck:
    def make(self, tmp_path, lease_s, advance):
        clock = Clock()
        q = Queue(tmp_path / "q.sqlite3", clock=clock)
        q.publish("t", {"n": 1})
        q.claim(["t"], worker_id="w", lease_s=lease_s)
        clock.now += advance
        return q

    def test_a_claim_inside_its_lease_is_ok(self, tmp_path):
        q = self.make(tmp_path, 3600.0, 600.0)
        assert sc.check_stuck(q) == sc.Check("stuck", "ok", "nothing claimed longer than a lease; oldest claim 10.0 min")

    def test_an_expired_lease_is_reported_for_the_reaper(self, tmp_path):
        q = self.make(tmp_path, 60.0, 600.0)
        check = sc.check_stuck(q)
        assert check.level == "warn" and check.message.startswith("1 claimed messages have an expired lease (their worker died)")

    def test_a_claim_older_than_any_live_lease_warns_with_its_age(self, tmp_path):
        q = self.make(tmp_path, 10 ** 6, 7 * 3600.0)
        check = sc.check_stuck(q)
        assert check.level == "warn" and check.message == "a message has been claimed for 7.0 h, longer than any live worker keeps a lease (6.0 h)"


class TestWorkers:
    def beat(self, tmp_path, worker_id, stage, pid, ts, state="idle"):
        hb = worker_runtime.Heartbeat(tmp_path, worker_id, stage, clock=lambda: ts, pid=pid)
        hb.beat(state, force=True)

    def test_no_heartbeats_is_a_warning(self, tmp_path):
        assert sc.check_workers(None, tmp_path / "hb", now=NOW).level == "warn"

    def test_live_recent_workers_are_ok(self, tmp_path):
        self.beat(tmp_path, "nas-tag-1", "tag", 1, NOW - 5)
        check = sc.check_workers(None, tmp_path, now=NOW, alive=lambda pid: True)
        assert check == sc.Check("workers", "ok", "1 workers alive: nas-tag-1")

    def test_a_dead_pid_and_a_silent_worker_are_named(self, tmp_path):
        self.beat(tmp_path, "nas-tag-1", "tag", 1, NOW - 30)
        self.beat(tmp_path, "nas-publish-2", "publish", 2, NOW - 500)
        check = sc.check_workers(None, tmp_path, now=NOW, alive=lambda pid: pid == 2)
        assert check.level == "warn"
        assert check.message == "nas-publish-2: alive but silent for 8.3 min (state idle); nas-tag-1: pid 1 is gone (last beat 30 s ago)"

    def test_a_long_handler_is_not_silent_while_working(self, tmp_path):
        self.beat(tmp_path, "nas-lightcraft-1", "lightcraft", 1, NOW - 4000, state="working")
        assert sc.check_workers(None, tmp_path, now=NOW, alive=lambda pid: True).level == "ok"

    def test_work_waiting_for_a_stage_with_no_live_worker_warns(self, tmp_path):
        self.beat(tmp_path, "nas-tag-1", "tag", 1, NOW - 5)
        with Queue(tmp_path / "q.sqlite3", clock=Clock()) as q:
            q.publish(topics.STOCK_RENDERED, {"day": "d", "csv": "c", "render_dir": "r"})
            check = sc.check_workers(q, tmp_path, now=NOW, alive=lambda pid: True)
            assert check.level == "warn" and check.message == "1 messages wait for stage publish, which has no live worker"
            self.beat(tmp_path, "nas-publish-3", "publish", 3, NOW - 5)
            assert sc.check_workers(q, tmp_path, now=NOW, alive=lambda pid: True).level == "ok"


class TestCrashLoop:
    def write(self, tmp_path, state):
        (tmp_path / "worker_guard.json").write_text(json.dumps(state), encoding="utf-8")

    def test_no_record_is_ok(self, tmp_path):
        assert sc.check_crash_loop(tmp_path, now=NOW).level == "ok"

    def test_below_the_alert_count_is_ok_and_at_it_fails(self, tmp_path):
        self.write(tmp_path, {"render-0": {"quick_exits": 2, "counted": True, "last_start": NOW - 50}})
        assert sc.check_crash_loop(tmp_path, now=NOW).level == "ok"
        self.write(tmp_path, {"render-0": {"quick_exits": 3, "counted": True, "last_start": NOW - 50},
                              "tag-1": {"quick_exits": 9, "counted": False, "last_start": NOW - 10}})
        check = sc.check_crash_loop(tmp_path, now=NOW)
        assert check.level == "fail" and "render-0 x3, tag-1 x9" in check.message

    def test_a_worker_that_has_stayed_up_since_its_last_start_is_not_looping(self, tmp_path):
        self.write(tmp_path, {"render-0": {"quick_exits": 5, "counted": False, "last_start": NOW - 600}})
        assert sc.check_crash_loop(tmp_path, now=NOW).level == "ok"
        self.write(tmp_path, {"render-0": {"quick_exits": 5, "counted": False, "last_start": NOW - 30}})
        assert sc.check_crash_loop(tmp_path, now=NOW).level == "fail"

    def test_a_torn_file_is_a_warning(self, tmp_path):
        (tmp_path / "worker_guard.json").write_text("{", encoding="utf-8")
        assert sc.check_crash_loop(tmp_path, now=NOW).level == "warn"


class TestGpu:
    def test_a_listed_card_is_ok(self):
        check = sc.check_gpu(lambda: done("NVIDIA GeForce RTX 4070 Ti, 6367 MiB, 12282 MiB\n"))
        assert check == sc.Check("gpu", "ok", "1 GPU(s) reachable: NVIDIA GeForce RTX 4070 Ti, 6367 MiB, 12282 MiB")

    def test_a_failing_or_empty_smi_fails(self):
        assert sc.check_gpu(lambda: done("", "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver", rc=9)).level == "fail"
        assert sc.check_gpu(lambda: done("", "", rc=0)).level == "fail"

    def test_a_hung_driver_fails_and_a_missing_tool_warns(self):
        def hang():
            raise subprocess.TimeoutExpired("nvidia-smi", 20)

        def missing():
            raise FileNotFoundError("nvidia-smi")
        assert sc.check_gpu(hang).level == "fail"
        assert sc.check_gpu(missing).level == "warn"


class TestInatPace:
    def make(self, tmp_path, next_slot):
        path = tmp_path / "rate.sqlite3"
        con = sqlite3.connect(str(path))
        con.execute("CREATE TABLE pace (k INTEGER PRIMARY KEY CHECK (k = 0), next_slot REAL NOT NULL)")
        con.execute("INSERT INTO pace VALUES (0, ?)", (next_slot,))
        con.commit()
        con.close()
        return path

    def test_no_file_and_a_near_slot_are_ok(self, tmp_path):
        assert sc.check_inat_pace(tmp_path / "none.sqlite3", now=NOW).level == "ok"
        assert sc.check_inat_pace(self.make(tmp_path, NOW + 2.0), now=NOW).message == "iNat pace file is current (next slot in 2.0 s)"

    def test_a_long_hold_off_warns_with_the_remaining_time(self, tmp_path):
        check = sc.check_inat_pace(self.make(tmp_path, NOW + 1800.0), now=NOW)
        assert check.level == "warn" and check.message.startswith("every iNat request is held off for another 30.0 min")


class TestImports:
    def test_the_worker_modules_import(self):
        assert sc.check_imports().level == "ok"

    def test_a_module_that_fails_to_import_fails_the_check_naming_it(self, monkeypatch):
        real = sc.importlib.import_module

        def boom(name, *a):
            if name.endswith(".lightcraft_mcp"):
                raise ImportError("No module named 'syntax_error_here'")
            return real(name, *a)
        monkeypatch.setattr(sc.importlib, "import_module", boom)
        check = sc.check_imports()
        assert check.level == "fail" and "lightcraft_mcp (ImportError: No module named 'syntax_error_here')" in check.message

    def test_the_ops_repo_is_import_checked_in_a_fresh_interpreter(self, tmp_path):
        ops = tmp_path / "ops"
        ops.mkdir()
        (ops / "nas_pipeline.py").write_text("import bio_automation\n", encoding="utf-8")
        (ops / "pipeline_upkeep.py").write_text("", encoding="utf-8")
        assert sc.check_imports(ops).level == "ok"
        (ops / "nas_pipeline.py").write_text("import module_that_is_not_there\n", encoding="utf-8")
        check = sc.check_imports(ops)
        assert check.level == "fail" and "ModuleNotFoundError: No module named 'module_that_is_not_there'" in check.message

    def test_a_missing_ops_script_fails(self, tmp_path):
        assert sc.check_imports(tmp_path).level == "fail"


class TestDisks:
    def test_levels_follow_the_floor(self):
        gb = 1024 ** 3
        checks = sc.check_disks({"C": "c", "D": "d"}, free=lambda p: {"c": 14 * gb, "d": 10 * gb}[p])
        assert [(c.name, c.level) for c in checks] == [("disk-C", "fail"), ("disk-D", "ok")]
        assert sc.check_disks({"C": "c"}, free=lambda p: 20 * gb)[0].level == "warn"          # inside 1.5 x the floor
        assert sc.check_disks({"C": "c"}, free=lambda p: 23 * gb)[0].level == "ok"


class TestRunAndDeploy:
    def run(self, tmp_path, **kw):
        db = tmp_path / "q.sqlite3"
        Queue(db).close()
        gb = 1024 ** 3
        return sc.run_checks(db=db, pipeline_dir=tmp_path / "pipe", now=NOW, gpu_probe=lambda: done("RTX, 1 MiB, 2 MiB\n"),
                             drives={"C": tmp_path, "D": tmp_path}, free=lambda p: 100 * gb, rate_db=tmp_path / "none.sqlite3", **kw)

    def test_a_healthy_machine_with_no_workers_is_a_warning_only_for_the_workers(self, tmp_path):
        checks = self.run(tmp_path)
        assert [(c.name, c.level) for c in checks] == [
            ("imports", "ok"), ("queue", "ok"), ("depth", "ok"), ("stuck", "ok"), ("workers", "warn"), ("crash-loop", "ok"), ("gpu", "ok"),
            ("inat-pace", "ok"), ("disk-C", "ok"), ("disk-D", "ok")]
        assert sc.worst(checks) == "warn"

    def test_deploy_mode_downgrades_machine_state_failures_but_not_code_failures(self):
        checks = [sc.Check("gpu", "fail", "x"), sc.Check("crash-loop", "fail", "y"), sc.Check("imports", "fail", "z"), sc.Check("queue", "fail", "w")]
        out = {c.name: c.level for c in sc.for_deploy(checks)}
        assert out == {"gpu": "warn", "crash-loop": "warn", "imports": "fail", "queue": "fail"}
        assert sc.for_deploy([sc.Check("gpu", "fail", "x")])[0].message == "x [environmental: a warning in --deploy mode]"

    def test_the_command_prints_the_summary_line_last_and_returns_the_exit_code(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(sc, "run_checks", lambda **kw: [sc.Check("a", "ok", "fine"), sc.Check("b", "warn", "meh")])
        assert sc.cmd_pipeline_selfcheck(Namespace(deploy=True, json=False, ops_dir=None)) == 1
        lines = capsys.readouterr().out.splitlines()
        assert lines == ["[ok  ] a: fine", "[warn] b: meh", "selfcheck: warn"]

    def test_a_crash_is_a_fail_in_the_form_the_deploy_script_reads(self, monkeypatch, capsys):
        def boom(**kw):
            raise RuntimeError("kaboom")
        monkeypatch.setattr(sc, "run_checks", boom)
        assert sc.cmd_pipeline_selfcheck(Namespace(deploy=True, json=False, ops_dir=None)) == 2
        captured = capsys.readouterr()
        assert captured.out.splitlines()[-1] == "selfcheck: fail (RuntimeError)"
        assert "selfcheck crashed: RuntimeError: kaboom" in captured.err

    def test_json_output_carries_the_level_and_every_check(self, monkeypatch, capsys):
        monkeypatch.setattr(sc, "run_checks", lambda **kw: [sc.Check("a", "ok", "fine")])
        assert sc.cmd_pipeline_selfcheck(Namespace(deploy=False, json=True, ops_dir=None)) == 0
        assert json.loads(capsys.readouterr().out) == {"level": "ok", "exit": 0, "checks": [{"name": "a", "level": "ok", "message": "fine"}]}

    def test_the_real_command_line_runs_and_ends_with_the_summary(self, tmp_path):
        import os
        from pathlib import Path

        env = {**os.environ, "PYTHONPATH": str(Path(sc.__file__).resolve().parent.parent), "BIO_QUEUE_DB": str(tmp_path / "none.sqlite3"),
               "BIO_NAS_PIPELINE_DIR": str(tmp_path / "pipe"), "BIO_INAT_RATE_DB": str(tmp_path / "rate.sqlite3")}
        done_ = subprocess.run([sys.executable, "-m", "bio_automation.cli", "pipeline-selfcheck", "--deploy"], capture_output=True, text=True, env=env, timeout=120)
        lines = done_.stdout.strip().splitlines()
        assert lines[-1] in ("selfcheck: ok", "selfcheck: warn") and done_.returncode == (0 if lines[-1] == "selfcheck: ok" else 1)
        assert lines[0].startswith("[ok  ] imports:") and any(ln.startswith("[warn] queue:") for ln in lines)
