"""bio_automation.worker_runtime: retry policy, GPU-fault classification, heartbeat files, pid liveness."""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from bio_automation import worker_runtime as wr
from bio_automation.queue import Queue, QueueLeaseLost


class Clock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now


class TestRetryDelay:
    @pytest.mark.parametrize("attempts,expected", [(0, 60.0), (1, 60.0), (2, 120.0), (3, 240.0), (4, 480.0), (6, 1800.0), (30, 1800.0)])
    def test_doubles_per_attempt_and_is_capped(self, attempts, expected):
        assert wr.retry_delay_s(attempts) == expected


class TestGpuFault:
    @pytest.mark.parametrize("text", [
        "RuntimeError: a GPU render panicked",
        "RuntimeError: CUDA error 999: unknown error",
        "onnxruntime: CUDNN_STATUS_EXECUTION_FAILED",
        "DXGI_ERROR_DEVICE_REMOVED (887A0005)",
        "torch.cuda: CUDA failure 999 'unknown error'",
    ])
    def test_gpu_loss_texts_are_recognised(self, text):
        assert wr.is_gpu_fault(text) is True

    @pytest.mark.parametrize("text", [
        "OSError: [Errno 28] No space left on device",
        "FileNotFoundError: raw gone",
        "RuntimeError: 3 candidates have no render",
        "KeyError: 'cuda'",
    ])
    def test_ordinary_errors_are_not(self, text):
        assert wr.is_gpu_fault(text) is False


class TestSettle:
    def claim(self, tmp_path, attempts_before=0):
        clock = Clock()
        q = Queue(tmp_path / "q.sqlite3", clock=clock)
        mid = q.publish("t", {"day": "d"})
        for _ in range(attempts_before):
            q.claim(["t"], worker_id="w", lease_s=1.0)
            clock.now += 2.0
        m = q.claim(["t"], worker_id="w", lease_s=60.0)
        assert m.id == mid
        return q, m, clock

    def test_a_plain_failure_waits_the_backoff_for_its_attempt(self, tmp_path):
        q, m, clock = self.claim(tmp_path)
        s = wr.settle(q, m, OSError("disk full"), worker_id="w", max_attempts=4)
        assert s == wr.Settled("retry", 60.0, False, "OSError: disk full")
        assert q.claim(["t"], worker_id="w2") is None
        clock.now += 60.0
        assert q.claim(["t"], worker_id="w2").attempts == 2

    def test_a_gpu_fault_waits_at_least_the_gpu_delay(self, tmp_path):
        q, m, _ = self.claim(tmp_path)
        s = wr.settle(q, m, RuntimeError("a GPU render panicked"), worker_id="w", max_attempts=4)
        assert (s.outcome, s.delay_s, s.gpu_fault) == ("retry", 300.0, True)

    def test_the_last_allowed_attempt_dead_letters(self, tmp_path):
        q, m, _ = self.claim(tmp_path, attempts_before=2)
        assert m.attempts == 3
        assert wr.settle(q, m, ValueError("bad"), worker_id="w", max_attempts=3).outcome == "dead"
        assert q.dead_count("t") == 1

    def test_losing_the_claim_propagates(self, tmp_path):
        q, m, clock = self.claim(tmp_path)
        clock.now += 61.0
        q.claim(["t"], worker_id="w2", lease_s=60.0)
        with pytest.raises(QueueLeaseLost, match="claimed by 'w2', not by 'w'"):
            wr.settle(q, m, ValueError("late"), worker_id="w")


class TestHeartbeat:
    def test_beat_writes_the_exact_document_atomically(self, tmp_path):
        clock = Clock(500.0)
        hb = wr.Heartbeat(tmp_path / "hb", "nas-tag-7", "tag", clock=clock, pid=7)
        assert hb.beat("working", message_id=3, topic="nas.day_imported") is True
        assert json.loads((tmp_path / "hb" / "nas-tag-7.json").read_text(encoding="utf-8")) == {
            "worker_id": "nas-tag-7", "stage": "tag", "pid": 7, "state": "working", "message_id": 3, "topic": "nas.day_imported",
            "ts": 500.0, "started": 500.0}
        assert [p.name for p in (tmp_path / "hb").iterdir()] == ["nas-tag-7.json"]

    def test_beats_are_throttled_unless_forced(self, tmp_path):
        clock = Clock(0.0)
        hb = wr.Heartbeat(tmp_path, "nas-tag-7", "tag", clock=clock, pid=7)
        assert hb.beat("idle") is True
        clock.now = 5.0
        assert hb.beat("idle") is False
        assert hb.beat("working", force=True) is True and json.loads(hb.path.read_text(encoding="utf-8"))["state"] == "working"
        clock.now = 5.0 + wr.HEARTBEAT_EVERY_S
        assert hb.beat("idle") is True

    def test_an_unwritable_directory_is_not_an_error(self, tmp_path):
        blocker = tmp_path / "file"
        blocker.write_text("x", encoding="utf-8")
        assert wr.Heartbeat(blocker / "hb", "w", "tag").beat("idle") is False

    def test_read_heartbeats_returns_documents_and_reports_torn_files(self, tmp_path):
        wr.Heartbeat(tmp_path, "nas-a-1", "a", clock=Clock(1.0), pid=1).beat("idle")
        (tmp_path / "torn.json").write_text("{not json", encoding="utf-8")
        got = wr.read_heartbeats(tmp_path)
        assert got[0]["worker_id"] == "nas-a-1"
        assert got[1]["path"] == str(tmp_path / "torn.json") and got[1]["error"].startswith("JSONDecodeError")
        assert wr.read_heartbeats(tmp_path / "missing") == []

    def test_remove_deletes_the_file(self, tmp_path):
        hb = wr.Heartbeat(tmp_path, "w", "tag")
        hb.beat("idle")
        hb.remove()
        assert list(tmp_path.iterdir()) == []


class TestPidAlive:
    def test_the_current_process_is_alive(self):
        import os

        assert wr.pid_alive(os.getpid()) is True

    def test_a_finished_child_is_not_and_the_check_does_not_signal_anything(self):
        child = subprocess.Popen([sys.executable, "-c", "pass"])
        child.wait()
        assert wr.pid_alive(child.pid) is False

    def test_a_running_child_is_alive_and_still_running_after_the_check(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            assert wr.pid_alive(child.pid) is True
            assert child.poll() is None                     # os.kill(pid, 0) on Windows would have terminated it
        finally:
            child.kill()
            child.wait()

    @pytest.mark.parametrize("pid", [0, -5])
    def test_non_pids_are_not_alive(self, pid):
        assert wr.pid_alive(pid) is False


@pytest.mark.parametrize("worker_id,pid", [("nas-tag-123", 123), ("nas-render-1-4567", 4567), ("nas-lightcraft-9", 9), ("odd", None), ("nas-x-abc", None)])
def test_worker_pid_is_the_last_segment(worker_id, pid):
    assert wr.worker_pid(worker_id) == pid
