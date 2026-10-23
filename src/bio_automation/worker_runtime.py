"""What a queue worker does around its handler, kept out of the stage scripts so it is one tested definition: how a failed message is retried
(exponential backoff, a bounded number of attempts, then the dead-letter table), which failures mean the GPU is gone (CUDA error 999 after a
display-driver reset (TDR), a render engine that panicked: the process is not trusted any more and exits so its supervisor starts a fresh
one), and the heartbeat file a worker keeps so a selfcheck can tell a live worker from a dead one.

Nothing here touches the queue except settle(); the heartbeat is one small JSON file per worker, replaced atomically."""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from . import settings

MAX_ATTEMPTS = settings.QUEUE_MAX_ATTEMPTS
RETRY_BASE_S = 60.0                  # wait before the second delivery; doubles each attempt
RETRY_CAP_S = 1800.0
GPU_RETRY_S = 300.0                  # after a GPU fault: long enough for a display-driver reset to finish and the card to come back
HEARTBEAT_EVERY_S = 10.0
HEARTBEAT_STALE_S = 120.0            # a heartbeat older than this means the worker is hung or gone (an idle worker beats every poll)

# Lower-case fragments of an exception text that mean the GPU or the render engine is in a state this process cannot recover from.
GPU_FAULT_MARKERS = (
    "cuda error 999", "cuda failure 999", "cudaerrorunknown", "cuda_error_unknown",       # unknown error: what a TDR leaves the CUDA context in
    "cuda error 700", "cuda_error_illegal_address", "device-side assert", "cuda_error_launch_failed",
    "device removed", "device_removed", "device lost", "dxgi_error_device",
    "gpu render panicked",                                                                # the LightCraft engine (Rust) panicked on a render
    "cudnn_status_execution_failed", "cudnn_status_internal_error",
)


def retry_delay_s(attempts: int) -> float:
    """Seconds to wait before delivery number attempts+1: RETRY_BASE_S doubling per attempt already made, at most RETRY_CAP_S."""
    return min(RETRY_CAP_S, RETRY_BASE_S * 2.0 ** max(0, attempts - 1))


def is_gpu_fault(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in GPU_FAULT_MARKERS)


@dataclass(frozen=True)
class Settled:
    outcome: str            # 'retry' | 'dead'
    delay_s: float
    gpu_fault: bool
    reason: str


def settle(q, message, exc: BaseException, *, worker_id: str, max_attempts: int | None = None) -> Settled:
    """Record a failed attempt of `message` (queue.Message just handled by `worker_id`) after `exc`: back to pending after a backoff, or to the
    dead-letter table when its attempts are used up. A GPU fault waits GPU_RETRY_S. QueueLeaseLost (the claim was lost) propagates."""
    max_attempts = MAX_ATTEMPTS if max_attempts is None else max_attempts
    reason = f"{type(exc).__name__}: {exc}"
    gpu = is_gpu_fault(reason)
    delay = max(GPU_RETRY_S, retry_delay_s(message.attempts)) if gpu else retry_delay_s(message.attempts)
    outcome = q.retry(message.id, reason, worker_id=worker_id, delay_s=delay, max_attempts=max_attempts)
    return Settled(outcome, delay, gpu, reason)


# ------------------------------------------------------------------------------------------------------------------- heartbeat

def pid_alive(pid: int) -> bool:
    """True when a process with this pid exists. Never signals the process (on Windows os.kill(pid, 0) would TERMINATE it)."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        handle = kernel32.OpenProcess(0x1000, False, pid)         # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.get_last_error() == 5                    # access denied: it exists, it is just not ours
        try:
            code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True
            return code.value == 259                               # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def worker_pid(worker_id: str) -> int | None:
    """The pid in a worker id of the form nas-<stage>[-<engine>]-<pid>, else None."""
    tail = worker_id.rsplit("-", 1)[-1]
    return int(tail) if tail.isdigit() else None


class Heartbeat:
    """One JSON file per worker in `directory`: {worker_id, stage, pid, state, message_id, topic, ts, started}. beat() rewrites it at most every
    HEARTBEAT_EVERY_S (force=True: now). A directory that cannot be written is not an error for the worker (the selfcheck will say so)."""

    def __init__(self, directory: Path | str, worker_id: str, stage: str, *, clock=time.time, pid: int | None = None):
        self.dir, self.worker_id, self.stage, self.clock = Path(directory), worker_id, stage, clock
        self.pid = os.getpid() if pid is None else pid
        self.started = clock()
        self._last: float | None = None
        self.path = self.dir / f"{worker_id}.json"

    def beat(self, state: str, *, message_id: int | None = None, topic: str | None = None, force: bool = False) -> bool:
        now = self.clock()
        if not force and self._last is not None and now - self._last < HEARTBEAT_EVERY_S:
            return False
        doc = {"worker_id": self.worker_id, "stage": self.stage, "pid": self.pid, "state": state, "message_id": message_id, "topic": topic,
               "ts": now, "started": self.started}
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(json.dumps(doc), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError:
            return False
        self._last = now
        return True

    def remove(self) -> None:
        try:
            self.path.unlink()
        except OSError:
            pass


def read_heartbeats(directory: Path | str) -> list[dict]:
    """Every readable heartbeat file in `directory`, oldest worker id first; a torn or foreign file is reported as {'path', 'error'}."""
    out = []
    d = Path(directory)
    if not d.is_dir():
        return out
    for p in sorted(d.glob("*.json")):
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(doc, dict) or "worker_id" not in doc:
                raise ValueError("not a heartbeat")
            out.append(doc)
        except (OSError, ValueError) as e:
            out.append({"path": str(p), "error": f"{type(e).__name__}: {e}"})
    return out
