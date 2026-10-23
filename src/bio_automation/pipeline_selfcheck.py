"""`bio pipeline-selfcheck [--json] [--deploy]`: one read-only pass over what the NAS pipeline workers depend on, in the shape of social-selfcheck.

Each check answers ok, warn or fail with a one-line message; the exit code is the worst level (0 ok, 1 warn, 2 fail) and the last output line is
`selfcheck: ok|warn|fail`. Nothing is changed: the queue database is opened read-only (no migration), the heartbeat and crash-loop files are only read.

  imports      the modules the workers import load with the new code (and, with --ops-dir, the ops repo's nas_pipeline.py imports in a fresh
               interpreter against this tree: the check a bad deploy fails, instead of a worker restart loop)
  queue        the database opens and its schema is one this code understands (a database newer than the code is what a rollback to older code meets)
  depth        messages per status, dead-lettered, retrying; failed or dead-lettered messages are a warning
  stuck        the oldest claimed message is within the longest lease a live worker renews; expired claims are being reaped
  workers      every heartbeat file: its pid is alive and it beat recently; a stage with waiting work and no live worker is a warning
  crash-loop   the supervisor's worker_guard.json: a worker that keeps exiting within seconds of starting
  gpu          nvidia-smi answers and lists a card (a display-driver reset or a lost card shows here before the next render dies)
  inat-pace    the machine-wide iNat pace file is not holding every request off for minutes (a 429 backoff)
  disk         free space on C: and D:

scripts/deploy_prod.ps1 runs it with --deploy after the social selfcheck and rolls back unless it exits 0 or 1 AND prints the summary line. In
--deploy mode only the checks that depend on the code being deployed (imports, queue schema) may fail; the rest depend on the machine's state
at that second (workers restart on new code only between messages) and are warnings at worst."""
from __future__ import annotations

import importlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, NamedTuple

from . import settings, topics, worker_runtime
from .queue import MIGRATIONS, Queue, QueueError

OK, WARN, FAIL = "ok", "warn", "fail"
EXIT = {OK: 0, WARN: 1, FAIL: 2}
CODE_DEPENDENT = ("imports", "queue")
WORKER_MODULES = ("queue", "topics", "inat", "inat_stock", "lightcraft_mcp", "raw_cache", "nas_io", "stock_select", "worker_runtime",
                  "ort_providers", "render", "settings")
LONGEST_LEASE_S = 6 * 3600.0                   # nas_pipeline.LEASE_MAX_S: a live worker renews its lease for at most this long
CLAIM_WARN_S = LONGEST_LEASE_S + 1800.0
PENDING_WARN_S = 24 * 3600.0                   # a claimable message this old and nobody took it
PACE_WARN_S = 300.0
GUARD_ALERT_AT = 3                             # worker_guard.GUARD_ALERT_AT
QUICK_RUN_S = 120.0                            # worker_guard.QUICK_RUN_S
STAGE_TOPICS = {                               # which worker stage consumes which topic (nas_pipeline.py's run_worker calls)
    "lightcraft": (topics.STOCK_RENDER_IN_LIBRARY, topics.NAS_DAY_MATCHED, topics.LIBRARY_DAY_SELECTED),
    "render": (topics.STOCK_RENDER_REQUESTED,),
    "tag": (topics.NAS_DAY_IMPORTED,),
    "publish": (topics.STOCK_RENDERED,),
}
GB = 1024 ** 3


class Check(NamedTuple):
    name: str
    level: str
    message: str


def worst(checks: list[Check]) -> str:
    return max((c.level for c in checks), key=lambda lv: EXIT[lv], default=OK)


def _age(seconds: float) -> str:
    return f"{seconds / 3600:.1f} h" if seconds >= 3600 else f"{seconds / 60:.1f} min" if seconds >= 120 else f"{seconds:.0f} s"


# ----------------------------------------------------------------------------------------------------------------------- checks

def check_imports(ops_dir: Path | str | None = None, *, python: str | None = None, timeout_s: float = 90.0,
                  runner: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> Check:
    failed = []
    for name in WORKER_MODULES:
        try:
            importlib.import_module(f"{__package__}.{name}")
        except Exception as e:  # noqa: BLE001 -- any failure to import is the finding
            failed.append(f"{name} ({type(e).__name__}: {e})")
    if failed:
        return Check("imports", FAIL, "workers would not start on this code: " + "; ".join(failed))
    if not ops_dir:
        return Check("imports", OK, f"{len(WORKER_MODULES)} worker modules import (ops repo not checked: no --ops-dir)")
    ops = Path(ops_dir)
    if not (ops / "nas_pipeline.py").is_file():
        return Check("imports", FAIL, f"{ops / 'nas_pipeline.py'} does not exist")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).resolve().parent.parent), str(ops)])
    try:
        done = runner([python or sys.executable, "-I", "-c", "import sys; sys.path[:0] = [p for p in sys.argv[1:]]; import nas_pipeline, pipeline_upkeep",
                       *env["PYTHONPATH"].split(os.pathsep)], capture_output=True, text=True, timeout=timeout_s, env=env, cwd=str(ops))
    except subprocess.TimeoutExpired:
        return Check("imports", FAIL, f"importing nas_pipeline from {ops} did not finish in {timeout_s:g} s")
    if done.returncode != 0:
        tail = (done.stderr or done.stdout).strip().splitlines()[-1:] or ["(no output)"]
        return Check("imports", FAIL, f"nas_pipeline / pipeline_upkeep from {ops} do not import against this tree: {tail[0][:300]}")
    return Check("imports", OK, f"{len(WORKER_MODULES)} worker modules and the ops repo's nas_pipeline / pipeline_upkeep import")


def open_queue(db: Path | str, clock=time.time) -> tuple[Queue | None, Check]:
    try:
        q = Queue(db, clock=clock, readonly=True)
        have = q.schema_version()
    except (QueueError, sqlite3.Error) as e:
        return None, Check("queue", WARN, f"queue database {db} cannot be read: {e}")
    if have > len(MIGRATIONS):
        q.close()
        return None, Check("queue", FAIL, f"{db} is schema version {have}, newer than this code understands ({len(MIGRATIONS)}): this code cannot run the "
                           "workers on it. Roll the database back with `PRAGMA user_version=3` (the migration only added a column and a table) "
                           "or restore a backup from BIO_RESULTS_DIR/backups/queue")
    if have < len(MIGRATIONS):
        return q, Check("queue", OK, f"{db} is schema version {have}; the first worker to start on this code migrates it to {len(MIGRATIONS)}")
    return q, Check("queue", OK, f"{db} is schema version {have}")


def check_depth(q: Queue) -> Check:
    h = q.health()
    counts = ", ".join(f"{k}={v}" for k, v in sorted(h["counts"].items())) or "empty"
    text = f"{counts}; dead-lettered={h['dead']}, retrying={h['retrying']}"
    bad = h["counts"].get("failed", 0) + h["dead"]
    if bad:
        newest = ((q.peek_dead(limit=1) if h["dead"] else q.peek_failed(limit=1)) or [{}])[0]
        why = newest.get("reason") or newest.get("error") or ""
        return Check("depth", WARN, f"{text}; {bad} failed or dead-lettered (newest: {why[:200]}); `nas_pipeline.py requeue --reason ...` lists them")
    return Check("depth", OK, text)


def check_stuck(q: Queue) -> Check:
    h = q.health()
    oldest, pending = h["oldest_claimed_age_s"], h["oldest_pending_age_s"]
    if oldest is not None and oldest > CLAIM_WARN_S:
        return Check("stuck", WARN, f"a message has been claimed for {_age(oldest)}, longer than any live worker keeps a lease ({_age(LONGEST_LEASE_S)})")
    if h["expired_claims"]:
        return Check("stuck", WARN, f"{h['expired_claims']} claimed messages have an expired lease (their worker died); the upkeep reaper or the next claim returns them")
    if pending is not None and pending > PENDING_WARN_S:
        return Check("stuck", WARN, f"the oldest claimable message has waited {_age(pending)}")
    return Check("stuck", OK, "nothing claimed longer than a lease" + (f"; oldest claim {_age(oldest)}" if oldest is not None else ""))


def check_workers(q: Queue | None, heartbeat_dir: Path | str, *, now: float, alive: Callable[[int], bool] = worker_runtime.pid_alive) -> Check:
    beats = worker_runtime.read_heartbeats(heartbeat_dir)
    if not beats:
        return Check("workers", WARN, f"no worker heartbeat in {heartbeat_dir}: no worker is running, or none has run on code that writes one yet")
    problems, live_stages = [], set()
    for b in beats:
        if "error" in b:
            problems.append(f"unreadable heartbeat {b['path']}: {b['error']}")
            continue
        age = now - float(b.get("ts", 0))
        if not alive(int(b.get("pid", 0))):
            problems.append(f"{b['worker_id']}: pid {b.get('pid')} is gone (last beat {_age(age)} ago)")
        elif age > worker_runtime.HEARTBEAT_STALE_S and b.get("state") != "working":
            problems.append(f"{b['worker_id']}: alive but silent for {_age(age)} (state {b.get('state')})")
        else:
            live_stages.add(b.get("stage"))
    if q is not None:
        for stage, names in STAGE_TOPICS.items():
            waiting = sum(q.counts(t).get("pending", 0) for t in names)
            if waiting and stage not in live_stages:
                problems.append(f"{waiting} messages wait for stage {stage}, which has no live worker")
    if problems:
        return Check("workers", WARN, "; ".join(problems))
    return Check("workers", OK, f"{len(beats)} workers alive: " + ", ".join(sorted(str(b["worker_id"]) for b in beats)))


def check_crash_loop(directory: Path | str, *, now: float | None = None) -> Check:
    """worker_guard.json (written by the ops repo's worker_guard.py for the supervisor): a worker is crash-looping when GUARD_ALERT_AT quick exits
    in a row are on record and its latest start has already ended (counted) or is younger than the quick-run limit; one that has stayed up past
    the limit since is running again and its old count is history."""
    now = time.time() if now is None else now
    path = Path(directory) / "worker_guard.json"
    if not path.is_file():
        return Check("crash-loop", OK, "no crash-loop record")
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
        looping = {n: s for n, s in state.items() if int(s.get("quick_exits", 0)) >= GUARD_ALERT_AT
                   and (s.get("counted") or now - float(s.get("last_start", 0)) < QUICK_RUN_S)}
    except (OSError, ValueError, AttributeError, TypeError) as e:
        return Check("crash-loop", WARN, f"{path} is unreadable: {type(e).__name__}: {e}")
    if looping:
        return Check("crash-loop", FAIL, "workers exiting within seconds of starting, being held back by the supervisor: " +
                     ", ".join(f"{n} x{s['quick_exits']}" for n, s in sorted(looping.items())) + f" (see {Path(directory) / 'ALERT-*.txt'})")
    return Check("crash-loop", OK, "no worker is crash-looping")


def default_gpu_probe() -> subprocess.CompletedProcess:
    return subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader"],
                          capture_output=True, text=True, timeout=20)


def check_gpu(probe: Callable[[], subprocess.CompletedProcess] = default_gpu_probe) -> Check:
    try:
        done = probe()
    except FileNotFoundError:
        return Check("gpu", WARN, "nvidia-smi is not installed or not on PATH: the GPU cannot be checked")
    except subprocess.TimeoutExpired:
        return Check("gpu", FAIL, "nvidia-smi did not answer in 20 s: the driver is hung or resetting")
    lines = [ln.strip() for ln in (done.stdout or "").splitlines() if ln.strip()]
    if done.returncode != 0 or not lines:
        return Check("gpu", FAIL, f"nvidia-smi exit {done.returncode}: {((done.stderr or done.stdout) or 'no output').strip()[:200]}")
    return Check("gpu", OK, f"{len(lines)} GPU(s) reachable: " + " | ".join(lines))


def check_inat_pace(path: Path | str, *, now: float) -> Check:
    path = Path(path)
    if not path.is_file():
        return Check("inat-pace", OK, "no pace file yet (no process has called the iNat API)")
    try:
        con = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=5)
        try:
            row = con.execute("SELECT next_slot FROM pace WHERE k = 0").fetchone()
        finally:
            con.close()
    except sqlite3.Error as e:
        return Check("inat-pace", WARN, f"pace file {path} cannot be read: {e}")
    held = (row[0] - now) if row else 0.0
    if held > PACE_WARN_S:
        return Check("inat-pace", WARN, f"every iNat request is held off for another {_age(held)} (a 429 backoff, or a stale future slot in {path})")
    return Check("inat-pace", OK, "iNat pace file is current" + (f" (next slot in {held:.1f} s)" if held > 0 else ""))


def _free(path: Path) -> int:
    return shutil.disk_usage(path).free


def check_disks(drives: dict[str, Path], free: Callable[[Path], int] = _free) -> list[Check]:
    out = []
    for name, path in drives.items():
        try:
            gb = free(path) / GB
        except OSError as e:
            out.append(Check(f"disk-{name}", WARN, f"{path}: {e}"))
            continue
        floor = settings.MIN_FREE_GB if name == "C" else 2.0
        level = FAIL if gb < floor else WARN if gb < floor * 1.5 else OK
        out.append(Check(f"disk-{name}", level, f"{gb:.1f} GB free on {path} (the pipeline pauses under {floor:g})"))
    return out


def default_drives() -> dict[str, Path]:
    return {"C": Path(os.environ.get("SystemDrive", "C:") + "\\"), "D": settings.CACHE_DIR}


# ----------------------------------------------------------------------------------------------------------------------- the run

def run_checks(*, db: Path | str | None = None, pipeline_dir: Path | str | None = None, now: float | None = None, ops_dir: Path | str | None = None,
               gpu_probe: Callable | None = None, drives: dict[str, Path] | None = None, free: Callable[[Path], int] | None = None,
               alive: Callable[[int], bool] | None = None, rate_db: Path | str | None = None, import_runner: Callable | None = None) -> list[Check]:
    now = time.time() if now is None else now
    pipeline_dir = Path(pipeline_dir or settings.NAS_PIPELINE_DIR)
    checks = [check_imports(ops_dir, **({"runner": import_runner} if import_runner else {}))]
    q, queue_check = open_queue(db or settings.QUEUE_DB, clock=lambda: now)
    checks.append(queue_check)
    try:
        if q is not None:
            checks += [check_depth(q), check_stuck(q)]
        checks.append(check_workers(q, pipeline_dir / "heartbeats", now=now, **({"alive": alive} if alive else {})))
    finally:
        if q is not None:
            q.close()
    checks += [check_crash_loop(pipeline_dir / "logs", now=now), check_gpu(gpu_probe or default_gpu_probe), check_inat_pace(rate_db or settings.INAT_RATE_DB, now=now),
               *check_disks(drives or default_drives(), free or _free)]
    return checks


def for_deploy(checks: list[Check]) -> list[Check]:
    """--deploy: only the code-dependent checks may fail; every other failure depends on the machine's state at that second and is a warning."""
    return [Check(c.name, WARN, f"{c.message} [environmental: a warning in --deploy mode]") if c.name not in CODE_DEPENDENT and c.level == FAIL else c
            for c in checks]


def report(checks: list[Check]) -> dict:
    level = worst(checks)
    return {"level": level, "exit": EXIT[level], "checks": [{"name": c.name, "level": c.level, "message": c.message} for c in checks]}


def cmd_pipeline_selfcheck(args) -> int:
    """0 ok, 1 warn, 2 fail. A crash is a fail in the form the deploy script reads: `selfcheck: fail (<exception class>)`, exit 2."""
    deploy = bool(getattr(args, "deploy", False))
    try:
        checks = run_checks(ops_dir=getattr(args, "ops_dir", None) or os.environ.get("BIO_OPS_DIR"))
        if deploy:
            checks = for_deploy(checks)
        out = report(checks)
        if getattr(args, "json", False) and not deploy:
            print(json.dumps(out, indent=2, ensure_ascii=False))
            return out["exit"]
        for c in out["checks"]:
            print(f"[{c['level']:<4}] {c['name']}: {c['message']}")
        print(f"selfcheck: {out['level']}")
        return out["exit"]
    except Exception as e:  # noqa: BLE001
        print(f"selfcheck crashed: {type(e).__name__}: {e}"[:500], file=sys.stderr, flush=True)
        print(f"selfcheck: fail ({type(e).__name__})")
        return EXIT[FAIL]


def add_parsers(sub) -> None:
    p = sub.add_parser("pipeline-selfcheck", help="Check what the NAS pipeline workers depend on (queue, workers, GPU, iNat pace, disk); exit 0 ok, 1 warn, 2 fail. Read-only.")
    p.add_argument("--json", action="store_true", help="print the result as JSON")
    p.add_argument("--ops-dir", help="the ops repo checkout (nas_pipeline.py); also import-checks it against this tree (default: $BIO_OPS_DIR)")
    p.add_argument("--deploy", action="store_true", help="what scripts/deploy_prod.ps1 runs: only imports and the queue schema can fail; the last line is `selfcheck: ok|warn|fail`")
