"""One gate for the NAS reads of every pipeline process on this machine, so a high-value read is not queued behind a bulk one.

The NAS is a single slow link (Wi-Fi, a few MB/s). Tag copies a raw it is about to score, lightcraft imports (LightCraft reads each new raw),
render copies a raw the tag stage has not cached, and match reads previews: all at the same time, from several processes. `NasIoGate` lets
at most `slots` of them read at once and, among those waiting, serves the stage with the lowest `PRIO` number first (first come first served
within a stage), so a tag copy waits for one import chunk, not for a whole day's import.

State is one SQLite file shared by every process (the same pattern as inat.SharedRateLimiter: BEGIN IMMEDIATE, sleeping outside the
transaction, wall-clock time): `holders` are the reads in progress, `waiters` the reads queued. Nothing is ever killed:
 - a holder renews its row every hold_ttl_s/3 while its read runs, for at most hold_max_s (a hung read then gives its slot back);
   a process that dies frees its slot after hold_ttl_s;
 - a waiter refreshes its row on every poll; a waiter not seen for waiter_ttl_s is dead and blocks nobody.
The gate never fails a read: if its file cannot be used (sqlite3.Error) it says so once and lets the read go ungated.
"""
from __future__ import annotations

import contextlib
import logging
import os
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path

from . import settings

log = logging.getLogger("bio_automation.nas_io")

PRIO = {"tag": 0, "lightcraft": 1, "render": 2, "match": 3}      # the lower the number, the sooner it is served


def _prio(stage: str) -> int:
    try:
        return PRIO[stage]
    except KeyError:
        raise ValueError(f"unknown NAS stage {stage!r}: expected one of {sorted(PRIO)}") from None


def _wait_event(stop: threading.Event, seconds: float) -> bool:
    """True when `stop` was set (the read is over), False after `seconds`."""
    return stop.wait(seconds)


class NasIoGate:
    def __init__(self, path: Path | str, slots: int, *, hold_ttl_s: float = 120, hold_max_s: float = 1800, waiter_ttl_s: float = 5,
                 poll_s: float = 0.25, time_fn: Callable[[], float] = time.time, sleep_fn: Callable[[float], None] = time.sleep,
                 pid_fn: Callable[[], int] = os.getpid, renew_wait_fn: Callable[[threading.Event, float], bool] = _wait_event):
        if slots < 1:
            raise ValueError(f"slots must be >= 1, got {slots!r}")
        self.path = Path(path)
        self.slots = slots
        self.hold_ttl_s = hold_ttl_s
        self.hold_max_s = hold_max_s
        self.waiter_ttl_s = waiter_ttl_s
        self.poll_s = poll_s
        self._time_fn = time_fn
        self._sleep_fn = sleep_fn
        self._pid_fn = pid_fn
        self._renew_wait_fn = renew_wait_fn
        self._lock = threading.Lock()
        self._db: sqlite3.Connection | None = None
        self._warned = False

    # --- storage

    def _conn(self) -> sqlite3.Connection:
        if self._db is None:                    # opened on first use: importing this module never creates the file
            self.path.parent.mkdir(parents=True, exist_ok=True)
            db = sqlite3.connect(str(self.path), timeout=30, isolation_level=None, check_same_thread=False)
            db.execute("CREATE TABLE IF NOT EXISTS holders (token TEXT PRIMARY KEY, pid INTEGER NOT NULL, stage TEXT NOT NULL, "
                       "prio INTEGER NOT NULL, expires REAL NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS waiters (token TEXT PRIMARY KEY, prio INTEGER NOT NULL, since REAL NOT NULL, "
                       "seen REAL NOT NULL)")
            self._db = db
        return self._db

    def _write(self, step: Callable[[sqlite3.Connection, float], object]):
        """`step(db, now)` inside one write transaction (after the expired rows are purged); its return value is returned."""
        with self._lock:
            db = self._conn()
            db.execute("BEGIN IMMEDIATE")
            try:
                now = self._time_fn()
                db.execute("DELETE FROM holders WHERE expires < ?", (now,))
                db.execute("DELETE FROM waiters WHERE seen < ?", (now - self.waiter_ttl_s,))
                result = step(db, now)
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
        return result

    def _fail_open(self, e: sqlite3.Error) -> None:
        if not self._warned:
            self._warned = True
            log.warning("NAS I/O gate %s is unusable (%s: %s): reads go ungated", self.path, type(e).__name__, e)

    # --- one decision

    def _attempt(self, token: str, stage: str, prio: int, since: float) -> bool:
        """Take a slot if one is free and nobody in front of this reader is waiting; otherwise (re)register as a waiter. True = granted."""
        def step(db, now):
            held = db.execute("SELECT COUNT(*) FROM holders").fetchone()[0]
            ahead = db.execute("SELECT COUNT(*) FROM waiters WHERE token != ? AND (prio < ? OR (prio = ? AND (since < ? OR (since = ? AND token < ?))))",
                               (token, prio, prio, since, since, token)).fetchone()[0]
            if held < self.slots and ahead == 0:
                db.execute("DELETE FROM waiters WHERE token = ?", (token,))
                db.execute("INSERT INTO holders (token, pid, stage, prio, expires) VALUES (?, ?, ?, ?, ?)",
                           (token, self._pid_fn(), stage, prio, now + self.hold_ttl_s))
                return True
            db.execute("INSERT INTO waiters (token, prio, since, seen) VALUES (?, ?, ?, ?) ON CONFLICT (token) DO UPDATE SET seen = excluded.seen",
                       (token, prio, since, now))
            return False
        return self._write(step)

    def _drop(self, token: str) -> None:
        """Remove a reader's rows, best effort (the ttls would remove them anyway)."""
        def step(db, now):
            db.execute("DELETE FROM waiters WHERE token = ?", (token,))
            db.execute("DELETE FROM holders WHERE token = ?", (token,))
        with contextlib.suppress(sqlite3.Error):
            self._write(step)

    def _wait_for_slot(self, stage: str, prio: int, token: str) -> None:
        since = self._time_fn()
        granted = False
        try:
            while not self._attempt(token, stage, prio, since):
                self._sleep_fn(self.poll_s)
            granted = True
        finally:
            if not granted:                     # interrupted while queued: do not leave a waiter in front of everyone
                self._drop(token)

    # --- the holder's renewal

    def _renew_loop(self, token: str, stage: str, stop: threading.Event) -> None:
        """Renew the holder's row every hold_ttl_s/3 until the read is over, but for at most hold_max_s: a read that hangs (a LightCraft call
        that never answers, a stalled SMB read) then lets its slot expire instead of holding it from every other process for good."""
        started = self._time_fn()

        def step(db, now):
            """'renewed', 'retaken' (the row had expired and its slot was still free: the holder took it back) or 'lost' (another reader
            took the slot meanwhile; this read now runs beyond the gate)."""
            if db.execute("UPDATE holders SET expires = ? WHERE token = ?", (now + self.hold_ttl_s, token)).rowcount:
                return "renewed"
            if db.execute("SELECT COUNT(*) FROM holders").fetchone()[0] < self.slots:
                db.execute("INSERT INTO holders (token, pid, stage, prio, expires) VALUES (?, ?, ?, ?, ?)",
                           (token, self._pid_fn(), stage, _prio(stage), now + self.hold_ttl_s))
                return "retaken"
            return "lost"
        lost = False
        while not self._renew_wait_fn(stop, self.hold_ttl_s / 3):
            held = self._time_fn() - started
            if held >= self.hold_max_s:
                log.warning("NAS I/O gate: a %s read has held its slot for %.0f s (hold_max_s %.0f): no longer renewed, the slot is free "
                            "again within %.0f s", stage, held, self.hold_max_s, self.hold_ttl_s)
                return
            try:
                outcome = self._write(step)
            except sqlite3.Error:
                continue
            if outcome == "renewed":
                lost = False
            elif outcome == "retaken":
                log.warning("NAS I/O gate: a %s read's row expired before its renewal at %.0f s: the slot was free and is held again",
                            stage, self._time_fn())
            elif not lost:
                lost = True
                log.warning("NAS I/O gate: a %s read lost its slot (its row expired before the renewal at %.0f s) and another reader took it: "
                            "this read now runs beyond the gate", stage, self._time_fn())

    # --- the API

    @contextlib.contextmanager
    def acquire(self, stage: str) -> Iterator[float]:
        """Hold one of the slots while the body runs; yields the seconds spent waiting for it. Waits for a free slot and for every waiting
        reader of a lower PRIO number (and earlier ones of the same). An unusable gate file lets the body run ungated."""
        prio = _prio(stage)
        token = f"{self._pid_fn()}-{uuid.uuid4().hex}"
        t0 = self._time_fn()
        try:
            self._wait_for_slot(stage, prio, token)
        except sqlite3.Error as e:
            self._fail_open(e)
            yield self._time_fn() - t0
            return
        waited = self._time_fn() - t0
        stop = threading.Event()
        renewer = threading.Thread(target=self._renew_loop, args=(token, stage, stop), daemon=True, name="nas-io-renew")
        renewer.start()
        try:
            yield waited
        finally:
            stop.set()
            renewer.join()
            self._drop(token)

    def yield_turn(self, stage: str) -> float:
        """Take no slot; wait while any reader of a lower PRIO number is queued. Returns the seconds waited."""
        prio = _prio(stage)
        t0 = self._time_fn()

        def queued_ahead(db, now):
            return db.execute("SELECT COUNT(*) FROM waiters WHERE prio < ?", (prio,)).fetchone()[0]
        try:
            while self._write(queued_ahead):
                self._sleep_fn(self.poll_s)
        except sqlite3.Error as e:
            self._fail_open(e)
        return self._time_fn() - t0


class NullGate:
    """The gate when BIO_NAS_IO_GATE=0: nobody waits."""

    @contextlib.contextmanager
    def acquire(self, stage: str) -> Iterator[float]:
        _prio(stage)
        yield 0.0

    def yield_turn(self, stage: str) -> float:
        _prio(stage)
        return 0.0


_defaults: dict[tuple, NasIoGate] = {}
_defaults_lock = threading.Lock()


def gate(*, enabled: str | bool | None = None, path: Path | str | None = None, slots: int | None = None,
         hold_ttl_s: float | None = None) -> NasIoGate | NullGate:
    """The process's gate: built on first use from the settings (read now; an argument overrides its setting), one per distinct
    configuration. A NullGate when the gate is off ('0', 'false', 'no' or False)."""
    on = settings.NAS_IO_GATE if enabled is None else enabled
    if on is False or str(on).lower() in ("0", "false", "no"):
        return NullGate()
    key = (str(settings.NAS_IO_DB if path is None else path), settings.NAS_IO_SLOTS if slots is None else slots,
           settings.NAS_IO_HOLD_TTL_S if hold_ttl_s is None else hold_ttl_s, settings.NAS_IO_HOLD_MAX_S)
    with _defaults_lock:
        if key not in _defaults:
            _defaults[key] = NasIoGate(key[0], key[1], hold_ttl_s=key[2], hold_max_s=key[3])
        return _defaults[key]
