"""The NAS read gate (see nas_io.py's docstring for why it exists), over Postgres instead of local SQLite -- so a read happening on a
different HOST (not just a different process on this one) still waits its turn behind the same slots and the same tag-before-lightcraft-
before-render-before-match priority. Everything about the gate's behavior (holders, waiters, PRIO, TTLs, fail-open) is identical to
NasIoGate; only the storage changed. PRIO, NullGate and the stage validation are imported from nas_io.py rather than duplicated.

Why a networked gate matters here specifically: the NAS is ONE slow link regardless of how many hosts are reading from it. A local-only
gate on each host would let host A and host B each think they have slots free and both hit the NAS at once -- the exact problem the gate
exists to prevent, just moved from "two processes on one host" to "two hosts". A SQLite file cannot do this (see queue_postgres.py's
docstring for why); this module can, because every host talks to the same Postgres server over the network instead of a local file.

Concurrency: PgNasIoGate takes a LOCK TABLE ... IN EXCLUSIVE MODE at the start of every write, exactly mirroring NasIoGate's BEGIN
IMMEDIATE (which takes SQLite's one write lock on the whole file) -- the count-then-insert in _attempt() must be atomic across every
host, not just within one transaction's statements, and a full-table lock is the simplest way to guarantee that without inventing a
second serialization mechanism. NAS_IO_SLOTS is small (2 by default) so this is not a throughput bottleneck.
"""
from __future__ import annotations

import contextlib
import logging
import os
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from typing import Optional

import psycopg

from . import settings
from .nas_io import PRIO, NullGate, _prio

log = logging.getLogger("bio_automation.nas_io_postgres")

SCHEMA = """
CREATE TABLE IF NOT EXISTS pg_nasio_holders (
    token TEXT PRIMARY KEY,
    pid INTEGER NOT NULL,
    stage TEXT NOT NULL,
    prio INTEGER NOT NULL,
    expires DOUBLE PRECISION NOT NULL
);
CREATE TABLE IF NOT EXISTS pg_nasio_waiters (
    token TEXT PRIMARY KEY,
    prio INTEGER NOT NULL,
    since DOUBLE PRECISION NOT NULL,
    seen DOUBLE PRECISION NOT NULL
);
"""


def _wait_event(stop: threading.Event, seconds: float) -> bool:
    return stop.wait(seconds)


class PgNasIoGate:
    def __init__(self, dsn: str, slots: int, *, hold_ttl_s: float = 120, hold_max_s: float = 1800, waiter_ttl_s: float = 5,
                 poll_s: float = 0.25, time_fn: Callable[[], float] = time.time, sleep_fn: Callable[[float], None] = time.sleep,
                 pid_fn: Callable[[], int] = os.getpid, renew_wait_fn: Callable[[threading.Event, float], bool] = _wait_event):
        if slots < 1:
            raise ValueError(f"slots must be >= 1, got {slots!r}")
        self.dsn = dsn
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
        self._db: Optional["psycopg.Connection"] = None
        self._warned = False

    # --- storage

    def _conn(self) -> "psycopg.Connection":
        if self._db is None:
            db = psycopg.connect(self.dsn, autocommit=False)
            try:
                with db.cursor() as cur:
                    cur.execute(SCHEMA)
                db.commit()
            except (psycopg.errors.DuplicateTable, psycopg.errors.UniqueViolation):
                # Two gates on different hosts can both run CREATE TABLE IF NOT EXISTS at the same instant: IF NOT EXISTS checks the
                # catalog before creating, which is not atomic against a concurrent session doing the same thing. CREATE TABLE also
                # implicitly creates a row type, and THAT catalog insert is what actually collides first -- a plain UniqueViolation
                # (pg_type_typname_nsp_index), not a DuplicateTable -- even though the net result (the table exists) is what both wanted.
                db.rollback()
            except BaseException:
                db.close()   # anything else: do not leave a half-failed connection lying around holding locks until GC gets to it
                raise
            self._db = db
        return self._db

    def _write(self, step: Callable[["psycopg.Connection", float], object]):
        with self._lock:
            db = self._conn()
            with db.cursor() as cur:
                cur.execute("LOCK TABLE pg_nasio_holders, pg_nasio_waiters IN EXCLUSIVE MODE")
                now = self._time_fn()
                cur.execute("DELETE FROM pg_nasio_holders WHERE expires < %s", (now,))
                cur.execute("DELETE FROM pg_nasio_waiters WHERE seen < %s", (now - self.waiter_ttl_s,))
                result = step(cur, now)
            db.commit()
        return result

    def _fail_open(self, e: "psycopg.Error") -> None:
        if not self._warned:
            self._warned = True
            log.warning("NAS I/O gate (postgres) is unusable (%s: %s): reads go ungated", type(e).__name__, e)

    # --- one decision

    def _attempt(self, token: str, stage: str, prio: int, since: float) -> bool:
        def step(cur, now):
            cur.execute("SELECT COUNT(*) FROM pg_nasio_holders")
            held = cur.fetchone()[0]
            cur.execute(
                "SELECT COUNT(*) FROM pg_nasio_waiters WHERE token != %s AND (prio < %s OR (prio = %s AND (since < %s OR (since = %s AND token < %s))))",
                (token, prio, prio, since, since, token))
            ahead = cur.fetchone()[0]
            if held < self.slots and ahead == 0:
                cur.execute("DELETE FROM pg_nasio_waiters WHERE token = %s", (token,))
                cur.execute("INSERT INTO pg_nasio_holders (token, pid, stage, prio, expires) VALUES (%s, %s, %s, %s, %s)",
                            (token, self._pid_fn(), stage, prio, now + self.hold_ttl_s))
                return True
            cur.execute(
                "INSERT INTO pg_nasio_waiters (token, prio, since, seen) VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (token) DO UPDATE SET seen = excluded.seen",
                (token, prio, since, now))
            return False
        return self._write(step)

    def _drop(self, token: str) -> None:
        def step(cur, now):
            cur.execute("DELETE FROM pg_nasio_waiters WHERE token = %s", (token,))
            cur.execute("DELETE FROM pg_nasio_holders WHERE token = %s", (token,))
        with contextlib.suppress(psycopg.Error):
            self._write(step)

    def _wait_for_slot(self, stage: str, prio: int, token: str) -> None:
        since = self._time_fn()
        granted = False
        try:
            while not self._attempt(token, stage, prio, since):
                self._sleep_fn(self.poll_s)
            granted = True
        finally:
            if not granted:
                self._drop(token)

    # --- the holder's renewal

    def _renew_loop(self, token: str, stage: str, stop: threading.Event) -> None:
        started = self._time_fn()

        def step(cur, now):
            cur.execute("UPDATE pg_nasio_holders SET expires = %s WHERE token = %s", (now + self.hold_ttl_s, token))
        while not self._renew_wait_fn(stop, self.hold_ttl_s / 3):
            held = self._time_fn() - started
            if held >= self.hold_max_s:
                log.warning("NAS I/O gate (postgres): a %s read has held its slot for %.0f s (hold_max_s %.0f): no longer renewed, the "
                            "slot is free again within %.0f s", stage, held, self.hold_max_s, self.hold_ttl_s)
                return
            with contextlib.suppress(psycopg.Error):
                self._write(step)

    # --- the API

    @contextlib.contextmanager
    def acquire(self, stage: str) -> Iterator[float]:
        prio = _prio(stage)
        token = f"{self._pid_fn()}-{uuid.uuid4().hex}"
        t0 = self._time_fn()
        try:
            self._wait_for_slot(stage, prio, token)
        except psycopg.Error as e:
            self._fail_open(e)
            yield self._time_fn() - t0
            return
        waited = self._time_fn() - t0
        stop = threading.Event()
        renewer = threading.Thread(target=self._renew_loop, args=(token, stage, stop), daemon=True, name="nas-io-pg-renew")
        renewer.start()
        try:
            yield waited
        finally:
            stop.set()
            renewer.join()
            self._drop(token)

    def yield_turn(self, stage: str) -> float:
        prio = _prio(stage)
        t0 = self._time_fn()

        def queued_ahead(cur, now):
            cur.execute("SELECT COUNT(*) FROM pg_nasio_waiters WHERE prio < %s", (prio,))
            return cur.fetchone()[0]
        try:
            while self._write(queued_ahead):
                self._sleep_fn(self.poll_s)
        except psycopg.Error as e:
            self._fail_open(e)
        return self._time_fn() - t0


_defaults: dict[tuple, PgNasIoGate] = {}
_defaults_lock = threading.Lock()


def gate(*, enabled: "str | bool | None" = None, dsn: "str | None" = None, slots: "int | None" = None,
         hold_ttl_s: "float | None" = None) -> "PgNasIoGate | NullGate":
    """Same contract as nas_io.gate(): one gate per distinct (dsn, slots, hold_ttl_s) configuration, a NullGate when disabled."""
    on = settings.NAS_IO_GATE if enabled is None else enabled
    if on is False or str(on).lower() in ("0", "false", "no"):
        return NullGate()
    key = (str(settings.NAS_IO_PG_DSN if dsn is None else dsn), settings.NAS_IO_SLOTS if slots is None else slots,
           settings.NAS_IO_HOLD_TTL_S if hold_ttl_s is None else hold_ttl_s, settings.NAS_IO_HOLD_MAX_S)
    with _defaults_lock:
        if key not in _defaults:
            _defaults[key] = PgNasIoGate(key[0], key[1], hold_ttl_s=key[2], hold_max_s=key[3])
        return _defaults[key]
