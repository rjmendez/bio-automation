"""A durable NETWORKED work queue: same contract as queue.Queue (publish/claim/ack/fail with lease-based redelivery), but backed by
Postgres instead of a local SQLite file, so any host on the network -- not just the one that owns the disk -- can publish and claim.

Why this exists: queue.Queue's docstring says "one SQLite file ... never a network share" on purpose -- SQLite's locking protocol needs a
real local filesystem lock, which network filesystems (SMB/NFS) do not reliably provide (silent lost updates or journal corruption, not
just slowness). That made the queue a single-host mechanism. This module gives every host its own ordinary network connection to one
Postgres server and lets Postgres's own row locking do the concurrency control: claim() uses `SELECT ... FOR UPDATE SKIP LOCKED`, so two
hosts claiming at the same moment never block each other and never double-claim -- the exact property a local file lock could not give
a second host.

Message identity, states and transitions are identical to queue.Queue (see its docstring): pending -> claimed (under a lease) -> done or
failed; a lease that is never ack()ed or fail()ed expires and the message is claimed again. `Message`, `QueueError` and `QueueLeaseLost`
are the same types queue.py defines, imported from there, so code that catches QueueLeaseLost does not care which backend raised it.
"""
from __future__ import annotations

import json
import threading
import time
from typing import Any, Callable, Iterable

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Json

from . import settings, topics
from .queue import Message, QueueError, QueueLeaseLost, _quantile

SCHEMA = """
CREATE TABLE IF NOT EXISTS pg_queue_messages (
    id BIGSERIAL PRIMARY KEY,
    topic TEXT NOT NULL,
    payload JSONB NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'claimed', 'done', 'failed')),
    created DOUBLE PRECISION NOT NULL,
    claimed_by TEXT,
    claimed_at DOUBLE PRECISION,
    lease_until DOUBLE PRECISION,
    attempts INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    finished_at DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS pg_queue_messages_claimable ON pg_queue_messages (status, lease_until, id);
CREATE INDEX IF NOT EXISTS pg_queue_messages_topic ON pg_queue_messages (topic, status, id);
CREATE INDEX IF NOT EXISTS pg_queue_messages_finished ON pg_queue_messages (topic, finished_at);

CREATE TABLE IF NOT EXISTS pg_queue_retired (
    id BIGINT PRIMARY KEY,
    topic TEXT NOT NULL,
    payload JSONB NOT NULL,
    created DOUBLE PRECISION NOT NULL,
    attempts INTEGER NOT NULL,
    error TEXT,
    finished_at DOUBLE PRECISION,
    superseded_by BIGINT NOT NULL,
    retired_at DOUBLE PRECISION NOT NULL
);
"""

DEFAULT_LEASE_S = 60.0
DEFAULT_DSN = settings.QUEUE_PG_DSN


class PgQueue:
    """One connection per instance; create one PgQueue per thread (psycopg connections are not safe for concurrent use from several
    threads at once) -- but any number of processes, on any number of hosts, may each hold their own PgQueue against the same DSN:
    correctness across them comes from Postgres's row locks, not from anything in this class."""

    def __init__(self, dsn: str = DEFAULT_DSN, clock: Callable[[], float] = time.time):
        self.dsn = dsn
        self._clock = clock
        self._lock = threading.RLock()
        self._db = psycopg.connect(dsn, autocommit=False, row_factory=dict_row)
        try:
            with self._db.cursor() as cur:
                cur.execute(SCHEMA)
            self._db.commit()
        except (psycopg.errors.DuplicateTable, psycopg.errors.UniqueViolation):
            # Same race as nas_io_postgres._conn(): two hosts' first-ever connect can both run CREATE TABLE IF NOT EXISTS at once, and
            # the implicit row-type catalog insert collides as a plain UniqueViolation before DuplicateTable would ever apply.
            self._db.rollback()
        except BaseException:
            self._db.close()
            raise

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def __enter__(self) -> "PgQueue":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- producer side
    def publish(self, topic: str, payload: dict) -> int:
        """Enqueue one message, pending immediately. Returns its id. Same payload contract as queue.Queue.publish."""
        topics.check_payload(topic, payload)
        with self._lock, self._db.cursor() as cur:
            cur.execute(
                "INSERT INTO pg_queue_messages (topic, payload, status, created, attempts) VALUES (%s, %s, 'pending', %s, 0) RETURNING id",
                (topic, Json(payload), self._clock()))
            mid = cur.fetchone()["id"]
            self._db.commit()
            return mid

    # -- consumer side
    def claim(self, topics: Iterable[str] | None = None, *, worker_id: str, lease_s: float = DEFAULT_LEASE_S) -> Message | None:
        """The oldest claimable message (pending, or claimed with an expired lease) in `topics` (None or empty = any topic), or None.
        Uses SELECT ... FOR UPDATE SKIP LOCKED: a second caller racing for the same row never blocks on it and never claims it twice --
        it just moves on to try the next-oldest claimable row instead."""
        now = self._clock()
        topic_list = list(topics) if topics else None   # empty list behaves like None, matching queue.Queue's claim()
        with self._lock, self._db.cursor() as cur:
            sql = ("SELECT id FROM pg_queue_messages WHERE (status='pending' OR (status='claimed' AND lease_until < %s))")
            args: list[Any] = [now]
            if topic_list:
                sql += " AND topic = ANY(%s)"
                args.append(topic_list)
            sql += " ORDER BY id LIMIT 1 FOR UPDATE SKIP LOCKED"
            cur.execute(sql, args)
            row = cur.fetchone()
            if row is None:
                self._db.commit()
                return None
            mid = row["id"]
            cur.execute(
                "UPDATE pg_queue_messages SET status='claimed', claimed_by=%s, claimed_at=%s, lease_until=%s, attempts=attempts+1 "
                "WHERE id=%s RETURNING id, topic, payload, attempts, created",
                (worker_id, now, now + lease_s, mid))
            m = cur.fetchone()
            self._db.commit()
        return Message(id=m["id"], topic=m["topic"], payload=m["payload"], attempts=m["attempts"], created=m["created"])

    def _finish(self, message_id: int, worker_id: str, set_sql: str, args: tuple) -> None:
        with self._lock, self._db.cursor() as cur:
            cur.execute(f"UPDATE {set_sql} WHERE id=%s AND status='claimed' AND claimed_by=%s", (*args, message_id, worker_id))
            if cur.rowcount == 1:
                self._db.commit()
                return
            cur.execute("SELECT status, claimed_by FROM pg_queue_messages WHERE id=%s", (message_id,))
            row = cur.fetchone()
            self._db.commit()
        if row is None:
            raise QueueError(f"message {message_id} does not exist")
        if row["status"] == "claimed":
            raise QueueLeaseLost(f"message {message_id} is claimed by {row['claimed_by']!r}, not by {worker_id!r}: its lease expired and "
                                 "another worker took it over, so this worker's result is not recorded")
        raise QueueLeaseLost(f"message {message_id} is not claimed (status {row['status']!r}, last claimed by {row['claimed_by']!r}): "
                             f"{worker_id!r} no longer holds it")

    def extend_lease(self, message_id: int, *, worker_id: str, lease_s: float = DEFAULT_LEASE_S) -> float:
        until = self._clock() + lease_s
        self._finish(message_id, worker_id, "pg_queue_messages SET lease_until=%s", (until,))
        return until

    def ack(self, message_id: int, *, worker_id: str) -> None:
        self._finish(message_id, worker_id, "pg_queue_messages SET status='done', finished_at=%s", (self._clock(),))

    def fail(self, message_id: int, error: str, *, worker_id: str) -> None:
        self._finish(message_id, worker_id, "pg_queue_messages SET status='failed', error=%s, finished_at=%s", (error, self._clock()))

    def requeue_failed(self, topic: str | None = None) -> int:
        with self._lock, self._db.cursor() as cur:
            if topic is None:
                cur.execute("UPDATE pg_queue_messages SET status='pending', error=NULL, finished_at=NULL WHERE status='failed'")
            else:
                cur.execute("UPDATE pg_queue_messages SET status='pending', error=NULL, finished_at=NULL WHERE status='failed' AND topic=%s",
                            (topic,))
            n = cur.rowcount
            self._db.commit()
            return n

    def retire_superseded(self) -> list[tuple[int, int]]:
        with self._lock, self._db.cursor() as cur:
            cur.execute(
                "SELECT f.id, MIN(d.id) AS by FROM pg_queue_messages f JOIN pg_queue_messages d ON d.topic = f.topic "
                "AND d.payload = f.payload AND d.id > f.id WHERE f.status = 'failed' AND d.status = 'done' GROUP BY f.id ORDER BY f.id")
            pairs = [(r["id"], r["by"]) for r in cur.fetchall()]
            now = self._clock()
            for failed_id, by in pairs:
                cur.execute(
                    "INSERT INTO pg_queue_retired (id, topic, payload, created, attempts, error, finished_at, superseded_by, retired_at) "
                    "SELECT id, topic, payload, created, attempts, error, finished_at, %s, %s FROM pg_queue_messages WHERE id = %s",
                    (by, now, failed_id))
                cur.execute("DELETE FROM pg_queue_messages WHERE id = %s", (failed_id,))
            self._db.commit()
        return pairs

    # -- introspection
    def counts(self, topic: str | None = None) -> dict[str, int]:
        with self._lock, self._db.cursor() as cur:
            if topic is None:
                cur.execute("SELECT status, COUNT(*) n FROM pg_queue_messages GROUP BY status")
            else:
                cur.execute("SELECT status, COUNT(*) n FROM pg_queue_messages WHERE topic=%s GROUP BY status", (topic,))
            rows = cur.fetchall()
            self._db.commit()
        return {r["status"]: r["n"] for r in rows}

    def timings(self, since: float | None = None) -> dict[str, dict[str, float]]:
        sql = ("SELECT topic, claimed_at - created AS wait, finished_at - claimed_at AS run FROM pg_queue_messages "
               "WHERE status='done' AND finished_at IS NOT NULL")
        args: list[Any] = []
        if since is not None:
            sql, args = sql + " AND finished_at >= %s", [since]
        with self._lock, self._db.cursor() as cur:
            cur.execute(sql, args)
            rows = cur.fetchall()
            self._db.commit()
        by_topic: dict[str, list] = {}
        for r in rows:
            by_topic.setdefault(r["topic"], []).append((r["wait"], r["run"]))
        out = {}
        for topic, pairs in sorted(by_topic.items()):
            waits, runs = sorted(w for w, _ in pairs), sorted(r for _, r in pairs)
            out[topic] = {"done": len(pairs), "wait_median_s": _quantile(waits, 0.5), "wait_p90_s": _quantile(waits, 0.9),
                          "run_median_s": _quantile(runs, 0.5), "run_p90_s": _quantile(runs, 0.9)}
        return out

    def open_payloads(self, topic: str) -> list[dict]:
        with self._lock, self._db.cursor() as cur:
            cur.execute("SELECT payload FROM pg_queue_messages WHERE topic=%s AND status IN ('pending', 'claimed') ORDER BY id", (topic,))
            rows = cur.fetchall()
            self._db.commit()
        return [r["payload"] for r in rows]

    def pending_payloads(self, topic: str) -> list[dict]:
        with self._lock, self._db.cursor() as cur:
            cur.execute("SELECT payload FROM pg_queue_messages WHERE topic=%s AND status='pending' ORDER BY id", (topic,))
            rows = cur.fetchall()
            self._db.commit()
        return [r["payload"] for r in rows]

    def peek_failed(self, topic: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock, self._db.cursor() as cur:
            if topic is None:
                cur.execute("SELECT * FROM pg_queue_messages WHERE status='failed' ORDER BY id DESC LIMIT %s", (limit,))
            else:
                cur.execute("SELECT * FROM pg_queue_messages WHERE status='failed' AND topic=%s ORDER BY id DESC LIMIT %s", (topic, limit))
            rows = cur.fetchall()
            self._db.commit()
        return [dict(r) for r in rows]
