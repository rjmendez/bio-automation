"""A durable local work queue: one stage finishes and publishes an event instead of a person (or a script) running the next stage by hand.

One SQLite file (WAL mode, local disk only -- never a network share, same reasoning as store.py); several processes may publish and claim
concurrently. A message is claimed under a lease: if the worker that claimed it never acks or fails it (crash, kill -9), the lease expires and
another claim() picks it back up -- a stuck worker loses at most one lease period of latency, never the message. A message is `failed` only when
a worker calls fail() with a reason; nothing here times a message out into `failed` on its own, so a failure is always something a worker chose
to report, never a status the queue invented. This module is self-contained (its own schema/migrations) so it does not depend on the
feat/state-foundation Store landing first; once it does, this queue's table can move into that same database file.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

MIGRATIONS = (
    """
    CREATE TABLE messages (
        id INTEGER PRIMARY KEY,
        topic TEXT NOT NULL,
        payload TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('pending', 'claimed', 'done', 'failed')),
        created REAL NOT NULL,
        claimed_by TEXT,
        claimed_at REAL,
        lease_until REAL,
        attempts INTEGER NOT NULL DEFAULT 0,
        error TEXT
    );
    CREATE INDEX messages_claimable ON messages (status, lease_until, id);
    CREATE INDEX messages_topic ON messages (topic, status, id);
    """,
)

DEFAULT_LEASE_S = 60.0
DEFAULT_DB = Path(os.environ.get("BIO_QUEUE_DB", os.path.join(os.environ.get("BIO_CACHE_DIR", r"D:\bio-cache"), "state", "queue.sqlite3")))


class QueueError(RuntimeError):
    """The queue cannot be used as asked (newer schema than this code, unknown message, wrong status transition)."""


@dataclass(frozen=True)
class Message:
    id: int
    topic: str
    payload: dict
    attempts: int
    created: float


class Queue:
    def __init__(self, path: Path | str = DEFAULT_DB, clock: Callable[[], float] = time.time):
        self.path = Path(path)
        self._clock = clock
        self._lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path), check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute("PRAGMA busy_timeout=5000")
        self._migrate()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def __enter__(self) -> "Queue":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def schema_version(self) -> int:
        with self._lock:
            return self._db.execute("PRAGMA user_version").fetchone()[0]

    def _migrate(self) -> None:
        with self._lock:
            have = self.schema_version()
            if have > len(MIGRATIONS):
                raise QueueError(f"{self.path} is schema version {have}, newer than this code understands ({len(MIGRATIONS)}); upgrade bio-automation")
            for version in range(have, len(MIGRATIONS)):
                self._db.execute("BEGIN IMMEDIATE")
                try:
                    for statement in MIGRATIONS[version].split(";"):
                        if statement.strip():
                            self._db.execute(statement)
                    self._db.execute(f"PRAGMA user_version={version + 1}")
                    self._db.execute("COMMIT")
                except BaseException:
                    self._db.execute("ROLLBACK")
                    raise

    # -- producer side
    def publish(self, topic: str, payload: dict) -> int:
        """Enqueue one message, pending immediately. Returns its id."""
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO messages (topic, payload, status, created, attempts) VALUES (?, ?, 'pending', ?, 0)",
                (topic, json.dumps(payload, sort_keys=True), self._clock()))
            return cur.lastrowid

    # -- consumer side
    def claim(self, topics: Iterable[str] | None = None, *, worker_id: str, lease_s: float = DEFAULT_LEASE_S) -> Message | None:
        """The oldest claimable message (pending, or claimed with an expired lease) in `topics` (None = any topic), or None when there isn't
        one. Claiming sets status='claimed', a fresh lease, and increments attempts -- the Nth claim is this message's Nth delivery, so a
        handler can tell a retry from a first attempt."""
        now = self._clock()
        topic_list = list(topics) if topics is not None else None
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                sql = "SELECT id FROM messages WHERE (status='pending' OR (status='claimed' AND lease_until < ?))"
                args: list[Any] = [now]
                if topic_list:
                    sql += f" AND topic IN ({','.join('?' * len(topic_list))})"
                    args += topic_list
                sql += " ORDER BY id LIMIT 1"
                row = self._db.execute(sql, args).fetchone()
                if row is None:
                    self._db.execute("COMMIT")
                    return None
                mid = row["id"]
                self._db.execute(
                    "UPDATE messages SET status='claimed', claimed_by=?, claimed_at=?, lease_until=?, attempts=attempts+1 WHERE id=?",
                    (worker_id, now, now + lease_s, mid))
                m = self._db.execute("SELECT * FROM messages WHERE id=?", (mid,)).fetchone()
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
        return Message(id=m["id"], topic=m["topic"], payload=json.loads(m["payload"]), attempts=m["attempts"], created=m["created"])

    def ack(self, message_id: int) -> None:
        with self._lock:
            cur = self._db.execute("UPDATE messages SET status='done' WHERE id=? AND status='claimed'", (message_id,))
            if cur.rowcount != 1:
                raise QueueError(f"message {message_id} does not exist or is not claimed")

    def fail(self, message_id: int, error: str) -> None:
        """Terminal: a worker decided this message cannot be processed. It stays in the table as status='failed' with its reason, visible
        to requeue_failed() or direct inspection -- never silently dropped."""
        with self._lock:
            cur = self._db.execute("UPDATE messages SET status='failed', error=? WHERE id=? AND status='claimed'", (error, message_id))
            if cur.rowcount != 1:
                raise QueueError(f"message {message_id} does not exist or is not claimed")

    def requeue_failed(self, topic: str | None = None) -> int:
        """Move every `failed` message (in `topic`, or all topics) back to `pending` for another attempt. Returns how many."""
        with self._lock:
            if topic is None:
                return self._db.execute("UPDATE messages SET status='pending', error=NULL WHERE status='failed'").rowcount
            return self._db.execute("UPDATE messages SET status='pending', error=NULL WHERE status='failed' AND topic=?", (topic,)).rowcount

    # -- introspection
    def counts(self, topic: str | None = None) -> dict[str, int]:
        sql, args = "SELECT status, COUNT(*) n FROM messages", []
        if topic is not None:
            sql, args = sql + " WHERE topic=?", [topic]
        with self._lock:
            rows = self._db.execute(sql + " GROUP BY status", args).fetchall()
        return {r["status"]: r["n"] for r in rows}

    def peek_failed(self, topic: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        sql, args = "SELECT * FROM messages WHERE status='failed'", []
        if topic is not None:
            sql, args = sql + " AND topic=?", [topic]
        with self._lock:
            rows = self._db.execute(sql + " ORDER BY id DESC LIMIT ?", (*args, limit)).fetchall()
        return [{**dict(r), "payload": json.loads(r["payload"])} for r in rows]
