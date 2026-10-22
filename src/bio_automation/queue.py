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

from . import settings, topics

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
    # when ack()/fail() closed the message: with created and claimed_at it gives each message's wait (created -> last claim) and run
    # (last claim -> finished), see timings(). Messages finished before this column existed have none.
    """
    ALTER TABLE messages ADD COLUMN finished_at REAL;
    CREATE INDEX messages_finished ON messages (topic, finished_at);
    """,
    # failed messages a later identical message has since completed (retire_superseded): kept here, out of the live counts
    """
    CREATE TABLE retired (
        id INTEGER PRIMARY KEY,
        topic TEXT NOT NULL,
        payload TEXT NOT NULL,
        created REAL NOT NULL,
        attempts INTEGER NOT NULL,
        error TEXT,
        finished_at REAL,
        superseded_by INTEGER NOT NULL,
        retired_at REAL NOT NULL
    );
    """,
)

DEFAULT_LEASE_S = 60.0


def _quantile(ordered: list[float], q: float) -> float:
    """The q-quantile of an ascending, non-empty list, interpolating linearly between neighbours (q=0.5 is the usual median)."""
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return round(ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo), 3)
DEFAULT_DB = settings.QUEUE_DB


class QueueError(RuntimeError):
    """The queue cannot be used as asked (newer schema than this code, unknown message, wrong status transition)."""


class QueueLeaseLost(QueueError):
    """ack()/fail() by a worker that no longer holds the message's claim (its lease expired and another worker reclaimed it, or it is
    already done/failed/pending)."""


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
        """Enqueue one message, pending immediately. Returns its id. A payload for a topic named in topics.PAYLOAD_KEYS must carry exactly
        that topic's keys (topics.TopicPayloadError otherwise, and nothing is enqueued)."""
        topics.check_payload(topic, payload)
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

    def _finish(self, message_id: int, worker_id: str, sql: str, args: tuple) -> None:
        """Run a done/failed transition only while `worker_id` still holds the claim; otherwise say exactly why not."""
        with self._lock:
            cur = self._db.execute(sql + " WHERE id=? AND status='claimed' AND claimed_by=?", (*args, message_id, worker_id))
            if cur.rowcount == 1:
                return
            row = self._db.execute("SELECT status, claimed_by FROM messages WHERE id=?", (message_id,)).fetchone()
        if row is None:
            raise QueueError(f"message {message_id} does not exist")
        if row["status"] == "claimed":
            raise QueueLeaseLost(f"message {message_id} is claimed by {row['claimed_by']!r}, not by {worker_id!r}: its lease expired and "
                                 "another worker took it over, so this worker's result is not recorded")
        raise QueueLeaseLost(f"message {message_id} is not claimed (status {row['status']!r}, last claimed by {row['claimed_by']!r}): "
                             f"{worker_id!r} no longer holds it")

    def extend_lease(self, message_id: int, *, worker_id: str, lease_s: float = DEFAULT_LEASE_S) -> float:
        """Renew the claim of a message still being worked on: its lease now ends lease_s from now (returned). Only the claim holder may
        renew (QueueLeaseLost otherwise, e.g. its lease already ran out and another worker reclaimed the message), so a worker that keeps
        renewing never has its message handed to a second worker, and one that dies stops renewing and its message comes back."""
        until = self._clock() + lease_s
        self._finish(message_id, worker_id, "UPDATE messages SET lease_until=?", (until,))
        return until

    def ack(self, message_id: int, *, worker_id: str) -> None:
        """Done -- only by the worker that holds the claim (claim()'s worker_id). A worker whose lease expired and whose message was
        reclaimed gets QueueLeaseLost and changes nothing: the new claimant's result is the one that counts."""
        self._finish(message_id, worker_id, "UPDATE messages SET status='done', finished_at=?", (self._clock(),))

    def fail(self, message_id: int, error: str, *, worker_id: str) -> None:
        """Terminal: a worker decided this message cannot be processed. It stays in the table as status='failed' with its reason, visible
        to requeue_failed() or direct inspection -- never silently dropped. Like ack(), only the claim holder may do this (QueueLeaseLost)."""
        self._finish(message_id, worker_id, "UPDATE messages SET status='failed', error=?, finished_at=?", (error, self._clock()))

    def requeue_failed(self, topic: str | None = None) -> int:
        """Move every `failed` message (in `topic`, or all topics) back to `pending` for another attempt. Returns how many."""
        with self._lock:
            if topic is None:
                return self._db.execute("UPDATE messages SET status='pending', error=NULL, finished_at=NULL WHERE status='failed'").rowcount
            return self._db.execute("UPDATE messages SET status='pending', error=NULL, finished_at=NULL WHERE status='failed' AND topic=?",
                                    (topic,)).rowcount

    def retire_superseded(self) -> list[tuple[int, int]]:
        """Move every `failed` message that a later message with the same topic and payload has since completed (status done) out of
        `messages` into `retired`, with the id of the message that superseded it. Returns [(failed id, superseding id)]. A failure with no
        later success stays failed: only a done message proves the work was redone."""
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                pairs = [(r[0], r[1]) for r in self._db.execute(
                    "SELECT f.id, MIN(d.id) FROM messages f JOIN messages d ON d.topic = f.topic AND d.payload = f.payload AND d.id > f.id "
                    "WHERE f.status = 'failed' AND d.status = 'done' GROUP BY f.id ORDER BY f.id")]
                now = self._clock()
                for failed_id, by in pairs:
                    self._db.execute("INSERT INTO retired (id, topic, payload, created, attempts, error, finished_at, superseded_by, retired_at) "
                                     "SELECT id, topic, payload, created, attempts, error, finished_at, ?, ? FROM messages WHERE id = ?",
                                     (by, now, failed_id))
                    self._db.execute("DELETE FROM messages WHERE id = ?", (failed_id,))
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
        return pairs

    # -- introspection
    def counts(self, topic: str | None = None) -> dict[str, int]:
        sql, args = "SELECT status, COUNT(*) n FROM messages", []
        if topic is not None:
            sql, args = sql + " WHERE topic=?", [topic]
        with self._lock:
            rows = self._db.execute(sql + " GROUP BY status", args).fetchall()
        return {r["status"]: r["n"] for r in rows}

    def timings(self, since: float | None = None) -> dict[str, dict[str, float]]:
        """Per topic, over the messages acked (status done) at or after `since` (None = all that carry a finish time): how many (`done`), the
        median and 90th percentile of their wait (created -> last claim: time spent queued, including earlier failed attempts) and of their
        run (last claim -> ack: time the worker spent on it), in seconds. A topic with no such message is absent."""
        sql = "SELECT topic, claimed_at - created AS wait, finished_at - claimed_at AS run FROM messages WHERE status='done' AND finished_at IS NOT NULL"
        args: list[Any] = []
        if since is not None:
            sql, args = sql + " AND finished_at >= ?", [since]
        with self._lock:
            rows = self._db.execute(sql, args).fetchall()
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
        """Payloads of every `topic` message still waiting or in progress (pending or claimed), oldest first -- what a backfill must not
        publish a second time."""
        with self._lock:
            rows = self._db.execute("SELECT payload FROM messages WHERE topic=? AND status IN ('pending', 'claimed') ORDER BY id",
                                    (topic,)).fetchall()
        return [json.loads(r["payload"]) for r in rows]

    def pending_payloads(self, topic: str) -> list[dict]:
        """Payloads of every `topic` message not yet claimed, oldest first: work that will still read the latest inputs when it starts."""
        with self._lock:
            rows = self._db.execute("SELECT payload FROM messages WHERE topic=? AND status='pending' ORDER BY id", (topic,)).fetchall()
        return [json.loads(r["payload"]) for r in rows]

    def peek_failed(self, topic: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        sql, args = "SELECT * FROM messages WHERE status='failed'", []
        if topic is not None:
            sql, args = sql + " AND topic=?", [topic]
        with self._lock:
            rows = self._db.execute(sql + " ORDER BY id DESC LIMIT ?", (*args, limit)).fetchall()
        return [{**dict(r), "payload": json.loads(r["payload"])} for r in rows]
