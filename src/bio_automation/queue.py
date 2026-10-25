"""A durable local work queue: one stage finishes and publishes an event instead of a person (or a script) running the next stage by hand.

One SQLite file (WAL mode, local disk only -- never a network share, same reasoning as store.py); several processes may publish and claim
concurrently. A message is claimed under a lease: if the worker that claimed it never acks or fails it (crash, kill -9), the lease expires and
another claim() picks it back up -- a stuck worker loses at most one lease period of latency, never the message. A message is `failed` only when
a worker calls fail() with a reason, and moves to the dead-letter table only when its attempts are used up (retry(), or claim()/reap_expired()
with max_attempts): a poison message that kills its worker on every delivery stops after max_attempts instead of being redelivered forever.
Nothing here invents a `failed` status on its own. This module is self-contained (its own schema/migrations) so it does not depend on the
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
    # bounded retries: not_before = a retried message is not claimable before then (backoff); dead_letter = messages that used up their
    # attempts (a poison message that keeps killing its worker, or a handler that keeps failing). A separate table, not a new status, because
    # the status CHECK of `messages` cannot be altered in place and workers on older code still read that table.
    """
    ALTER TABLE messages ADD COLUMN not_before REAL;
    CREATE TABLE IF NOT EXISTS dead_letter (
        id INTEGER PRIMARY KEY,
        topic TEXT NOT NULL,
        payload TEXT NOT NULL,
        created REAL NOT NULL,
        attempts INTEGER NOT NULL,
        error TEXT,
        reason TEXT NOT NULL,
        dead_at REAL NOT NULL
    );
    """,
    # dead_letter gets its own key. `messages` has no AUTOINCREMENT, so SQLite reuses the largest id after it is deleted (a dead-lettered day is
    # republished by backfill and gets the same id), and a second burial of that id failed on the old `id PRIMARY KEY` with IntegrityError, which
    # jammed every claim() on the topic. `seq` is the unique key now; `id` stays as the id the message had (informational, may repeat). Old code
    # keeps working on the rebuilt table (it inserts and reads by `id`), but needs `PRAGMA user_version=4` to open it.
    """
    DROP TABLE IF EXISTS dead_letter_v2;
    CREATE TABLE dead_letter_v2 (
        seq INTEGER PRIMARY KEY AUTOINCREMENT,
        id INTEGER NOT NULL,
        topic TEXT NOT NULL,
        payload TEXT NOT NULL,
        created REAL NOT NULL,
        attempts INTEGER NOT NULL,
        error TEXT,
        reason TEXT NOT NULL,
        dead_at REAL NOT NULL
    );
    INSERT INTO dead_letter_v2 (id, topic, payload, created, attempts, error, reason, dead_at)
        SELECT id, topic, payload, created, attempts, error, reason, dead_at FROM dead_letter ORDER BY dead_at, id;
    DROP TABLE dead_letter;
    ALTER TABLE dead_letter_v2 RENAME TO dead_letter;
    CREATE INDEX dead_letter_topic ON dead_letter (topic, seq);
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
    def __init__(self, path: Path | str = DEFAULT_DB, clock: Callable[[], float] = time.time, *, readonly: bool = False):
        """readonly: open an existing database for reading only -- no migration, no directory created, no pragma written (a selfcheck must not
        change the schema of the database the workers use). introspection works on any schema version; everything that writes raises."""
        self.path = Path(path)
        self._clock = clock
        self._lock = threading.RLock()
        self.readonly = readonly
        if readonly:
            if not self.path.is_file():
                raise QueueError(f"queue database {self.path} does not exist")
            self._db = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True, check_same_thread=False, isolation_level=None)
            self._db.row_factory = sqlite3.Row
            self._db.execute("PRAGMA busy_timeout=5000")
            return
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
                            try:
                                self._db.execute(statement)
                            except sqlite3.OperationalError as e:
                                # a database taken back to an older user_version (a rollback: `PRAGMA user_version=3`) still has the column
                                if "duplicate column name" not in str(e):
                                    raise
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
    def claim(self, topics: Iterable[str] | None = None, *, worker_id: str, lease_s: float = DEFAULT_LEASE_S,
              max_attempts: int | None = None) -> Message | None:
        """The oldest claimable message (pending, or claimed with an expired lease) in `topics` (None = any topic), or None when there isn't
        one. Claiming sets status='claimed', a fresh lease, and increments attempts -- the Nth claim is this message's Nth delivery, so a
        handler can tell a retry from a first attempt. A message waiting out a retry delay (retry()) is not claimable until then. With
        max_attempts, a claimable message already delivered that many times (its worker keeps dying: lease expired each time) is moved to the
        dead-letter table instead of being delivered again."""
        now = self._clock()
        topic_list = list(topics) if topics is not None else None
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                topic_sql, topic_args = "", []
                if topic_list:
                    topic_sql, topic_args = f" AND topic IN ({','.join('?' * len(topic_list))})", list(topic_list)
                if max_attempts is not None:
                    self._bury_exhausted(now, max_attempts, topic_sql, topic_args)
                sql = ("SELECT id FROM messages WHERE ((status='pending' AND (not_before IS NULL OR not_before <= ?)) "
                       "OR (status='claimed' AND lease_until < ?))" + topic_sql + " ORDER BY id LIMIT 1")
                row = self._db.execute(sql, [now, now, *topic_args]).fetchone()
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
        self._raise_not_holder(message_id, worker_id, row)

    @staticmethod
    def _raise_not_holder(message_id: int, worker_id: str, row) -> None:
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

    def _bury_exhausted(self, now: float, max_attempts: int, topic_sql: str, topic_args: list) -> list[int]:
        """Inside a transaction: dead-letter every claimable message delivered at least max_attempts times. Returns their ids."""
        rows = self._db.execute(
            "SELECT id FROM messages WHERE attempts >= ? AND ((status='pending' AND (not_before IS NULL OR not_before <= ?)) "
            "OR (status='claimed' AND lease_until < ?))" + topic_sql + " ORDER BY id", [max_attempts, now, now, *topic_args]).fetchall()
        for r in rows:
            self._bury(r["id"], f"delivered {max_attempts} times without being acked: the worker died or hung each time (lease expired)", now)
        return [r["id"] for r in rows]

    def _bury(self, message_id: int, reason: str, now: float) -> None:
        self._db.execute("INSERT INTO dead_letter (id, topic, payload, created, attempts, error, reason, dead_at) "
                         "SELECT id, topic, payload, created, attempts, error, ?, ? FROM messages WHERE id = ?", (reason, now, message_id))
        self._db.execute("DELETE FROM messages WHERE id = ?", (message_id,))

    def retry(self, message_id: int, error: str, *, worker_id: str, delay_s: float, max_attempts: int | None = None) -> str:
        """The claim holder gives the message back after a failed attempt: it goes back to pending, not claimable for delay_s (backoff), with
        `error` kept as the last error. When the message has already been delivered max_attempts times it goes to the dead-letter table
        instead. Returns 'retry' or 'dead'. QueueLeaseLost when the caller no longer holds the claim."""
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                row = self._db.execute("SELECT status, claimed_by, attempts FROM messages WHERE id=?", (message_id,)).fetchone()
                if row is None or row["status"] != "claimed" or row["claimed_by"] != worker_id:
                    self._db.execute("ROLLBACK")
                    self._raise_not_holder(message_id, worker_id, row)
                now = self._clock()
                if max_attempts is not None and row["attempts"] >= max_attempts:
                    self._db.execute("UPDATE messages SET error=? WHERE id=?", (error, message_id))
                    self._bury(message_id, f"{row['attempts']} attempts used up; last error: {error}", now)
                    outcome = "dead"
                else:
                    self._db.execute("UPDATE messages SET status='pending', error=?, lease_until=NULL, not_before=? WHERE id=?",
                                     (error, now + delay_s, message_id))
                    outcome = "retry"
                self._db.execute("COMMIT")
            except QueueError:
                raise
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
        return outcome

    def reap_expired(self, *, max_attempts: int, topic: str | None = None) -> list[tuple[int, str]]:
        """The stuck-job reaper: every claimed message whose lease has run out (its worker died or hung and stopped renewing) goes back to
        pending, or to the dead-letter table when it was already delivered max_attempts times. claim() would redeliver such a message anyway,
        but only once a worker polls that topic; this makes the state true (no 'running' job with a dead owner) and bounds a poison message even
        when no worker is polling. Returns [(id, 'requeued' | 'dead')]."""
        now = self._clock()
        out: list[tuple[int, str]] = []
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                sql, args = "SELECT id, attempts FROM messages WHERE status='claimed' AND lease_until < ?", [now]
                if topic is not None:
                    sql, args = sql + " AND topic=?", [*args, topic]
                for r in self._db.execute(sql + " ORDER BY id", args).fetchall():
                    if r["attempts"] >= max_attempts:
                        self._bury(r["id"], f"lease expired after delivery {r['attempts']} of {max_attempts}: the worker died or hung each time", now)
                        out.append((r["id"], "dead"))
                    else:
                        self._db.execute("UPDATE messages SET status='pending', lease_until=NULL, "
                                         "error=COALESCE(error, 'lease expired: the worker died or hung') WHERE id=?", (r["id"],))
                        out.append((r["id"], "requeued"))
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
        return out

    def claims(self) -> list[dict[str, Any]]:
        """Every claimed message: id, topic, claimed_by, claimed_at, lease_until, attempts -- what is 'running' right now, by whom."""
        with self._lock:
            rows = self._db.execute("SELECT id, topic, claimed_by, claimed_at, lease_until, attempts FROM messages WHERE status='claimed' ORDER BY id").fetchall()
        return [dict(r) for r in rows]

    def expire_claim(self, message_id: int, worker_id: str) -> bool:
        """End the lease of a claim held by `worker_id` now (its process is gone), so reap_expired()/claim() hand the message on without waiting out
        the rest of the lease. False when that worker no longer holds it."""
        with self._lock:
            return self._db.execute("UPDATE messages SET lease_until=? WHERE id=? AND status='claimed' AND claimed_by=?",
                                    (self._clock() - 1.0, message_id, worker_id)).rowcount == 1

    def requeue_dead(self, topic: str | None = None, *, reason_contains: str | None = None) -> int:
        """Move dead-lettered messages (in `topic`, whose reason or last error contains `reason_contains`, case-insensitive) back to pending
        with their attempts reset to 0, under NEW message ids (the old id may have been reused since). Returns how many."""
        sql, args = "SELECT seq FROM dead_letter WHERE 1=1", []
        if topic is not None:
            sql, args = sql + " AND topic=?", [*args, topic]
        if reason_contains:
            sql = sql + " AND (instr(lower(reason), lower(?)) > 0 OR instr(lower(COALESCE(error, '')), lower(?)) > 0)"
            args = [*args, reason_contains, reason_contains]
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                ids = [r[0] for r in self._db.execute(sql, args).fetchall()]
                for i in ids:                       # a fresh message id: the old one may belong to a live message by now (ids are reused)
                    self._db.execute("INSERT INTO messages (topic, payload, status, created, attempts) "
                                     "SELECT topic, payload, 'pending', created, 0 FROM dead_letter WHERE seq = ?", (i,))
                    self._db.execute("DELETE FROM dead_letter WHERE seq = ?", (i,))
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
        return len(ids)

    def failed_matching(self, topic: str | None = None, *, error_contains: str | None = None) -> list[dict[str, Any]]:
        """The `failed` messages requeue_failed() with the same arguments would move (a dry run), oldest first."""
        sql, args = "SELECT * FROM messages WHERE status='failed'", []
        if topic is not None:
            sql, args = sql + " AND topic=?", [*args, topic]
        if error_contains:
            sql, args = sql + " AND instr(lower(COALESCE(error, '')), lower(?)) > 0", [*args, error_contains]
        with self._lock:
            rows = self._db.execute(sql + " ORDER BY id", args).fetchall()
        return [{**dict(r), "payload": json.loads(r["payload"])} for r in rows]

    def requeue_failed(self, topic: str | None = None, *, error_contains: str | None = None, reset_attempts: bool = False) -> int:
        """Move every `failed` message (in `topic`, or all topics; whose error contains `error_contains`, case-insensitive, or any error) back
        to `pending` for another attempt. attempts is kept (the next delivery is attempt N+1) unless reset_attempts: a worker that passes
        max_attempts to claim() would otherwise bury a requeued message that had already used them. Returns how many."""
        sql, args = ("UPDATE messages SET status='pending', error=NULL, finished_at=NULL, not_before=NULL"
                     + (", attempts=0" if reset_attempts else "") + " WHERE status='failed'"), []
        if topic is not None:
            sql, args = sql + " AND topic=?", [*args, topic]
        if error_contains:
            sql, args = sql + " AND instr(lower(COALESCE(error, '')), lower(?)) > 0", [*args, error_contains]
        with self._lock:
            return self._db.execute(sql, args).rowcount

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

    def dead_count(self, topic: str | None = None) -> int:
        sql, args = "SELECT COUNT(*) FROM dead_letter", []
        if topic is not None:
            sql, args = sql + " WHERE topic=?", [topic]
        with self._lock:
            return self._db.execute(sql, args).fetchone()[0]

    def peek_dead(self, topic: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        sql, args = "SELECT * FROM dead_letter", []
        if topic is not None:
            sql, args = sql + " WHERE topic=?", [topic]
        with self._lock:
            rows = self._db.execute(sql + " ORDER BY seq DESC LIMIT ?", (*args, limit)).fetchall()
        return [{**dict(r), "payload": json.loads(r["payload"])} for r in rows]

    def dead_payloads(self, topic: str) -> list[dict]:
        """Payloads of every dead-lettered `topic` message, oldest first. A backfill must treat these days as taken: republishing one only feeds
        the same poison message to the workers again, an hour later; it comes back by an explicit requeue_dead once the cause is fixed."""
        with self._lock:
            if self.schema_version() < len(MIGRATIONS):
                return []                                               # a read-only handle on a database no worker has migrated yet
            rows = self._db.execute("SELECT payload FROM dead_letter WHERE topic=? ORDER BY seq", (topic,)).fetchall()
        return [json.loads(r["payload"]) for r in rows]

    def health(self) -> dict[str, Any]:
        """Queue-wide numbers for a selfcheck: counts by status, dead-lettered, messages waiting out a retry delay, claimed leases already
        expired, and the age in seconds of the oldest claimed message and of the oldest pending message that is claimable now (None when none)."""
        now = self._clock()
        with self._lock:
            migrated = self.schema_version() >= len(MIGRATIONS)          # a read-only handle may see a database no worker has migrated yet
            oldest_claimed = self._db.execute("SELECT MIN(claimed_at) FROM messages WHERE status='claimed'").fetchone()[0]
            oldest_pending = self._db.execute(
                "SELECT MIN(created) FROM messages WHERE status='pending'" + (" AND (not_before IS NULL OR not_before <= ?)" if migrated else ""),
                (now,) if migrated else ()).fetchone()[0]
            retrying = self._db.execute("SELECT COUNT(*) FROM messages WHERE status='pending' AND not_before > ?", (now,)).fetchone()[0] if migrated else 0
            expired = self._db.execute("SELECT COUNT(*) FROM messages WHERE status='claimed' AND lease_until < ?", (now,)).fetchone()[0]
            dead = self.dead_count() if migrated else 0
        return {"counts": self.counts(), "dead": dead, "retrying": retrying, "expired_claims": expired,
                "oldest_claimed_age_s": None if oldest_claimed is None else round(now - oldest_claimed, 1),
                "oldest_pending_age_s": None if oldest_pending is None else round(now - oldest_pending, 1)}

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
