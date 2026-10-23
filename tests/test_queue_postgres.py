"""bio_automation.queue_postgres: the same queue contract as queue.Queue, over a real Postgres server instead of local SQLite.

Runs against a real Postgres by default (rule 10: no off-by-default tests) -- BIO_TEST_PG_DSN must point at a reachable server with
CREATE TABLE / DROP SCHEMA privilege; there is no mock and no skip. Each test gets its own schema (created in the fixture, dropped at
teardown) so tests never see one another's rows.
"""
from __future__ import annotations

import os
import threading

import psycopg
import pytest

from bio_automation.queue import Message, QueueError, QueueLeaseLost
from bio_automation.queue_postgres import PgQueue

BASE_DSN = os.environ.get("BIO_TEST_PG_DSN", "postgresql://bio_automation:bio_automation_dev@127.0.0.1:5432/bio_automation_dev")


class Clock:
    def __init__(self, now: float = 1000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


@pytest.fixture()
def dsn(request):
    """A DSN pointed at a fresh, empty schema named for this test; the schema (and everything in it) is dropped afterwards."""
    schema = "t_" + "".join(c if c.isalnum() else "_" for c in request.node.name)[:50]
    setup = psycopg.connect(BASE_DSN, autocommit=True)
    setup.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
    setup.execute(f'CREATE SCHEMA "{schema}"')
    setup.close()
    yield f"{BASE_DSN}?options=-csearch_path%3D{schema}"
    cleanup = psycopg.connect(BASE_DSN, autocommit=True)
    cleanup.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
    cleanup.close()


@pytest.fixture()
def q(dsn):
    with PgQueue(dsn, clock=Clock()) as queue:
        yield queue


class TestPublishAndClaim:
    def test_claim_returns_none_on_an_empty_queue(self, q):
        assert q.claim(worker_id="w1") is None

    def test_published_message_is_claimable_with_its_exact_fields(self, q):
        mid = q.publish("gallery.reindex", {"gallery_dir": "g", "folders": ["2026/2026-07"]})
        m = q.claim(worker_id="w1")
        assert m == Message(id=mid, topic="gallery.reindex", payload={"gallery_dir": "g", "folders": ["2026/2026-07"]}, attempts=1, created=1000.0)

    def test_claimed_message_is_not_claimed_again_while_its_lease_holds(self, q):
        q.publish("t", {})
        q.claim(worker_id="w1", lease_s=60)
        assert q.claim(worker_id="w2") is None

    def test_topic_filter_only_matches_named_topics(self, q):
        q.publish("a", {"n": 1})
        q.publish("b", {"n": 2})
        m = q.claim(topics=["b"], worker_id="w1")
        assert m.topic == "b" and m.payload == {"n": 2}
        assert q.claim(topics=["b"], worker_id="w1") is None           # 'a' is still there but not in the filter
        assert q.claim(topics=["a"], worker_id="w1").topic == "a"      # positive twin: it IS claimable under its own topic

    def test_messages_are_claimed_oldest_first(self, q):
        first = q.publish("t", {"n": 1})
        q.publish("t", {"n": 2})
        assert q.claim(worker_id="w1").id == first

    def test_attempts_counts_each_claim_as_a_delivery(self, dsn):
        clock = Clock(1000.0)
        with PgQueue(dsn, clock=clock) as q:
            q.publish("t", {})
            first = q.claim(worker_id="w1", lease_s=5)
            assert first.attempts == 1
            clock.now = 1006.0
            second = q.claim(worker_id="w2", lease_s=5)
            assert second.attempts == 2


class TestLeaseExpiryAndFinish:
    def test_an_expired_lease_is_reclaimable(self, dsn):
        clock = Clock(1000.0)
        with PgQueue(dsn, clock=clock) as q:
            q.publish("t", {})
            q.claim(worker_id="w1", lease_s=5)
            clock.now = 1006.0
            assert q.claim(worker_id="w2").attempts == 2

    def test_ack_by_the_claim_holder_marks_the_message_done(self, q):
        mid = q.publish("t", {})
        q.claim(worker_id="w1")
        q.ack(mid, worker_id="w1")
        assert q.counts() == {"done": 1}

    def test_ack_by_a_worker_that_never_held_the_claim_raises_lease_lost(self, q):
        mid = q.publish("t", {})
        q.claim(worker_id="w1")
        with pytest.raises(QueueLeaseLost, match=r"not by 'w2'"):
            q.ack(mid, worker_id="w2")

    def test_fail_by_the_claim_holder_marks_the_message_failed_with_its_reason(self, q):
        mid = q.publish("t", {})
        q.claim(worker_id="w1")
        q.fail(mid, "disk full", worker_id="w1")
        peeked = q.peek_failed()
        assert len(peeked) == 1 and peeked[0]["id"] == mid and peeked[0]["error"] == "disk full"

    def test_ack_of_a_message_that_does_not_exist_raises_queue_error(self, q):
        with pytest.raises(QueueError, match=r"message 999 does not exist"):
            q.ack(999, worker_id="w1")


class TestCrossHostConcurrency:
    def test_two_connections_claiming_at_once_never_double_claim(self, dsn):
        """The property a local SQLite file could not give a second host: two independent PgQueue connections (standing in for two
        hosts) racing to claim the same ten messages must land exactly ten claims between them, with no message claimed twice."""
        with PgQueue(dsn) as publisher:
            for i in range(10):
                publisher.publish("t", {"n": i})

        claimed_ids: list[int] = []
        lock = threading.Lock()

        def worker(worker_id: str):
            with PgQueue(dsn) as q:
                while True:
                    m = q.claim(worker_id=worker_id)
                    if m is None:
                        return
                    with lock:
                        claimed_ids.append(m.id)

        threads = [threading.Thread(target=worker, args=(f"w{i}",)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert sorted(claimed_ids) == list(range(1, 11))   # every message claimed exactly once, none missed, none doubled


class TestIntrospection:
    def test_counts_reflects_every_status_present(self, q):
        a = q.publish("t", {})
        q.claim(worker_id="w1")      # claims `a`: it is the oldest (only) pending message so far
        q.fail(a, "x", worker_id="w1")
        q.publish("t", {})          # published after the claim, so it stays pending
        assert q.counts() == {"pending": 1, "failed": 1}

    def test_counts_with_a_topic_filter_ignores_other_topics(self, q):
        q.publish("a", {})
        q.publish("b", {})
        assert q.counts(topic="a") == {"pending": 1}

    def test_pending_payloads_excludes_claimed_messages(self, q):
        q.publish("t", {"n": 1})
        q.publish("t", {"n": 2})
        q.claim(worker_id="w1")
        assert q.pending_payloads("t") == [{"n": 2}]

    def test_open_payloads_includes_claimed_messages(self, q):
        q.publish("t", {"n": 1})
        q.claim(worker_id="w1")
        assert q.open_payloads("t") == [{"n": 1}]
