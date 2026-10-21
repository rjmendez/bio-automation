"""bio_automation.queue: a durable local work queue (publish/claim/ack/fail with lease-based redelivery)."""
from __future__ import annotations

import sqlite3
import threading

import pytest

from bio_automation.queue import MIGRATIONS, Message, Queue, QueueError


class Clock:
    def __init__(self, now: float = 1000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


@pytest.fixture()
def q(tmp_path):
    with Queue(tmp_path / "q.sqlite3", clock=Clock()) as queue:
        yield queue


class TestSchema:
    def test_fresh_db_is_at_the_latest_schema_version(self, tmp_path):
        with Queue(tmp_path / "q.sqlite3") as q:
            assert q.schema_version() == len(MIGRATIONS)

    def test_a_schema_newer_than_this_code_understands_is_refused(self, tmp_path):
        path = tmp_path / "future.sqlite3"
        with Queue(path):
            pass
        conn = sqlite3.connect(str(path))
        conn.execute(f"PRAGMA user_version={len(MIGRATIONS) + 1}")
        conn.close()
        with pytest.raises(QueueError, match=r"is schema version \d+, newer than this code understands"):
            Queue(path)


class TestPublishAndClaim:
    def test_claim_returns_none_on_an_empty_queue(self, q):
        assert q.claim(worker_id="w1") is None

    def test_published_message_is_claimable_with_its_exact_fields(self, q):
        mid = q.publish("gallery.reindex", {"folders": ["2026/2026-07"]})
        m = q.claim(worker_id="w1")
        assert m == Message(id=mid, topic="gallery.reindex", payload={"folders": ["2026/2026-07"]}, attempts=1, created=1000.0)

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

    def test_attempts_counts_each_claim_as_a_delivery(self, tmp_path):
        clock = Clock(1000.0)
        with Queue(tmp_path / "q.sqlite3", clock=clock) as q:
            q.publish("t", {})
            first = q.claim(worker_id="w1", lease_s=5)
            assert first.attempts == 1
            clock.now = 1006.0                                # the lease expired without an ack: this is a redelivery
            second = q.claim(worker_id="w2", lease_s=5)
            assert second.attempts == 2


class TestLeaseExpiry:
    def test_an_unacked_message_is_reclaimed_once_its_lease_expires(self, tmp_path):
        clock = Clock(1000.0)
        with Queue(tmp_path / "q.sqlite3", clock=clock) as q:
            q.publish("t", {})
            q.claim(worker_id="crashed", lease_s=10)
            assert q.claim(worker_id="w2") is None          # lease still holds
            clock.now = 1011.0
            m = q.claim(worker_id="w2")
            assert m is not None and m.attempts == 2        # reclaimed: this is its second delivery

    def test_exactly_at_the_lease_boundary_is_not_yet_expired(self, tmp_path):
        clock = Clock(1000.0)
        with Queue(tmp_path / "q.sqlite3", clock=clock) as q:
            q.publish("t", {})
            q.claim(worker_id="w1", lease_s=10)
            clock.now = 1010.0                                # == lease_until, not >
            assert q.claim(worker_id="w2") is None
            clock.now = 1010.0001
            assert q.claim(worker_id="w2") is not None


class TestAckAndFail:
    def test_acked_message_is_never_claimed_again(self, q):
        q.publish("t", {})
        m = q.claim(worker_id="w1")
        q.ack(m.id)
        assert q.claim(worker_id="w2") is None

    def test_acking_an_unclaimed_or_unknown_message_is_refused(self, q):
        mid = q.publish("t", {})
        with pytest.raises(QueueError, match=f"message {mid} does not exist or is not claimed"):
            q.ack(mid)                                        # never claimed
        with pytest.raises(QueueError, match="message 999999 does not exist or is not claimed"):
            q.ack(999999)

    def test_failed_message_is_not_reclaimed_but_is_visible_with_its_reason(self, q):
        mid = q.publish("t", {"x": 1})
        m = q.claim(worker_id="w1")
        q.fail(m.id, "disk full")
        assert q.claim(worker_id="w2") is None
        [row] = q.peek_failed()
        assert (row["id"], row["error"], row["payload"]) == (mid, "disk full", {"x": 1})

    def test_requeue_failed_makes_it_claimable_again_and_clears_the_error(self, q):
        q.publish("t", {})
        m = q.claim(worker_id="w1")
        q.fail(m.id, "transient")
        assert q.requeue_failed() == 1
        again = q.claim(worker_id="w2")
        assert again is not None and again.attempts == 2
        q.ack(again.id)
        assert q.peek_failed() == []

    def test_requeue_failed_is_scoped_to_one_topic_when_given(self, q):
        q.publish("a", {})
        q.publish("b", {})
        q.fail(q.claim(topics=["a"], worker_id="w1").id, "e")
        q.fail(q.claim(topics=["b"], worker_id="w1").id, "e")
        assert q.requeue_failed(topic="a") == 1
        assert {r["topic"] for r in q.peek_failed()} == {"b"}


class TestCounts:
    def test_counts_reflect_every_status(self, q):
        a = q.publish("t", {})
        b = q.publish("t", {})
        c = q.publish("t", {})
        q.claim(worker_id="w1")                               # a -> claimed
        m = q.claim(worker_id="w1")
        q.ack(m.id)                                           # b -> done
        m2 = q.claim(worker_id="w1")
        q.fail(m2.id, "e")                                    # c -> failed
        assert q.counts() == {"claimed": 1, "done": 1, "failed": 1}

    def test_counts_can_be_scoped_to_one_topic(self, q):
        q.publish("a", {})
        q.publish("b", {})
        assert q.counts(topic="a") == {"pending": 1}
        assert q.counts(topic="b") == {"pending": 1}


class TestConcurrency:
    def test_two_threads_claiming_at_once_never_both_get_the_same_message(self, tmp_path):
        with Queue(tmp_path / "q.sqlite3") as q:
            for i in range(20):
                q.publish("t", {"n": i})
            got = []
            lock = threading.Lock()

            def worker(name):
                with Queue(tmp_path / "q.sqlite3") as wq:
                    while True:
                        m = wq.claim(worker_id=name)
                        if m is None:
                            return
                        with lock:
                            got.append(m.id)
                        wq.ack(m.id)

            threads = [threading.Thread(target=worker, args=(f"w{i}",)) for i in range(6)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            assert sorted(got) == list(range(1, 21))          # every message delivered exactly once, none lost or doubled
