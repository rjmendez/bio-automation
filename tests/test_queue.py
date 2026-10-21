"""bio_automation.queue: a durable local work queue (publish/claim/ack/fail with lease-based redelivery)."""
from __future__ import annotations

import sqlite3
import threading

import pytest

from bio_automation.queue import MIGRATIONS, Message, Queue, QueueError, QueueLeaseLost


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
        q.ack(m.id, worker_id="w1")
        assert q.claim(worker_id="w2") is None

    def test_acking_an_unclaimed_or_unknown_message_is_refused(self, q):
        mid = q.publish("t", {})
        with pytest.raises(QueueLeaseLost, match=rf"message {mid} is not claimed \(status 'pending', last claimed by None\): 'w1' no longer holds it"):
            q.ack(mid, worker_id="w1")                        # never claimed
        with pytest.raises(QueueError, match="message 999999 does not exist"):
            q.ack(999999, worker_id="w1")
        m = q.claim(worker_id="w1")                           # positive twin: the claim holder's ack goes through
        q.ack(m.id, worker_id="w1")
        assert q.counts() == {"done": 1}

    def test_failed_message_is_not_reclaimed_but_is_visible_with_its_reason(self, q):
        mid = q.publish("t", {"x": 1})
        m = q.claim(worker_id="w1")
        q.fail(m.id, "disk full", worker_id="w1")
        assert q.claim(worker_id="w2") is None
        [row] = q.peek_failed()
        assert (row["id"], row["error"], row["payload"]) == (mid, "disk full", {"x": 1})

    def test_requeue_failed_makes_it_claimable_again_and_clears_the_error(self, q):
        q.publish("t", {})
        m = q.claim(worker_id="w1")
        q.fail(m.id, "transient", worker_id="w1")
        assert q.requeue_failed() == 1
        again = q.claim(worker_id="w2")
        assert again is not None and again.attempts == 2
        q.ack(again.id, worker_id="w2")
        assert q.peek_failed() == []

    def test_requeue_failed_is_scoped_to_one_topic_when_given(self, q):
        q.publish("a", {})
        q.publish("b", {})
        q.fail(q.claim(topics=["a"], worker_id="w1").id, "e", worker_id="w1")
        q.fail(q.claim(topics=["b"], worker_id="w1").id, "e", worker_id="w1")
        assert q.requeue_failed(topic="a") == 1
        assert {r["topic"] for r in q.peek_failed()} == {"b"}


class TestCounts:
    def test_counts_reflect_every_status(self, q):
        a = q.publish("t", {})
        b = q.publish("t", {})
        c = q.publish("t", {})
        q.claim(worker_id="w1")                               # a -> claimed
        m = q.claim(worker_id="w1")
        q.ack(m.id, worker_id="w1")                           # b -> done
        m2 = q.claim(worker_id="w1")
        q.fail(m2.id, "e", worker_id="w1")                    # c -> failed
        assert q.counts() == {"claimed": 1, "done": 1, "failed": 1}

    def test_counts_can_be_scoped_to_one_topic(self, q):
        q.publish("a", {})
        q.publish("b", {})
        assert q.counts(topic="a") == {"pending": 1}
        assert q.counts(topic="b") == {"pending": 1}


class TestOpenPayloads:
    def test_pending_and_claimed_are_open_done_and_failed_are_not(self, q):
        for day in ("d1", "d2", "d3", "d4"):
            q.publish("t", {"day": day})
        q.publish("other", {"day": "x"})
        m1 = q.claim(["t"], worker_id="w")
        q.ack(m1.id, worker_id="w")                       # d1 done
        m2 = q.claim(["t"], worker_id="w")
        q.fail(m2.id, "boom", worker_id="w")              # d2 failed
        q.claim(["t"], worker_id="w")                     # d3 claimed
        assert q.open_payloads("t") == [{"day": "d3"}, {"day": "d4"}]

    def test_an_unknown_topic_has_none(self, q):
        q.publish("t", {"day": "d1"})
        assert q.open_payloads("nope") == []


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
                        wq.ack(m.id, worker_id=name)

            threads = [threading.Thread(target=worker, args=(f"w{i}",)) for i in range(6)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            assert sorted(got) == list(range(1, 21))          # every message delivered exactly once, none lost or doubled


class TestLeaseLost:
    """ack/fail only count from the worker that holds the claim: an expired lease that another worker reclaimed is not the old worker's."""

    def reclaimed(self, tmp_path):
        clock = Clock()
        q = Queue(tmp_path / "q.sqlite3", clock=clock)
        mid = q.publish("t", {"day": "d"})
        q.claim(worker_id="A", lease_s=10)
        clock.now += 11                                          # A overran its lease
        m = q.claim(worker_id="B", lease_s=10)
        assert (m.id, m.attempts) == (mid, 2)
        return q, mid

    def test_the_old_claimant_cannot_ack_a_reclaimed_message(self, tmp_path):
        q, mid = self.reclaimed(tmp_path)
        with pytest.raises(QueueLeaseLost, match=f"message {mid} is claimed by 'B', not by 'A': its lease expired"):
            q.ack(mid, worker_id="A")
        assert q.counts() == {"claimed": 1}                     # nothing changed: B still works on it
        q.ack(mid, worker_id="B")                               # positive twin: the holder's ack is recorded
        assert q.counts() == {"done": 1}

    def test_the_old_claimant_cannot_fail_a_reclaimed_message(self, tmp_path):
        q, mid = self.reclaimed(tmp_path)
        with pytest.raises(QueueLeaseLost, match=f"message {mid} is claimed by 'B', not by 'A'"):
            q.fail(mid, "slow", worker_id="A")
        assert q.peek_failed() == []
        q.fail(mid, "really broken", worker_id="B")             # positive twin
        assert [(r["id"], r["error"]) for r in q.peek_failed()] == [(mid, "really broken")]

    def test_a_second_ack_by_the_holder_after_done_is_refused(self, tmp_path):
        q, mid = self.reclaimed(tmp_path)
        q.ack(mid, worker_id="B")
        with pytest.raises(QueueLeaseLost, match=rf"message {mid} is not claimed \(status 'done', last claimed by 'B'\)"):
            q.fail(mid, "late", worker_id="B")
        assert q.counts() == {"done": 1}

    def test_an_expired_lease_nobody_reclaimed_is_still_the_holders(self, tmp_path):
        clock = Clock()
        q = Queue(tmp_path / "q.sqlite3", clock=clock)
        mid = q.publish("t", {})
        q.claim(worker_id="A", lease_s=10)
        clock.now += 11
        q.ack(mid, worker_id="A")
        assert q.counts() == {"done": 1}


class TestPayloadContract:
    """Queue.publish checks a known topic's payload keys against topics.PAYLOAD_KEYS -- the contract the producers in nas_pipeline.py send."""

    def test_the_producers_payload_shapes_are_accepted(self, q):
        from bio_automation import topics

        ids = [q.publish(topics.NAS_DAY_MATCHED, {"day": "2024-07-09"}),
               q.publish(topics.NAS_DAY_IMPORTED, {"day": "2024-07-09", "frames": "imported/2024-07-09.json"}),
               q.publish(topics.STOCK_RENDER_REQUESTED, {"day": "2024-07-09", "csv": "c.csv", "render_dir": "r"}),
               q.publish(topics.STOCK_RENDER_IN_LIBRARY, {"day": "2024-07-09", "csv": "c.csv", "render_dir": "r"}),
               q.publish(topics.STOCK_RENDERED, {"day": "2024-07-09", "csv": "c.csv", "render_dir": "r"}),
               q.publish(topics.GALLERY_REINDEX, {"gallery_dir": "g", "folders": ["2024/2024-07"]})]
        assert ids == [1, 2, 3, 4, 5, 6]
        assert q.counts() == {"pending": 6}

    def test_a_render_request_without_render_dir_is_refused(self, q):
        from bio_automation import topics

        with pytest.raises(topics.TopicPayloadError, match=r"stock.render_requested"):
            q.publish(topics.STOCK_RENDER_REQUESTED, {"day": "2024-07-09", "csv": "c.csv"})
        assert q.counts() == {}

    def test_the_documented_but_never_sent_ids_json_key_is_refused(self, q):
        from bio_automation import topics

        with pytest.raises(topics.TopicPayloadError,
                           match=r"nas\.day_imported payload must carry exactly \['day', 'frames'\]: missing \['frames'\], unexpected \['ids_json'\]"):
            q.publish(topics.NAS_DAY_IMPORTED, {"day": "2024-07-09", "ids_json": "[1, 2]"})
        assert q.counts() == {}

    def test_an_extra_key_is_refused(self, q):
        from bio_automation import topics

        with pytest.raises(topics.TopicPayloadError, match=r"missing \[\], unexpected \['extra'\]"):
            q.publish(topics.NAS_DAY_MATCHED, {"day": "d", "extra": 1})

    def test_a_topic_without_a_contract_is_not_checked(self, q):
        assert q.publish("scratch.topic", {"anything": 1}) == 1

    def test_every_topic_has_a_payload_contract(self):
        from bio_automation import topics

        assert set(topics.PAYLOAD_KEYS) == set(topics.ALL)
