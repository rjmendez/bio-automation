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


class TestTimings:
    def run_one(self, q, clock, topic, queued_s, run_s, *, fail=False):
        clock.now += 1.0
        mid = q.publish(topic, {})
        clock.now += queued_s
        q.claim([topic], worker_id="w")
        clock.now += run_s
        if fail:
            q.fail(mid, "boom", worker_id="w")
        else:
            q.ack(mid, worker_id="w")
        return mid

    def test_ack_and_fail_record_when_the_message_finished(self, tmp_path):
        clock = Clock()
        with Queue(tmp_path / "q.sqlite3", clock=clock) as q:
            done = self.run_one(q, clock, "t", 5.0, 7.0)
            failed = self.run_one(q, clock, "t", 1.0, 2.0, fail=True)
            rows = {r["id"]: r["finished_at"] for r in q._db.execute("SELECT id, finished_at FROM messages")}
        assert rows == {done: 1013.0, failed: 1017.0}

    def test_timings_give_the_median_and_p90_wait_and_run_per_topic(self, tmp_path):
        clock = Clock()
        with Queue(tmp_path / "q.sqlite3", clock=clock) as q:
            for queued, run in ((10.0, 1.0), (20.0, 2.0), (30.0, 3.0)):
                self.run_one(q, clock, "a", queued, run)
            self.run_one(q, clock, "b", 4.0, 8.0)
            self.run_one(q, clock, "b", 99.0, 99.0, fail=True)          # failed: not part of the done timings
            assert q.timings() == {
                "a": {"done": 3, "wait_median_s": 20.0, "wait_p90_s": 28.0, "run_median_s": 2.0, "run_p90_s": 2.8},
                "b": {"done": 1, "wait_median_s": 4.0, "wait_p90_s": 4.0, "run_median_s": 8.0, "run_p90_s": 8.0}}

    def test_timings_since_only_count_messages_finished_from_then_on(self, tmp_path):
        clock = Clock()
        with Queue(tmp_path / "q.sqlite3", clock=clock) as q:
            self.run_one(q, clock, "a", 10.0, 1.0)
            cut = clock.now + 0.5
            self.run_one(q, clock, "a", 20.0, 2.0)
            assert q.timings(since=cut) == {"a": {"done": 1, "wait_median_s": 20.0, "wait_p90_s": 20.0, "run_median_s": 2.0, "run_p90_s": 2.0}}
            assert q.timings(since=clock.now + 1) == {}

    def test_requeue_failed_clears_the_finish_time(self, tmp_path):
        clock = Clock()
        with Queue(tmp_path / "q.sqlite3", clock=clock) as q:
            mid = self.run_one(q, clock, "t", 1.0, 1.0, fail=True)
            assert q.requeue_failed("t") == 1
            assert q._db.execute("SELECT finished_at FROM messages WHERE id=?", (mid,)).fetchone()[0] is None

    def test_a_version_1_queue_is_migrated_with_its_messages_kept(self, tmp_path):
        path = tmp_path / "old.sqlite3"
        conn = sqlite3.connect(str(path))
        conn.executescript(MIGRATIONS[0])
        conn.execute("INSERT INTO messages (topic, payload, status, created, claimed_at, attempts) VALUES ('t', '{}', 'done', 1.0, 2.0, 1)")
        conn.execute("PRAGMA user_version=1")
        conn.commit()
        conn.close()
        with Queue(path) as q:
            assert q.schema_version() == len(MIGRATIONS) == 4
            assert q.counts() == {"done": 1}
            assert q.timings() == {}                                   # finished before finish times were recorded: no made-up duration


class TestExtendLease:
    def test_a_renewed_lease_keeps_the_message_from_being_reclaimed(self, tmp_path):
        clock = Clock()
        with Queue(tmp_path / "q.sqlite3", clock=clock) as q:
            mid = q.publish("t", {})
            q.claim(["t"], worker_id="w1", lease_s=60.0)
            clock.now += 50.0
            assert q.extend_lease(mid, worker_id="w1", lease_s=60.0) == 1110.0
            clock.now += 50.0                                       # past the first lease, inside the renewed one
            assert q.claim(["t"], worker_id="w2") is None
            q.ack(mid, worker_id="w1")
            assert q.counts("t") == {"done": 1}

    def test_without_renewal_the_same_message_is_reclaimed(self, tmp_path):
        clock = Clock()
        with Queue(tmp_path / "q.sqlite3", clock=clock) as q:
            mid = q.publish("t", {})
            q.claim(["t"], worker_id="w1", lease_s=60.0)
            clock.now += 100.0
            assert q.claim(["t"], worker_id="w2").id == mid

    def test_a_worker_that_lost_its_claim_cannot_renew_it(self, tmp_path):
        clock = Clock()
        with Queue(tmp_path / "q.sqlite3", clock=clock) as q:
            mid = q.publish("t", {})
            q.claim(["t"], worker_id="w1", lease_s=60.0)
            clock.now += 100.0
            q.claim(["t"], worker_id="w2", lease_s=60.0)
            with pytest.raises(QueueLeaseLost, match=r"is claimed by 'w2', not by 'w1'"):
                q.extend_lease(mid, worker_id="w1")
            assert q._db.execute("SELECT lease_until FROM messages WHERE id=?", (mid,)).fetchone()[0] == 1160.0

    def test_a_finished_message_cannot_be_renewed(self, q):
        mid = q.publish("t", {})
        q.claim(["t"], worker_id="w1")
        q.ack(mid, worker_id="w1")
        with pytest.raises(QueueLeaseLost, match=r"is not claimed \(status 'done'"):
            q.extend_lease(mid, worker_id="w1")


class TestRetireSuperseded:
    def fail_then(self, q, payload, later_status, topic="t"):
        """A failed message, then a later one with the same payload left in later_status (None = no later message)."""
        first = q.publish(topic, payload)
        q.claim([topic], worker_id="w")
        q.fail(first, "boom", worker_id="w")
        later = None
        if later_status:
            later = q.publish(topic, payload)
            if later_status != "pending":
                q.claim([topic], worker_id="w")
                if later_status == "done":
                    q.ack(later, worker_id="w")
                elif later_status == "failed":
                    q.fail(later, "again", worker_id="w")
        return first, later

    def test_a_failure_redone_successfully_moves_to_retired_with_its_successor(self, q):
        first, later = self.fail_then(q, {"day": "a"}, "done")
        assert q.retire_superseded() == [(first, later)]
        assert q.counts("t") == {"done": 1}
        row = q._db.execute("SELECT id, topic, payload, error, superseded_by, retired_at FROM retired").fetchone()
        assert tuple(row) == (first, "t", '{"day": "a"}', "boom", later, 1000.0)

    def test_a_failure_without_a_later_success_stays_failed(self, q):
        cases = (("a", None), ("b", "pending"), ("c", "claimed"), ("d", "failed"))
        kept = [self.fail_then(q, {"day": d}, status, topic=f"t{d}")[0] for d, status in cases]
        assert q.retire_superseded() == []
        assert q._db.execute("SELECT COUNT(*) FROM retired").fetchone()[0] == 0
        assert [q.peek_failed(f"t{d}")[-1]["id"] for d, _ in cases] == kept

    def test_a_success_for_another_payload_or_topic_does_not_supersede(self, q):
        first, _ = self.fail_then(q, {"day": "a"}, None)
        for topic, payload in (("t", {"day": "b"}), ("u", {"day": "a"})):
            mid = q.publish(topic, payload)
            q.claim([topic], worker_id="w")
            q.ack(mid, worker_id="w")
        assert q.retire_superseded() == []
        assert q.counts("t")["failed"] == 1

    def test_an_earlier_success_does_not_supersede_a_later_failure(self, q):
        mid = q.publish("t", {"day": "a"})
        q.claim(["t"], worker_id="w")
        q.ack(mid, worker_id="w")
        self.fail_then(q, {"day": "a"}, None)
        assert q.retire_superseded() == []


class TestPendingPayloads:
    def test_only_unclaimed_messages_are_pending(self, q):
        q.publish("t", {"day": "a"})
        q.publish("t", {"day": "b"})
        q.publish("u", {"day": "c"})
        q.claim(["t"], worker_id="w")
        assert q.pending_payloads("t") == [{"day": "b"}]
        assert q.open_payloads("t") == [{"day": "a"}, {"day": "b"}]


class TestBoundedRetries:
    def claimed(self, q, payload=None, lease_s=60.0, worker="w1"):
        mid = q.publish("t", payload or {"day": "a"})
        assert q.claim(["t"], worker_id=worker, lease_s=lease_s).id == mid
        return mid

    def test_a_retried_message_is_not_claimable_until_its_delay_passes(self, tmp_path):
        clock = Clock()
        with Queue(tmp_path / "q.sqlite3", clock=clock) as q:
            mid = self.claimed(q)
            assert q.retry(mid, "boom", worker_id="w1", delay_s=120.0, max_attempts=3) == "retry"
            assert q.counts("t") == {"pending": 1}
            clock.now += 119.0
            assert q.claim(["t"], worker_id="w2") is None
            clock.now += 1.0
            again = q.claim(["t"], worker_id="w2")
            assert (again.id, again.attempts) == (mid, 2)
            assert q.peek_failed("t") == []

    def test_the_last_error_is_kept_on_a_retried_message(self, tmp_path):
        with Queue(tmp_path / "q.sqlite3", clock=Clock()) as q:
            mid = self.claimed(q)
            q.retry(mid, "OSError: disk full", worker_id="w1", delay_s=0.0)
            row = q._db.execute("SELECT status, error, lease_until FROM messages WHERE id=?", (mid,)).fetchone()
            assert tuple(row) == ("pending", "OSError: disk full", None)

    def test_the_attempt_that_uses_the_last_try_goes_to_the_dead_letter_table(self, tmp_path):
        with Queue(tmp_path / "q.sqlite3", clock=Clock()) as q:
            mid = self.claimed(q, {"day": "x"})
            assert q.retry(mid, "boom", worker_id="w1", delay_s=0.0, max_attempts=1) == "dead"
            assert q.counts("t") == {} and q.dead_count() == 1 and q.dead_count("other") == 0
            dead = q.peek_dead("t")[0]
            assert (dead["id"], dead["payload"], dead["attempts"], dead["error"], dead["reason"], dead["dead_at"]) == (
                mid, {"day": "x"}, 1, "boom", "1 attempts used up; last error: boom", 1000.0)

    def test_one_attempt_below_the_limit_is_still_retried(self, tmp_path):
        with Queue(tmp_path / "q.sqlite3", clock=Clock()) as q:
            mid = self.claimed(q)
            assert q.retry(mid, "boom", worker_id="w1", delay_s=0.0, max_attempts=2) == "retry"
            assert q.dead_count() == 0

    def test_only_the_claim_holder_may_retry(self, tmp_path):
        clock = Clock()
        with Queue(tmp_path / "q.sqlite3", clock=clock) as q:
            mid = self.claimed(q, lease_s=10.0)
            clock.now += 11.0
            q.claim(["t"], worker_id="w2", lease_s=60.0)
            with pytest.raises(QueueLeaseLost, match=r"claimed by 'w2', not by 'w1'"):
                q.retry(mid, "late", worker_id="w1", delay_s=0.0)
            assert q.counts("t") == {"claimed": 1}

    def test_a_poison_message_whose_worker_keeps_dying_is_dead_lettered_not_redelivered_forever(self, tmp_path):
        clock = Clock()
        with Queue(tmp_path / "q.sqlite3", clock=clock) as q:
            q.publish("t", {"day": "poison"})
            for n in (1, 2, 3):                                    # three deliveries, each worker dies without acking
                assert q.claim(["t"], worker_id=f"w{n}", lease_s=60.0, max_attempts=3).attempts == n
                clock.now += 61.0
            assert q.claim(["t"], worker_id="w4", lease_s=60.0, max_attempts=3) is None
            assert q.counts("t") == {} and q.dead_count("t") == 1
            assert q.peek_dead("t")[0]["reason"] == "delivered 3 times without being acked: the worker died or hung each time (lease expired)"

    def test_without_max_attempts_redelivery_is_unbounded_as_before(self, tmp_path):
        clock = Clock()
        with Queue(tmp_path / "q.sqlite3", clock=clock) as q:
            q.publish("t", {})
            for n in range(1, 8):
                assert q.claim(["t"], worker_id="w", lease_s=1.0).attempts == n
                clock.now += 2.0
            assert q.dead_count() == 0

    def test_a_message_still_inside_its_lease_is_never_buried(self, tmp_path):
        with Queue(tmp_path / "q.sqlite3", clock=Clock()) as q:
            mid = self.claimed(q)
            q._db.execute("UPDATE messages SET attempts=9 WHERE id=?", (mid,))
            assert q.claim(["t"], worker_id="w2", max_attempts=3) is None
            assert q.counts("t") == {"claimed": 1} and q.dead_count() == 0


class TestReapExpired:
    def test_an_expired_claim_goes_back_to_pending_and_a_live_one_is_left(self, tmp_path):
        clock = Clock()
        with Queue(tmp_path / "q.sqlite3", clock=clock) as q:
            dead_owner = q.publish("t", {"day": "a"})
            q.claim(["t"], worker_id="w1", lease_s=60.0)
            q.publish("t", {"day": "b"})
            q.claim(["t"], worker_id="w2", lease_s=600.0)
            clock.now += 61.0
            assert q.reap_expired(max_attempts=3) == [(dead_owner, "requeued")]
            assert q.counts("t") == {"pending": 1, "claimed": 1}
            assert q.claim(["t"], worker_id="w3").id == dead_owner

    def test_an_expired_claim_that_used_all_its_attempts_is_dead_lettered(self, tmp_path):
        clock = Clock()
        with Queue(tmp_path / "q.sqlite3", clock=clock) as q:
            mid = q.publish("t", {})
            for delivery in (1, 2, 3):
                q.claim(["t"], worker_id="w", lease_s=10.0)
                clock.now += 11.0
                expected = "requeued" if delivery < 3 else "dead"
                assert q.reap_expired(max_attempts=3) == [(mid, expected)]
            assert q.counts("t") == {} and q.dead_count("t") == 1

    def test_the_topic_filter_limits_the_reaper(self, tmp_path):
        clock = Clock()
        with Queue(tmp_path / "q.sqlite3", clock=clock) as q:
            first = q.publish("a", {})
            q.publish("b", {})
            q.claim(["a"], worker_id="w", lease_s=1.0)
            q.claim(["b"], worker_id="w", lease_s=1.0)
            clock.now += 2.0
            assert q.reap_expired(max_attempts=3, topic="a") == [(first, "requeued")]
            assert q.counts() == {"pending": 1, "claimed": 1}


class TestRequeueByReason:
    def failed(self, q, topic, error, payload=None):
        mid = q.publish(topic, payload or {"n": error})
        q.claim([topic], worker_id="w")
        q.fail(mid, error, worker_id="w")
        return mid

    def test_only_failures_whose_error_contains_the_reason_are_requeued(self, q):
        panic = self.failed(q, "t", "RuntimeError: a GPU render panicked")
        self.failed(q, "t", "OSError: disk full")
        assert [m["id"] for m in q.failed_matching("t", error_contains="gpu render PANICKED")] == [panic]
        assert q.requeue_failed("t", error_contains="gpu render panicked") == 1
        assert q.counts("t") == {"pending": 1, "failed": 1}
        assert q.claim(["t"], worker_id="w2").id == panic

    def test_reset_attempts_zeroes_the_count_and_the_default_keeps_it(self, q):
        a = self.failed(q, "t", "boom")
        b = self.failed(q, "t", "boom")
        assert q.requeue_failed("t", reset_attempts=True) == 2
        assert [r[0] for r in q._db.execute("SELECT attempts FROM messages WHERE id IN (?, ?)", (a, b))] == [0, 0]
        self.failed(q, "u", "boom")
        q.requeue_failed("u")
        assert q._db.execute("SELECT attempts FROM messages WHERE topic='u'").fetchone()[0] == 1

    def test_dead_letters_are_requeued_under_their_old_id_with_attempts_reset(self, q):
        mid = q.publish("t", {"day": "z"})
        q.claim(["t"], worker_id="w")
        q.retry(mid, "CUDA error 999", worker_id="w", delay_s=0.0, max_attempts=1)
        assert q.requeue_dead("t", reason_contains="no such thing") == 0
        assert q.requeue_dead("t", reason_contains="cuda error 999") == 1
        assert q.dead_count() == 0
        m = q.claim(["t"], worker_id="w2")
        assert (m.id, m.payload, m.attempts) == (mid, {"day": "z"}, 1)


class TestHealth:
    def test_health_reports_ages_expired_claims_retrying_and_dead(self, tmp_path):
        clock = Clock()
        with Queue(tmp_path / "q.sqlite3", clock=clock) as q:
            assert q.health() == {"counts": {}, "dead": 0, "retrying": 0, "expired_claims": 0,
                                  "oldest_claimed_age_s": None, "oldest_pending_age_s": None}
            q.publish("t", {"n": 1})
            q.claim(["t"], worker_id="w", lease_s=30.0)                  # claimed at 1000, lease ends 1030
            clock.now += 100.0
            q.publish("t", {"n": 2})                                     # created at 1100, claimable now, but n=1 is older and its lease ran out
            waiting = q.publish("u", {"n": 3})
            q.claim(["u"], worker_id="w", lease_s=600.0)
            q.retry(waiting, "x", worker_id="w", delay_s=500.0)          # n=3 waits out a delay
            clock.now += 10.0
            assert q.health() == {"counts": {"claimed": 1, "pending": 2}, "dead": 0, "retrying": 1, "expired_claims": 1,
                                  "oldest_claimed_age_s": 110.0, "oldest_pending_age_s": 10.0}


class TestMigrationFromVersion3:
    def test_a_version_3_queue_keeps_its_messages_and_gains_the_new_columns(self, tmp_path):
        path = tmp_path / "v3.sqlite3"
        conn = sqlite3.connect(str(path))
        for m in MIGRATIONS[:3]:
            conn.executescript(m)
        conn.execute("INSERT INTO messages (topic, payload, status, created, attempts, error) VALUES ('t', '{}', 'failed', 1.0, 2, 'boom')")
        conn.execute("PRAGMA user_version=3")
        conn.commit()
        conn.close()
        with Queue(path) as q:
            assert q.schema_version() == 4
            assert q.counts() == {"failed": 1} and q.dead_count() == 0
            assert q.requeue_failed(error_contains="boom") == 1
            assert q.claim(["t"], worker_id="w", max_attempts=5).attempts == 3


class TestClaimsAndExpire:
    def test_claims_lists_who_holds_what(self, q):
        mid = q.publish("t", {})
        q.claim(["t"], worker_id="nas-tag-42", lease_s=30.0)
        q.publish("t", {})
        assert q.claims() == [{"id": mid, "topic": "t", "claimed_by": "nas-tag-42", "claimed_at": 1000.0, "lease_until": 1030.0, "attempts": 1}]

    def test_expire_claim_frees_the_message_for_the_reaper_but_only_for_its_holder(self, q):
        mid = q.publish("t", {})
        q.claim(["t"], worker_id="nas-tag-42", lease_s=3600.0)
        assert q.expire_claim(mid, "someone-else") is False
        assert q.reap_expired(max_attempts=3) == []
        assert q.expire_claim(mid, "nas-tag-42") is True
        assert q.reap_expired(max_attempts=3) == [(mid, "requeued")]
        assert q.claim(["t"], worker_id="w2").attempts == 2
