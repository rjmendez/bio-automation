"""PgNasIoGate: the same contract as NasIoGate (see test_nas_io.py), over a real Postgres server instead of local SQLite, so a read on a
different HOST still waits its turn behind the same slots and priority.

Runs against a real Postgres by default (rule 10: no off-by-default tests) -- BIO_TEST_PG_DSN must point at a reachable server with
CREATE TABLE / DROP SCHEMA privilege; there is no mock and no skip. Each test gets its own schema.
"""
from __future__ import annotations

import contextlib
import os
import threading
import time

import psycopg
import pytest

from bio_automation.nas_io import PRIO
from bio_automation.nas_io_postgres import PgNasIoGate

BASE_DSN = os.environ.get("BIO_TEST_PG_DSN", "postgresql://bio_automation:bio_automation_dev@127.0.0.1:5432/bio_automation_dev")
HOLD_TTL = 30.0
POLL = 0.25


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


class Sleeper:
    def __init__(self, clock=None, hook=None, limit=200):
        self.calls, self.clock, self.hook, self.limit = [], clock, hook, limit

    def __call__(self, seconds):
        self.calls.append(seconds)
        if len(self.calls) > self.limit:
            pytest.fail(f"the gate never granted the slot after {self.limit} sleeps")
        if self.clock is not None:
            self.clock.advance(seconds)
        if self.hook is not None:
            self.hook(len(self.calls))


def no_renewal(stop, seconds):
    return stop.wait()


def wait_until(predicate, timeout=15.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            pytest.fail("condition not reached")
        time.sleep(0.01)


@pytest.fixture()
def dsn(request):
    schema = "t_" + "".join(c if c.isalnum() else "_" for c in request.node.name)[:50]
    setup = psycopg.connect(BASE_DSN, autocommit=True)
    setup.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
    setup.execute(f'CREATE SCHEMA "{schema}"')
    setup.close()
    yield f"{BASE_DSN}?options=-csearch_path%3D{schema}"
    cleanup = psycopg.connect(BASE_DSN, autocommit=True)
    cleanup.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
    cleanup.close()


def holders(g):
    with g._conn().cursor() as cur:
        cur.execute("SELECT stage, pid, prio FROM pg_nasio_holders ORDER BY stage")
        return cur.fetchall()


def waiters(g):
    with g._conn().cursor() as cur:
        cur.execute("SELECT token, prio FROM pg_nasio_waiters ORDER BY token")
        return cur.fetchall()


def make_gate(dsn, slots, clock, sleeper, **kw):
    kw.setdefault("hold_ttl_s", HOLD_TTL)
    return PgNasIoGate(dsn, slots, time_fn=clock, sleep_fn=sleeper, pid_fn=lambda: 4242, poll_s=POLL, renew_wait_fn=no_renewal, **kw)


def queue_waiter(g, token, stage, since):
    assert g._attempt(token, stage, PRIO[stage], since) is False, "the gate was not full: no waiter was queued"


@pytest.fixture()
def holds():
    stacks = []

    def hold(g, stage):
        stack = contextlib.ExitStack()
        stack.enter_context(g.acquire(stage))
        stacks.append(stack)
        return stack
    yield hold
    for s in stacks:
        s.close()


class TestSlots:
    def test_a_third_reader_is_granted_only_after_a_holder_releases(self, dsn, holds):
        clock = Clock()
        first_cell = []
        sleeper = Sleeper(clock, hook=lambda n: first_cell[0].close() if n == 1 else None)
        g = make_gate(dsn, 2, clock, sleeper)
        first_cell.append(holds(g, "tag"))
        holds(g, "render")
        with g.acquire("lightcraft") as waited:
            assert sleeper.calls == [POLL]
            assert waited == POLL
            assert holders(g) == [("lightcraft", 4242, 1), ("render", 4242, 2)]

    def test_with_three_slots_the_third_reader_is_granted_at_once(self, dsn, holds):
        clock = Clock()
        sleeper = Sleeper(clock)
        g = make_gate(dsn, 3, clock, sleeper)
        holds(g, "tag")
        holds(g, "render")
        with g.acquire("lightcraft") as waited:
            assert sleeper.calls == []
            assert waited == 0.0
            assert holders(g) == [("lightcraft", 4242, 1), ("render", 4242, 2), ("tag", 4242, 0)]

    def test_leaving_the_body_frees_the_slot_even_when_the_body_raises(self, dsn):
        clock = Clock()
        sleeper = Sleeper(clock)
        g = make_gate(dsn, 1, clock, sleeper)
        with pytest.raises(ZeroDivisionError, match="division by zero"):
            with g.acquire("tag"):
                1 / 0
        assert holders(g) == []
        with g.acquire("render"):
            assert sleeper.calls == []


class TestPriority:
    def test_a_queued_tag_reader_holds_a_render_reader_back_even_though_a_slot_is_free(self, dsn, holds):
        clock = Clock()
        sleeper = Sleeper(clock, hook=lambda n: g._drop("T"))
        g = make_gate(dsn, 1, clock, sleeper)
        h = holds(g, "lightcraft")
        queue_waiter(g, "T", "tag", clock())
        h.close()
        assert holders(g) == []
        with g.acquire("render"):
            assert sleeper.calls == [POLL]
            assert holders(g) == [("render", 4242, 2)]

    def test_within_one_priority_the_reader_that_queued_first_goes_first(self, dsn, holds):
        clock = Clock()
        sleeper = Sleeper(clock)
        g = make_gate(dsn, 1, clock, sleeper)
        h = holds(g, "lightcraft")
        queue_waiter(g, "older", "render", 1000.0)
        queue_waiter(g, "newer", "render", 1001.0)
        h.close()
        assert g._attempt("newer", "render", PRIO["render"], 1001.0) is False
        assert g._attempt("older", "render", PRIO["render"], 1000.0) is True
        assert holders(g) == [("render", 4242, 2)]

    def test_waiting_readers_are_served_by_priority_not_by_arrival(self, dsn):
        g = PgNasIoGate(dsn, 1, poll_s=0.01)
        order, release = [], {"tag": threading.Event(), "render": threading.Event()}

        def reader(stage):
            with PgNasIoGate(dsn, 1, poll_s=0.01).acquire(stage):
                order.append(stage)
                release[stage].wait(30)
        lightcraft = contextlib.ExitStack()
        lightcraft.enter_context(g.acquire("lightcraft"))
        render = threading.Thread(target=reader, args=("render",))
        tag = threading.Thread(target=reader, args=("tag",))
        render.start()
        wait_until(lambda: sorted(p for _, p in waiters(g)) == [PRIO["render"]])
        tag.start()
        wait_until(lambda: sorted(p for _, p in waiters(g)) == [PRIO["tag"], PRIO["render"]])
        lightcraft.close()
        wait_until(lambda: order)
        assert order == ["tag"]
        release["tag"].set()
        wait_until(lambda: len(order) == 2)
        release["render"].set()
        render.join(15)
        tag.join(15)
        assert order == ["tag", "render"]


class TestRenewalAndTtl:
    def test_a_holder_that_stops_renewing_frees_its_slot_after_the_ttl(self, dsn):
        clock = Clock()
        g = PgNasIoGate(dsn, 1, time_fn=clock, hold_ttl_s=HOLD_TTL)
        g._attempt("dead", "tag", PRIO["tag"], clock())
        assert holders(g) == [("tag", os.getpid(), 0)]
        clock.advance(HOLD_TTL + 1)
        assert g._attempt("new", "render", PRIO["render"], clock()) is True
        assert holders(g) == [("render", os.getpid(), 2)]


class TestCrossHostConcurrency:
    def test_two_connections_never_both_hold_a_single_slot(self, dsn):
        """The property a local SQLite file could not give a second host: two independent PgNasIoGate connections (standing in for two
        hosts) both trying to acquire the single slot must never both report themselves holding it at once."""
        both_held = threading.Event()
        results = []

        def worker(name, barrier):
            g = PgNasIoGate(dsn, 1, poll_s=0.01)
            barrier.wait()
            with g.acquire("tag"):
                with g._conn().cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM pg_nasio_holders")
                    results.append((name, cur.fetchone()[0]))
                time.sleep(0.05)

        barrier = threading.Barrier(2)
        threads = [threading.Thread(target=worker, args=(f"h{i}", barrier)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(15)

        assert len(results) == 2
        assert all(count == 1 for _, count in results), f"a holder saw more than 1 row while holding the single slot: {results}"
