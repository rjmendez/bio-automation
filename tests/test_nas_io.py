"""NasIoGate: at most `slots` NAS reads at once across processes, served by stage priority, never failing a read.

The gate is real (a SQLite file under tmp_path); only time, sleep and pid are injected. A Sleeper records every poll sleep and advances the
injected clock by it, so waits are exact numbers. A queued waiter is made by `queue_waiter`: one denied attempt, which is exactly what a
reader blocked in acquire() has done before it sleeps.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from bio_automation import nas_io, raw_cache, settings
from bio_automation.nas_io import PRIO, NasIoGate, NullGate
from bio_automation.raw_cache import RawCache

WORKER = Path(__file__).parent / "nas_io_worker.py"
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
    """Records each sleep, advances the clock by it, then runs `hook(n)` (n = 1 for the first sleep). A gate that never grants fails the
    test after `limit` sleeps instead of hanging it."""

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
    """The holder's renewal timer that never fires: blocks until the read is over."""
    return stop.wait()


def rows(db, sql):
    con = sqlite3.connect(str(db))
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


def holders(g):
    return rows(g.path, "SELECT stage, pid, prio FROM holders ORDER BY stage")


def waiters(g):
    return rows(g.path, "SELECT token, prio FROM waiters ORDER BY token")


def make_gate(tmp_path, slots, clock, sleeper, **kw):
    kw.setdefault("hold_ttl_s", HOLD_TTL)
    return NasIoGate(tmp_path / "gate.sqlite3", slots, time_fn=clock, sleep_fn=sleeper, pid_fn=lambda: 4242, poll_s=POLL,
                     renew_wait_fn=no_renewal, **kw)


@pytest.fixture()
def holds():
    """hold(gate, stage) takes a slot and keeps it until the returned stack is closed (or the test ends)."""
    stacks = []

    def hold(g, stage):
        stack = contextlib.ExitStack()
        stack.enter_context(g.acquire(stage))
        stacks.append(stack)
        return stack
    yield hold
    for s in stacks:
        s.close()


def queue_waiter(g, token, stage, since):
    assert g._attempt(token, stage, PRIO[stage], since) is False, "the gate was not full: no waiter was queued"


def wait_until(predicate, timeout=15.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            pytest.fail("condition not reached")
        time.sleep(0.01)


class TestSlots:
    def test_a_third_reader_is_granted_only_after_a_holder_releases(self, tmp_path, holds):
        clock = Clock()
        first_cell = []
        sleeper = Sleeper(clock, hook=lambda n: first_cell[0].close() if n == 1 else None)
        g = make_gate(tmp_path, 2, clock, sleeper)
        first_cell.append(holds(g, "tag"))
        holds(g, "render")
        with g.acquire("lightcraft") as waited:
            assert sleeper.calls == [POLL]                      # denied once (both slots taken), granted after the first holder let go
            assert waited == POLL
            assert holders(g) == [("lightcraft", 4242, 1), ("render", 4242, 2)]

    def test_with_three_slots_the_third_reader_is_granted_at_once(self, tmp_path, holds):
        clock = Clock()
        sleeper = Sleeper(clock)
        g = make_gate(tmp_path, 3, clock, sleeper)
        holds(g, "tag")
        holds(g, "render")
        with g.acquire("lightcraft") as waited:
            assert sleeper.calls == []
            assert waited == 0.0
            assert holders(g) == [("lightcraft", 4242, 1), ("render", 4242, 2), ("tag", 4242, 0)]

    def test_leaving_the_body_frees_the_slot_even_when_the_body_raises(self, tmp_path):
        clock = Clock()
        sleeper = Sleeper(clock)
        g = make_gate(tmp_path, 1, clock, sleeper)
        with pytest.raises(ZeroDivisionError, match="division by zero"):
            with g.acquire("tag"):
                1 / 0
        assert holders(g) == []
        with g.acquire("render"):
            assert sleeper.calls == []


class TestPriority:
    def test_a_queued_tag_reader_holds_a_render_reader_back_even_though_a_slot_is_free(self, tmp_path, holds):
        clock = Clock()
        sleeper = Sleeper(clock, hook=lambda n: g._drop("T"))             # the tag reader gets its turn elsewhere after the render reader's first poll
        g = make_gate(tmp_path, 1, clock, sleeper)
        h = holds(g, "lightcraft")
        queue_waiter(g, "T", "tag", clock())
        h.close()
        assert holders(g) == []                                           # the slot is free, the tag reader has not polled yet
        with g.acquire("render"):
            assert sleeper.calls == [POLL]
            assert holders(g) == [("render", 4242, 2)]

    def test_a_render_reader_alone_is_granted_at_once(self, tmp_path):
        clock = Clock()
        sleeper = Sleeper(clock)
        g = make_gate(tmp_path, 1, clock, sleeper)
        with g.acquire("render"):
            assert sleeper.calls == []
            assert holders(g) == [("render", 4242, 2)]

    def test_a_queued_match_reader_does_not_hold_a_render_reader_back(self, tmp_path, holds):
        clock = Clock()
        sleeper = Sleeper(clock)
        g = make_gate(tmp_path, 1, clock, sleeper)
        h = holds(g, "lightcraft")
        queue_waiter(g, "M", "match", clock())
        h.close()
        with g.acquire("render"):
            assert sleeper.calls == []

    def test_within_one_priority_the_reader_that_queued_first_goes_first(self, tmp_path, holds):
        clock = Clock()
        sleeper = Sleeper(clock)
        g = make_gate(tmp_path, 1, clock, sleeper)
        h = holds(g, "lightcraft")
        queue_waiter(g, "older", "render", 1000.0)
        queue_waiter(g, "newer", "render", 1001.0)
        h.close()
        assert g._attempt("newer", "render", PRIO["render"], 1001.0) is False
        assert g._attempt("older", "render", PRIO["render"], 1000.0) is True
        assert holders(g) == [("render", 4242, 2)]

    def test_waiting_readers_are_served_by_priority_not_by_arrival(self, tmp_path):
        g = NasIoGate(tmp_path / "gate.sqlite3", 1, poll_s=0.01)
        order, release = [], {"tag": threading.Event(), "render": threading.Event()}

        def reader(stage):
            with g.acquire(stage):
                order.append(stage)
                release[stage].wait(30)
        lightcraft = contextlib.ExitStack()
        lightcraft.enter_context(g.acquire("lightcraft"))
        render = threading.Thread(target=reader, args=("render",))
        tag = threading.Thread(target=reader, args=("tag",))
        render.start()
        wait_until(lambda: [p for _, p in waiters(g)] == [PRIO["render"]])
        tag.start()
        wait_until(lambda: sorted(p for _, p in waiters(g)) == [PRIO["tag"], PRIO["render"]])
        lightcraft.close()
        wait_until(lambda: order)
        assert order == ["tag"]                                           # the render reader queued first and is still waiting
        assert [(s, p) for s, _, p in holders(g)] == [("tag", PRIO["tag"])]
        release["tag"].set()
        wait_until(lambda: len(order) == 2)
        release["render"].set()
        render.join(15)
        tag.join(15)
        assert order == ["tag", "render"]


class Interrupted(Exception):
    pass


class TestInterruption:
    def test_a_reader_interrupted_while_queued_leaves_no_waiter_in_front_of_the_others(self, tmp_path, holds):
        clock = Clock()

        def interrupt(seconds):
            raise Interrupted("reader interrupted while queued")
        g = make_gate(tmp_path, 1, clock, interrupt)
        holds(g, "lightcraft")
        with pytest.raises(Interrupted, match="reader interrupted while queued"):
            with g.acquire("tag"):
                pytest.fail("the body must not run: the slot was never granted")
        assert waiters(g) == []
        assert holders(g) == [("lightcraft", 4242, 1)]

    def test_a_reader_that_is_granted_leaves_no_waiter_row_behind(self, tmp_path, holds):
        clock = Clock()
        first_cell = []
        sleeper = Sleeper(clock, hook=lambda n: first_cell[0].close())
        g = make_gate(tmp_path, 1, clock, sleeper)
        first_cell.append(holds(g, "lightcraft"))
        with g.acquire("tag"):
            assert sleeper.calls == [POLL]
            assert waiters(g) == []


class TestExpiry:
    def test_a_holder_whose_ttl_has_passed_is_purged_and_its_slot_granted(self, tmp_path, holds):
        clock = Clock()
        sleeper = Sleeper(clock)
        g = make_gate(tmp_path, 1, clock, sleeper)
        holds(g, "tag")
        clock.advance(HOLD_TTL + 1)                                       # the holder died: nobody renewed its row
        with g.acquire("render"):
            assert sleeper.calls == []
            assert holders(g) == [("render", 4242, 2)]

    def test_a_holder_within_its_ttl_still_blocks(self, tmp_path, holds):
        clock = Clock()
        first_cell = []
        sleeper = Sleeper(clock, hook=lambda n: first_cell[0].close() if n == 1 else None)
        g = make_gate(tmp_path, 1, clock, sleeper)
        first_cell.append(holds(g, "tag"))
        clock.advance(HOLD_TTL - 1)
        with g.acquire("render"):
            assert sleeper.calls == [POLL]

    def test_a_waiter_not_seen_for_longer_than_the_waiter_ttl_blocks_nobody(self, tmp_path, holds):
        clock = Clock()
        sleeper = Sleeper(clock)
        g = make_gate(tmp_path, 1, clock, sleeper, hold_ttl_s=1000, waiter_ttl_s=5)
        h = holds(g, "lightcraft")
        queue_waiter(g, "T", "tag", clock())
        h.close()
        clock.advance(6)
        with g.acquire("render"):
            assert sleeper.calls == []
            assert waiters(g) == []

    def test_a_fresh_waiter_blocks_until_it_has_gone_unseen_for_the_waiter_ttl(self, tmp_path, holds):
        clock = Clock()
        sleeper = Sleeper(clock)
        g = make_gate(tmp_path, 1, clock, sleeper, hold_ttl_s=1000, waiter_ttl_s=5)
        h = holds(g, "lightcraft")
        queue_waiter(g, "T", "tag", clock())
        h.close()
        clock.advance(4)                                                  # 1 s of its ttl left: seen < now - 5 first holds at now = +5.25
        with g.acquire("render") as waited:
            assert sleeper.calls == [POLL] * 5
            assert waited == 1.25

    def test_a_waiter_that_keeps_polling_keeps_its_place_past_the_waiter_ttl(self, tmp_path, holds):
        clock = Clock()
        sleeper = Sleeper(clock)
        g = make_gate(tmp_path, 1, clock, sleeper, hold_ttl_s=1000, waiter_ttl_s=5)
        h = holds(g, "lightcraft")
        queue_waiter(g, "T", "tag", clock())                              # queued at +0
        clock.advance(4)
        queue_waiter(g, "T", "tag", 1000.0)                               # its next poll at +4 (same since): seen is now +4
        h.close()
        clock.advance(4)                                                  # +8: 8 s since it queued, 4 s since it was last seen
        with g.acquire("render") as waited:                               # it blocks until seen < now - 5, i.e. now = +9.25
            assert sleeper.calls == [POLL] * 5
            assert waited == 1.25


class TestRenewal:
    def test_a_holder_renews_its_row_every_third_of_the_ttl_so_a_long_read_keeps_its_slot(self, tmp_path):
        clock = Clock(1000.0)
        intervals, reached = [], threading.Event()

        def renew_wait(stop, seconds):
            intervals.append(seconds)
            if len(intervals) >= 4:                                      # three renewals are done: hold here until the read is over
                reached.set()
                return stop.wait()
            clock.advance(seconds)
            return False
        closing = []
        sleeper = Sleeper(clock, hook=lambda n: closing[0].close())
        g = NasIoGate(tmp_path / "gate.sqlite3", 1, hold_ttl_s=HOLD_TTL, time_fn=clock, sleep_fn=sleeper, pid_fn=lambda: 4242, poll_s=POLL,
                      renew_wait_fn=renew_wait)
        holder = contextlib.ExitStack()
        holder.enter_context(g.acquire("tag"))
        closing.append(holder)
        assert reached.wait(15)
        assert intervals == [10.0] * 4
        assert rows(g.path, "SELECT expires FROM holders") == [(1060.0,)]    # last renewed at 1030 for another 30 s (taken at 1000: it would have expired at 1030)
        clock.advance(1)                                                  # 1031: past the row's original expiry
        with g.acquire("render"):
            assert sleeper.calls == [POLL]                                # still held, so the render reader waited for the release

    def test_a_holder_stops_renewing_after_hold_max_s_so_a_hung_read_gives_its_slot_back(self, tmp_path, caplog):
        clock = Clock(1000.0)
        intervals = []

        def renew_wait(stop, seconds):
            intervals.append(seconds)
            if len(intervals) > 3:                                       # the later tag holder's timer: never fires
                return stop.wait()
            clock.advance(seconds)
            return False
        sleeper = Sleeper(clock)
        g = NasIoGate(tmp_path / "gate.sqlite3", 1, hold_ttl_s=HOLD_TTL, hold_max_s=25, time_fn=clock, sleep_fn=sleeper, pid_fn=lambda: 4242,
                      poll_s=POLL, renew_wait_fn=renew_wait)
        with caplog.at_level(logging.WARNING, logger="bio_automation.nas_io"), g.acquire("lightcraft"):    # the read never finishes here
            wait_until(lambda: not any(t.name == "nas-io-renew" and t.is_alive() for t in threading.enumerate()))
            assert intervals == [10.0, 10.0, 10.0]                         # renewed at 1010 and 1020; at 1030 (30 s >= 25) it stops
            assert rows(g.path, "SELECT expires FROM holders") == [(1050.0,)]
            clock.t = 1051.0
            with g.acquire("tag"):                                       # the hung holder's row has expired: granted with no wait
                assert sleeper.calls == []
        assert [r.getMessage() for r in caplog.records] == [
            "NAS I/O gate: a lightcraft read has held its slot for 30 s (hold_max_s 25): no longer renewed, the slot is free again within 30 s"]

    def test_a_holder_within_hold_max_s_keeps_renewing(self, tmp_path, caplog):
        clock = Clock(1000.0)
        intervals, reached = [], threading.Event()

        def renew_wait(stop, seconds):
            intervals.append(seconds)
            if len(intervals) >= 4:
                reached.set()
                return stop.wait()
            clock.advance(seconds)
            return False
        g = NasIoGate(tmp_path / "gate.sqlite3", 1, hold_ttl_s=HOLD_TTL, hold_max_s=31, time_fn=clock, sleep_fn=Sleeper(clock), pid_fn=lambda: 4242,
                      poll_s=POLL, renew_wait_fn=renew_wait)
        with caplog.at_level(logging.WARNING, logger="bio_automation.nas_io"), g.acquire("lightcraft"):
            assert reached.wait(15)
            assert rows(g.path, "SELECT expires FROM holders") == [(1060.0,)]     # renewed at 1010, 1020 and 1030 (30 s < 31)
        assert caplog.records == []

    def test_the_default_timer_renews_in_real_time(self, tmp_path):
        g = NasIoGate(tmp_path / "gate.sqlite3", 1, hold_ttl_s=0.3)
        with g.acquire("tag"):
            first = rows(g.path, "SELECT expires FROM holders")[0][0]
            time.sleep(0.6)                                               # twice the ttl
            later = rows(g.path, "SELECT expires FROM holders")[0][0]
            assert later > first + 0.2
            assert g._attempt("X", "render", PRIO["render"], time.time()) is False       # still held
        g._drop("X")


class TestErrors:
    def test_zero_slots_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match=r"slots must be >= 1, got 0"):
            NasIoGate(tmp_path / "g.sqlite3", 0)

    def test_one_slot_is_accepted(self, tmp_path):
        assert NasIoGate(tmp_path / "g.sqlite3", 1).slots == 1

    def test_an_unknown_stage_is_refused_before_the_body_runs(self, tmp_path):
        g = NasIoGate(tmp_path / "g.sqlite3", 1)
        ran = []
        with pytest.raises(ValueError, match=r"unknown NAS stage 'foo'"):
            with g.acquire("foo"):
                ran.append("body")
        with pytest.raises(ValueError, match=r"unknown NAS stage 'foo'"):
            g.yield_turn("foo")
        assert ran == []
        assert not (tmp_path / "g.sqlite3").exists()                      # nothing was opened for it

    def test_a_known_stage_is_granted(self, tmp_path):
        g = NasIoGate(tmp_path / "g.sqlite3", 1)
        with g.acquire("tag") as waited:
            assert holders(g) == [("tag", os.getpid(), 0)]
            assert waited >= 0.0


class TestFailOpen:
    def test_a_gate_file_sqlite_cannot_open_lets_the_read_run_once_ungated_and_logs_once(self, tmp_path, caplog):
        broken = tmp_path / "a-directory"
        broken.mkdir()
        g = NasIoGate(broken, 1)
        ran = []
        with caplog.at_level(logging.WARNING, logger="bio_automation.nas_io"):
            with g.acquire("tag") as waited:
                ran.append("first")
            with g.acquire("render"):
                ran.append("second")
            g.yield_turn("match")
        assert ran == ["first", "second"]
        assert waited >= 0.0
        assert [r.getMessage().split(" (")[0] for r in caplog.records] == [f"NAS I/O gate {broken} is unusable"]

    def test_a_usable_gate_file_holds_a_row_during_the_body(self, tmp_path, caplog):
        g = NasIoGate(tmp_path / "g.sqlite3", 1)
        with caplog.at_level(logging.WARNING, logger="bio_automation.nas_io"):
            with g.acquire("tag"):
                assert holders(g) == [("tag", os.getpid(), 0)]
        assert caplog.records == []


class TestYieldTurn:
    def test_it_waits_while_a_reader_of_a_lower_number_is_queued_and_takes_no_slot(self, tmp_path, holds):
        clock = Clock()
        seen = []

        def hook(n):
            seen.append(holders(g))
            g._drop("T")
        sleeper = Sleeper(clock, hook=hook)
        g = make_gate(tmp_path, 1, clock, sleeper)
        h = holds(g, "lightcraft")
        queue_waiter(g, "T", "tag", clock())
        assert g.yield_turn("match") == POLL
        assert sleeper.calls == [POLL]
        assert seen == [[("lightcraft", 4242, 1)]]                        # only the lightcraft holder: the yielding reader holds nothing
        h.close()

    def test_with_nobody_queued_it_returns_at_once(self, tmp_path, holds):
        clock = Clock()
        sleeper = Sleeper(clock)
        g = make_gate(tmp_path, 1, clock, sleeper)
        holds(g, "lightcraft")
        assert g.yield_turn("match") == 0.0
        assert sleeper.calls == []

    def test_a_queued_reader_of_a_higher_number_does_not_hold_it_back(self, tmp_path, holds):
        clock = Clock()
        sleeper = Sleeper(clock)
        g = make_gate(tmp_path, 1, clock, sleeper)
        h = holds(g, "lightcraft")
        queue_waiter(g, "M", "match", clock())
        assert g.yield_turn("tag") == 0.0
        assert sleeper.calls == []
        h.close()


class TestDefaultGate:
    @pytest.fixture(autouse=True)
    def private_gate(self, tmp_path, monkeypatch):
        monkeypatch.setattr(nas_io, "_defaults", {})
        monkeypatch.setattr(settings, "NAS_IO_DB", tmp_path / "state" / "nas_io.sqlite3")

    @pytest.mark.parametrize("off", ["0", "false", "no"])
    def test_the_off_switch_gives_a_gate_that_waits_for_nobody_and_creates_no_file(self, tmp_path, monkeypatch, off):
        monkeypatch.setattr(settings, "NAS_IO_GATE", off)
        g = nas_io.gate()
        assert isinstance(g, NullGate)
        with g.acquire("tag") as waited:
            assert waited == 0.0
        assert g.yield_turn("match") == 0.0
        assert not (tmp_path / "state").exists()

    def test_the_gate_on_gives_one_gate_per_configuration_built_from_the_settings(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "NAS_IO_GATE", "1")
        monkeypatch.setattr(settings, "NAS_IO_SLOTS", 3)
        monkeypatch.setattr(settings, "NAS_IO_HOLD_TTL_S", 77.0)
        monkeypatch.setattr(settings, "NAS_IO_HOLD_MAX_S", 600.0)
        g = nas_io.gate()
        assert isinstance(g, NasIoGate)
        assert (g.path, g.slots, g.hold_ttl_s, g.hold_max_s) == (tmp_path / "state" / "nas_io.sqlite3", 3, 77.0, 600.0)
        assert nas_io.gate() is g
        assert not (tmp_path / "state").exists()                          # lazily opened: building it creates no file
        other = nas_io.gate(slots=1)
        assert (other is g, other.slots) == (False, 1)

    def test_arguments_override_the_settings(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "NAS_IO_GATE", "0")
        assert isinstance(nas_io.gate(enabled="1", path=tmp_path / "x.sqlite3", slots=2, hold_ttl_s=9), NasIoGate)
        assert isinstance(nas_io.gate(enabled=False), NullGate)


class RecordingGate:
    """Records acquire(stage) calls and the order of enter/exit around the body."""

    def __init__(self):
        self.events = []

    @contextlib.contextmanager
    def acquire(self, stage):
        self.events.append(("enter", stage))
        yield 0.0
        self.events.append(("exit", stage))


class TestRawCacheGate:
    @staticmethod
    def put(nas, rel, data):
        p = nas / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return p

    def test_a_fill_copies_inside_a_slot_of_its_stage_and_frees_it_afterwards(self, tmp_path):
        nas = tmp_path / "nas"
        src = self.put(nas, "2024/d/x.NEF", b"raw bytes")
        g = NasIoGate(tmp_path / "gate.sqlite3", 2)
        during = []

        def copy(s, d):
            during.append(holders(g))
            return raw_cache.copy_file(s, d)
        c = RawCache(tmp_path / "cache", 10**9, nas, min_free_gb=0, copy_fn=copy, gate=g)
        got = c.local(src, stage="render")
        assert got.read_bytes() == b"raw bytes"
        assert during == [[("render", os.getpid(), PRIO["render"])]]
        assert holders(g) == []

    def test_a_hit_takes_no_slot_and_a_fill_takes_exactly_one(self, tmp_path):
        nas = tmp_path / "nas"
        src = self.put(nas, "2024/d/x.NEF", b"abc")
        gate = RecordingGate()
        c = RawCache(tmp_path / "cache", 10**9, nas, min_free_gb=0, gate=gate)
        c.local(src, stage="tag")
        c.local(src, stage="render")                                      # served from the copy
        assert gate.events == [("enter", "tag"), ("exit", "tag")]

    def test_a_fallback_for_a_full_disk_takes_no_slot(self, tmp_path):
        nas = tmp_path / "nas"
        src = self.put(nas, "2024/d/x.NEF", b"abc")
        gate = RecordingGate()
        full = RawCache(tmp_path / "cache", 10**9, nas, min_free_gb=10**9, gate=gate)
        assert full.local(src, stage="tag") == src
        assert gate.events == []
        roomy = RawCache(tmp_path / "cache", 10**9, nas, min_free_gb=0, gate=gate)
        assert roomy.local(src, stage="tag") == tmp_path / "cache" / "2024" / "d" / "x.NEF"
        assert gate.events == [("enter", "tag"), ("exit", "tag")]

    def test_a_path_outside_the_nas_takes_no_slot(self, tmp_path):
        other = self.put(tmp_path, "elsewhere/y.NEF", b"abc")
        gate = RecordingGate()
        c = RawCache(tmp_path / "cache", 10**9, tmp_path / "nas", min_free_gb=0, gate=gate)
        assert c.local(other, stage="tag") == other
        assert gate.events == []


def test_three_processes_never_hold_more_than_the_slots_and_tag_waits_less_than_render(tmp_path):
    """One tag and two render readers loop acquire() for 10 s against one gate file with 2 slots, each holding 50 ms."""
    db = tmp_path / "gate.sqlite3"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(Path(nas_io.__file__).parent.parent), os.environ.get("PYTHONPATH", "")])}
    outs, procs = {}, []
    for name, stage in (("tag", "tag"), ("render1", "render"), ("render2", "render")):
        outs[name] = tmp_path / f"{name}.json"
        procs.append(subprocess.Popen([sys.executable, str(WORKER), str(db), stage, "10", "0.05", str(outs[name])], env=env))
    for p in procs:
        assert p.wait(timeout=120) == 0
    records = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in outs.items()}
    assert min(len(r) for r in records.values()) >= 20                    # every process really competed
    edges = sorted([(r["enter"], 1) for rs in records.values() for r in rs] + [(r["exit"], -1) for rs in records.values() for r in rs])
    held = peak = 0
    for _, delta in edges:                                                # sorted: an exit sorts before an enter at the same instant
        held += delta
        peak = max(peak, held)
    assert peak == 2
    mean = lambda rs: sum(r["wait"] for r in rs) / len(rs)                # noqa: E731
    render = records["render1"] + records["render2"]
    assert mean(records["tag"]) < mean(render)
