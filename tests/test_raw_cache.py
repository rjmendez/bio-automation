"""RawCache: a NAS raw is read once into a bounded working copy on a local disk; everything else is left alone.

The fake NAS and the cache root are both under tmp_path. copy_fn / disk_usage_fn / sleep_fn / time_fn are the dependencies the cache is given;
the cache's own logic (keys, sidecars, locks, eviction, the byte check) runs for real, on real files.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections import namedtuple
from pathlib import Path

import pytest

from bio_automation import raw_cache, stock_select
from bio_automation.raw_cache import RawCache

FIXTURES = Path(__file__).parent / "fixtures"
Usage = namedtuple("Usage", "total used free")


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class Recorder:
    """Stands in for the copy: records (src name, dst name) of every copy and does the real one (or `behaviour` instead)."""

    def __init__(self, behaviour=None):
        self.calls = []
        self.behaviour = behaviour

    def __call__(self, src, dst):
        self.calls.append((Path(src).name, Path(dst).name.split(".part.")[0]))
        return (self.behaviour or raw_cache.copy_file)(src, dst)


class Clock:
    """A time source that moves one second per reading, so entries touched in a known order sort in that order."""

    def __init__(self, start=1_000_000.0):
        self.t = start

    def __call__(self):
        self.t += 1.0
        return self.t


@pytest.fixture()
def env(tmp_path):
    nas, root = tmp_path / "nas", tmp_path / "cache"
    nas.mkdir()
    return nas, root


def put(nas: Path, rel: str, data: bytes) -> Path:
    p = nas / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


def cache_for(env, *, max_bytes=10**9, copy=None, **kw):
    nas, root = env
    events = []
    kw.setdefault("min_free_gb", 0)
    return RawCache(root, max_bytes, nas, on_read=events.append, copy_fn=copy or Recorder(), **kw), events


def files(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


class TestFill:
    def test_a_nas_raw_is_copied_to_its_relative_path_with_the_same_bytes_and_served_from_there_afterwards(self, env):
        nas, root = env
        src = put(nas, "2024/2024-05-18/x.NEF", bytes(range(256)) * 12288)      # 3 MiB: more than one copy buffer's worth of 1 MiB
        rec = Recorder()
        c, events = cache_for(env, copy=rec)
        first = c.local(src, stage="tag")
        second = c.local(src, stage="render")
        assert first == root / "2024" / "2024-05-18" / "x.NEF" == second
        assert sha(first) == sha(src)
        assert rec.calls == [("x.NEF", "x.NEF")]                                  # one copy across both calls
        assert [(e["stage"], e["rel"], e["bytes"], e["outcome"]) for e in events] == [
            ("tag", "2024/2024-05-18/x.NEF", 3145728, "fill"), ("render", "2024/2024-05-18/x.NEF", 3145728, "hit")]
        assert all(e["seconds"] >= 0 for e in events)

    def test_a_successful_fill_leaves_only_the_raw_and_its_sidecar(self, env):
        nas, root = env
        src = put(nas, "2024/d/x.NEF", b"abc")
        cache_for(env)[0].local(src, stage="tag")
        assert files(root) == ["2024/d/x.NEF", "2024/d/x.NEF.src.json"]
        st = src.stat()
        assert json.loads((root / "2024/d/x.NEF.src.json").read_text(encoding="utf-8")) == {"size": 3, "mtime_ns": st.st_mtime_ns}

    def test_a_source_rewritten_with_another_size_is_copied_again(self, env):
        nas, _ = env
        src = put(nas, "2024/d/x.NEF", b"first")
        rec = Recorder()
        c, _ = cache_for(env, copy=rec)
        c.local(src, stage="tag")
        src.write_bytes(b"second, longer")
        got = c.local(src, stage="tag")
        assert rec.calls == [("x.NEF", "x.NEF"), ("x.NEF", "x.NEF")]
        assert got.read_bytes() == b"second, longer"

    def test_a_source_rewritten_with_the_same_size_and_a_new_mtime_is_copied_again(self, env):
        nas, _ = env
        src = put(nas, "2024/d/x.NEF", b"aaaa")
        os.utime(src, ns=(1_700_000_000_000_000_000, 1_700_000_000_000_000_000))
        rec = Recorder()
        c, _ = cache_for(env, copy=rec)
        c.local(src, stage="tag")
        src.write_bytes(b"bbbb")
        os.utime(src, ns=(1_700_000_001_000_000_000, 1_700_000_001_000_000_000))
        assert c.local(src, stage="tag").read_bytes() == b"bbbb"
        assert len(rec.calls) == 2

    def test_an_unchanged_source_is_copied_once(self, env):
        nas, _ = env
        src = put(nas, "2024/d/x.NEF", b"same")
        rec = Recorder()
        c, _ = cache_for(env, copy=rec)
        c.local(src, stage="tag")
        c.local(src, stage="tag")
        c.local(src, stage="tag")
        assert len(rec.calls) == 1

    def test_a_cached_file_without_a_matching_sidecar_is_not_trusted(self, env):
        nas, root = env
        src = put(nas, "2024/d/x.NEF", b"real")
        put(root, "2024/d/x.NEF", b"junk")                                         # a file at the key that this cache never wrote
        rec = Recorder()
        got = cache_for(env, copy=rec)[0].local(src, stage="tag")
        assert got.read_bytes() == b"real" and len(rec.calls) == 1

    def test_a_path_outside_the_nas_is_returned_unchanged_with_no_copy(self, env, tmp_path):
        _, root = env
        other = put(tmp_path / "elsewhere", "lib/y.NEF", b"y")
        rec = Recorder()
        c, events = cache_for(env, copy=rec)
        assert c.local(other, stage="tag") == other
        assert (rec.calls, events, root.exists()) == ([], [], False)

    def test_a_path_inside_the_nas_is_copied(self, env):
        nas, root = env
        src = put(nas, "2024/d/y.NEF", b"y")
        rec = Recorder()
        assert cache_for(env, copy=rec)[0].local(src, stage="tag") == root / "2024" / "d" / "y.NEF"
        assert rec.calls == [("y.NEF", "y.NEF")]


class TestFailOpen:
    def test_a_copy_that_dies_halfway_returns_the_nas_path_and_leaves_nothing_behind(self, env):
        nas, root = env
        src = put(nas, "2024/d/x.NEF", b"0123456789")

        def half_then_die(s, d):
            Path(d).write_bytes(Path(s).read_bytes()[:5])
            raise OSError("link dropped")

        c, events = cache_for(env, copy=Recorder(half_then_die))
        assert c.local(src, stage="tag") == src
        assert [f for f in files(root)] == []
        assert [(e["outcome"], e["bytes"]) for e in events] == [("fallback:OSError", 10)]

    def test_a_copy_that_reports_fewer_bytes_than_the_source_is_not_kept(self, env):
        nas, root = env
        src = put(nas, "2024/d/x.NEF", b"0123456789")

        def short(s, d):
            Path(d).write_bytes(Path(s).read_bytes()[:4])
            return 4

        c, events = cache_for(env, copy=Recorder(short))
        assert c.local(src, stage="tag") == src
        assert files(root) == [] and [e["outcome"] for e in events] == ["fallback:OSError"]

    def test_a_copy_that_claims_the_full_size_but_wrote_less_is_not_kept(self, env):
        nas, root = env
        src = put(nas, "2024/d/x.NEF", b"0123456789")

        def lies(s, d):
            Path(d).write_bytes(Path(s).read_bytes()[:4])
            return 10

        c, _ = cache_for(env, copy=Recorder(lies))
        assert c.local(src, stage="tag") == src and files(root) == []

    def test_a_missing_source_is_returned_as_it_is_and_reported(self, env):
        nas, _ = env
        ghost = nas / "2024" / "d" / "gone.NEF"
        rec = Recorder()
        c, events = cache_for(env, copy=rec)
        assert c.local(ghost, stage="tag") == ghost
        assert (rec.calls, [e["outcome"] for e in events]) == ([], ["fallback:FileNotFoundError"])

    def test_a_failing_report_does_not_fail_the_read(self, env):
        nas, root = env
        src = put(nas, "2024/d/x.NEF", b"abc")
        seen = []

        def bad_report(event):
            seen.append(event["outcome"])
            raise RuntimeError("log disk full")

        c = RawCache(root, 10**9, nas, min_free_gb=0, on_read=bad_report)
        assert c.local(src, stage="tag") == root / "2024" / "d" / "x.NEF"
        assert seen == ["fill"]

    def test_low_free_disk_reads_the_nas_and_copies_nothing(self, env):
        nas, root = env
        src = put(nas, "2024/d/x.NEF", b"0123456789")
        asked = []

        def disk(path):
            asked.append(Path(path))
            return Usage(100 * 10**9, 90 * 10**9, 10 * 10**9)                      # 10 GB free

        rec = Recorder()
        c, events = cache_for(env, copy=rec, min_free_gb=15, disk_usage_fn=disk)
        assert c.local(src, stage="tag") == src
        assert (rec.calls, [e["outcome"] for e in events], asked) == ([], ["fallback:disk"], [root])

    def test_with_enough_free_disk_the_same_file_is_copied(self, env):
        nas, root = env
        src = put(nas, "2024/d/x.NEF", b"0123456789")
        rec = Recorder()
        c, events = cache_for(env, copy=rec, min_free_gb=0, disk_usage_fn=lambda p: Usage(100 * 10**9, 90 * 10**9, 10 * 10**9))
        assert c.local(src, stage="tag") == root / "2024" / "d" / "x.NEF"
        assert len(rec.calls) == 1 and [e["outcome"] for e in events] == ["fill"]

    @pytest.mark.parametrize("free,outcome", [(2500, "fallback:disk"), (3500, "fill")])
    def test_the_file_being_copied_must_still_leave_min_free_after_it(self, env, free, outcome):
        nas, _ = env
        src = put(nas, "2024/d/x.NEF", b"x" * 2000)                                  # min_free 1000 B: 2500 - 2000 < 1000 <= 3500 - 2000
        c, events = cache_for(env, min_free_gb=0.000001, disk_usage_fn=lambda p: Usage(10**9, 0, free))
        assert (c.local(src, stage="tag") == src) is (outcome == "fallback:disk")
        assert [e["outcome"] for e in events] == [outcome]


class TestEviction:
    def fill_three(self, env, order):
        nas, root = env
        srcs = {n: put(nas, f"2024/d/{n}.NEF", n.encode() * 100) for n in "abc"}      # 100 bytes each
        clock = Clock()
        c, _ = cache_for(env, max_bytes=250, time_fn=clock)
        for n in order:
            c.local(srcs[n], stage="tag")
        return c, srcs, root

    def test_the_least_recently_touched_entry_goes_when_a_third_is_filled(self, env):
        _, _, root = self.fill_three(env, "abc")
        assert files(root) == ["2024/d/b.NEF", "2024/d/b.NEF.src.json", "2024/d/c.NEF", "2024/d/c.NEF.src.json"]

    def test_touching_the_first_entry_makes_the_second_the_one_evicted(self, env):
        nas, root = env
        srcs = {n: put(nas, f"2024/d/{n}.NEF", n.encode() * 100) for n in "abc"}
        c, _ = cache_for(env, max_bytes=250, time_fn=Clock())
        c.local(srcs["a"], stage="tag")
        c.local(srcs["b"], stage="tag")
        c.local(srcs["a"], stage="tag")                                               # a hit: a is now the most recent
        c.local(srcs["c"], stage="tag")
        assert files(root) == ["2024/d/a.NEF", "2024/d/a.NEF.src.json", "2024/d/c.NEF", "2024/d/c.NEF.src.json"]

    def test_the_entry_just_filled_is_never_the_one_evicted(self, env):
        nas, root = env
        src = put(nas, "2024/d/big.NEF", b"z" * 500)
        c, _ = cache_for(env, max_bytes=250)                                    # larger than the whole bound
        assert c.local(src, stage="tag") == root / "2024" / "d" / "big.NEF"
        assert files(root) == ["2024/d/big.NEF", "2024/d/big.NEF.src.json"]

    def test_evict_removes_exactly_the_given_entries_and_leaves_another_days_bytes_alone(self, env):
        nas, root = env
        names = ["2024/d1/a.NEF", "2024/d1/b.NEF", "2024/d2/c.NEF"]
        c, _ = cache_for(env)
        for n in names:
            c.local(put(nas, n, n.encode()), stage="tag")
        before = {f: (root / f).read_bytes() for f in files(root) if f.startswith("2024/d2/")}
        c.evict(["2024/d1/a.NEF"])
        assert files(root) == ["2024/d1/b.NEF", "2024/d1/b.NEF.src.json", "2024/d2/c.NEF", "2024/d2/c.NEF.src.json"]
        assert {f: (root / f).read_bytes() for f in files(root) if f.startswith("2024/d2/")} == before

    def test_evicting_a_day_s_last_entry_removes_its_empty_folder(self, env):
        nas, root = env
        c, _ = cache_for(env)
        c.local(put(nas, "2024/d1/a.NEF", b"a"), stage="tag")
        c.evict(["2024/d1/a.NEF"])
        assert [p.name for p in root.rglob("*")] == []

    def test_evicting_an_entry_that_is_not_there_is_a_no_op(self, env):
        nas, root = env
        c, _ = cache_for(env)
        c.local(put(nas, "2024/d1/a.NEF", b"a"), stage="tag")
        c.evict(["2024/d1/never-filled.NEF"])
        assert files(root) == ["2024/d1/a.NEF", "2024/d1/a.NEF.src.json"]

    def test_an_entry_held_open_is_skipped_and_kept_and_goes_once_released(self, env):
        nas, root = env
        c, _ = cache_for(env)
        keys = [c.local(put(nas, f"2024/d/{n}.NEF", n.encode()), stage="tag") for n in "ab"]
        with open(keys[0], "rb"):                                                    # LightCraft has it open
            c.evict(["2024/d/a.NEF", "2024/d/b.NEF"])
            assert files(root) == ["2024/d/a.NEF", "2024/d/a.NEF.src.json"]            # a: kept with its sidecar; b: gone
        c.evict(["2024/d/a.NEF"])
        assert files(root) == []

    def test_the_size_bound_skips_an_entry_held_open_and_removes_the_next_oldest(self, env):
        nas, root = env
        srcs = {n: put(nas, f"2024/d/{n}.NEF", n.encode() * 100) for n in "abc"}
        c, _ = cache_for(env, max_bytes=250, time_fn=Clock())
        first = c.local(srcs["a"], stage="tag")
        c.local(srcs["b"], stage="tag")
        with open(first, "rb"):
            c.local(srcs["c"], stage="tag")
            assert files(root) == ["2024/d/a.NEF", "2024/d/a.NEF.src.json", "2024/d/c.NEF", "2024/d/c.NEF.src.json"]

    def test_a_dead_process_s_old_partial_copy_is_swept_but_a_young_one_is_not(self, env):
        nas, root = env
        old = put(root, "2024/d/old.NEF.part.111", b"x" * 50)
        young = put(root, "2024/d/young.NEF.part.222", b"y" * 50)
        os.utime(old, (1000, 1000))
        c, _ = cache_for(env, lock_stale_s=600)
        c.local(put(nas, "2024/d/a.NEF", b"a"), stage="tag")
        assert files(root) == ["2024/d/a.NEF", "2024/d/a.NEF.src.json", "2024/d/young.NEF.part.222"]
        assert young.read_bytes() == b"y" * 50


class TestLocks:
    def test_a_fresh_lock_is_waited_out_and_then_the_entry_is_filled(self, env):
        nas, root = env
        src = put(nas, "2024/d/x.NEF", b"abc")
        lock = put(root, "2024/d/x.NEF.lock", b"999")
        slept = []

        def holder_finishes(s):
            slept.append(s)
            lock.unlink()

        rec = Recorder()
        c, events = cache_for(env, copy=rec, sleep_fn=holder_finishes)
        assert c.local(src, stage="tag") == root / "2024" / "d" / "x.NEF"
        assert (slept, len(rec.calls), [e["outcome"] for e in events]) == ([0.5], 1, ["fill"])
        assert files(root) == ["2024/d/x.NEF", "2024/d/x.NEF.src.json"]

    def test_a_waiter_uses_the_copy_the_lock_holder_made(self, env):
        nas, root = env
        src = put(nas, "2024/d/x.NEF", b"abc")
        lock = put(root, "2024/d/x.NEF.lock", b"999")
        slept = []

        def holder_finishes(s):                                                     # the other process completes its fill, then releases
            slept.append(s)
            put(root, "2024/d/x.NEF", b"abc")
            st = src.stat()
            put(root, "2024/d/x.NEF.src.json", json.dumps({"size": 3, "mtime_ns": st.st_mtime_ns}).encode())
            lock.unlink()

        rec = Recorder()
        c, events = cache_for(env, copy=rec, sleep_fn=holder_finishes)
        assert c.local(src, stage="tag") == root / "2024" / "d" / "x.NEF"
        assert (slept, rec.calls, [e["outcome"] for e in events]) == ([0.5], [], ["hit"])

    def test_a_stale_lock_is_taken_over_without_waiting(self, env):
        nas, root = env
        src = put(nas, "2024/d/x.NEF", b"abc")
        lock = put(root, "2024/d/x.NEF.lock", b"dead")
        os.utime(lock, (1000, 1000))
        slept = []

        def holder_finishes(s):                                                     # only reached if the lock were (wrongly) treated as live
            slept.append(s)
            lock.unlink()

        rec = Recorder()
        c, events = cache_for(env, copy=rec, lock_stale_s=600, sleep_fn=holder_finishes)
        assert c.local(src, stage="tag") == root / "2024" / "d" / "x.NEF"
        assert (slept, len(rec.calls), [e["outcome"] for e in events]) == ([], 1, ["fill"])
        assert files(root) == ["2024/d/x.NEF", "2024/d/x.NEF.src.json"]


class TestScoreIdentity:
    @pytest.mark.parametrize("name", ["sample_raw.NEF", "sample_raw_fungus.NEF"])
    def test_a_cached_raw_has_the_same_bytes_and_scores_exactly_as_the_original(self, name, tmp_path):
        c = RawCache(tmp_path / "cache", 10**10, FIXTURES, min_free_gb=0)
        cached = c.local(FIXTURES / name, stage="tag")
        assert cached == tmp_path / "cache" / name
        assert sha(cached) == sha(FIXTURES / name)
        want, got = stock_select.score_raw(FIXTURES / name), stock_select.score_raw(cached)
        assert (got.sharp, got.clip) == (want.sharp, want.clip)
        assert (want.sharp, want.clip) != (0.0, 0.0)


@pytest.mark.parametrize("value,want", [(None, True), ("1", True), ("0", False), ("false", False), ("no", False)])
def test_the_switch_reads_BIO_RAW_CACHE(value, want):
    child_env = {k: v for k, v in os.environ.items() if k != "BIO_RAW_CACHE"}
    if value is not None:
        child_env["BIO_RAW_CACHE"] = value
    out = subprocess.run([sys.executable, "-c", "from bio_automation import settings; print(settings.RAW_CACHE)"], env=child_env,
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == str(want)
