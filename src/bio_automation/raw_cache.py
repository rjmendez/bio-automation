"""A bounded working cache of raws on a local disk, so a raw that lives on the NAS is read over the network once.

`RawCache.local(src, stage=...)` returns the path to read: a NAS raw is copied (one sequential read) to `root/<path relative to the NAS>` on
first use and served from there afterwards; any other path is returned untouched. The copy is byte-identical, so a measurement made on it
(`stock_select.score_raw`) is the measurement of the NAS file, and the key is stable, so a library photo imported from it stays valid after
the entry is evicted and filled again.

The cache never makes a caller fail: if the copy cannot be made (NAS unreachable, disk low, an I/O error) `local` returns the source path, which
is exactly what the caller would have read without a cache, and reports why through `on_read`.

Entries: `<key>` (the raw), `<key>.src.json` (the source's size and mtime_ns when it was copied: a changed source is copied again),
`<key>.lock` (held while one process fills the entry), `<key>.part.<pid>` (the copy in progress). Least recently touched entries are evicted
when the total passes `max_bytes`; an entry another process has open (PermissionError on Windows) is skipped and kept.
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import time
from collections.abc import Callable
from pathlib import Path

CHUNK = 8 * 1024 * 1024
SIDECAR = ".src.json"
LOCK = ".lock"
PART = re.compile(r"\.part\.\d+$")
LOCK_POLL_S = 0.5


def copy_file(src: Path, dst: Path) -> int:
    """Copy src to dst in CHUNK-sized reads; returns the number of bytes written."""
    n = 0
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        while buf := fi.read(CHUNK):
            fo.write(buf)
            n += len(buf)
    return n


class RawCache:
    def __init__(self, root: Path | str, max_bytes: int, nas_root: Path | str, *, min_free_gb: float, lock_stale_s: float = 600,
                 on_read: Callable[[dict], None] | None = None, copy_fn: Callable[[Path, Path], int] | None = None,
                 stat_fn: Callable = os.stat, time_fn: Callable[[], float] = time.time,
                 disk_usage_fn: Callable = shutil.disk_usage, sleep_fn: Callable[[float], None] = time.sleep):
        self.root = Path(root)
        self.max_bytes = max_bytes
        self.nas_root = Path(nas_root)
        self.min_free_gb = min_free_gb
        self.lock_stale_s = lock_stale_s
        self.on_read = on_read
        self._copy = copy_fn or copy_file
        self._stat = stat_fn
        self._time = time_fn
        self._disk_usage = disk_usage_fn
        self._sleep = sleep_fn

    # --- keys

    def rel(self, src: Path | str) -> str | None:
        """The entry name of a path under the NAS root (posix, relative to it), or None for any other path."""
        try:
            return Path(src).relative_to(self.nas_root).as_posix()
        except ValueError:
            return None

    def _key(self, rel: str) -> Path:
        return self.root / rel

    @staticmethod
    def _sidecar(key: Path) -> Path:
        return key.with_name(key.name + SIDECAR)

    @staticmethod
    def _lock(key: Path) -> Path:
        return key.with_name(key.name + LOCK)

    def _report(self, stage: str, rel: str, size: int, t0: float, outcome: str) -> None:
        if self.on_read is None:
            return
        try:
            self.on_read({"stage": stage, "rel": rel, "bytes": size, "seconds": time.perf_counter() - t0, "outcome": outcome})
        except Exception:               # a failing report must never fail the read it describes
            pass

    # --- reading

    def local(self, src: Path | str, *, stage: str) -> Path:
        """The path to read `src` from (see the module docstring). A path outside the NAS root is returned unchanged, with no I/O."""
        src = Path(src)
        rel = self.rel(src)
        if rel is None:
            return src
        t0 = time.perf_counter()
        try:
            st = self._stat(src)
        except OSError as e:
            self._report(stage, rel, 0, t0, f"fallback:{type(e).__name__}")
            return src
        size, mtime_ns = st.st_size, st.st_mtime_ns
        key = self._key(rel)
        if self._fresh(key, size, mtime_ns):
            self._touch(key)
            self._report(stage, rel, size, t0, "hit")
            return key
        try:
            outcome = self._fill(src, key, size, mtime_ns)
        except OSError as e:
            self._report(stage, rel, size, t0, f"fallback:{type(e).__name__}")
            return src
        self._report(stage, rel, size, t0, outcome)
        return src if outcome.startswith("fallback") else key

    def _fresh(self, key: Path, size: int, mtime_ns: int) -> bool:
        """The entry exists, was copied from a source of this size and mtime, and is whole."""
        try:
            meta = json.loads(self._sidecar(key).read_text(encoding="utf-8"))
            return meta == {"size": size, "mtime_ns": mtime_ns} and key.stat().st_size == size
        except (OSError, ValueError):
            return False

    def _touch(self, key: Path) -> None:
        with contextlib.suppress(OSError):
            now = self._time()
            os.utime(key, (now, now))

    def _fill(self, src: Path, key: Path, size: int, mtime_ns: int) -> str:
        """'fill' (copied now), 'hit' (another process copied it while this one waited) or 'fallback:disk'. OSError: nothing usable was made."""
        self.root.mkdir(parents=True, exist_ok=True)
        if self._disk_usage(self.root).free - size < self.min_free_gb * 1e9:
            return "fallback:disk"
        key.parent.mkdir(parents=True, exist_ok=True)
        lock = self._lock(key)
        self._acquire(lock)
        if self._fresh(key, size, mtime_ns):                      # filled by the process that held the lock
            self._release(lock)
            self._touch(key)
            return "hit"
        part = key.with_name(f"{key.name}.part.{os.getpid()}")
        try:
            written = self._copy(src, part)
            on_disk = part.stat().st_size
            if written != size or on_disk != size:
                raise OSError(f"copied {written} bytes ({on_disk} on disk) of {size}: {src}")
            self._sidecar(key).unlink(missing_ok=True)            # until the new sidecar is written the entry reads as stale
            os.replace(part, key)
            self._sidecar(key).write_text(json.dumps({"size": size, "mtime_ns": mtime_ns}), encoding="utf-8")
            self._touch(key)
        finally:
            with contextlib.suppress(OSError):
                part.unlink(missing_ok=True)
            self._release(lock)
        with contextlib.suppress(OSError):                       # a failed sweep must not discard the copy just made
            self._enforce(keep=key)
        return "fill"

    def _acquire(self, lock: Path) -> None:
        """Create the lock exclusively. While another process holds it, poll until it goes; a lock older than lock_stale_s is a dead
        process's and is taken over."""
        while True:
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    age = self._time() - lock.stat().st_mtime
                except FileNotFoundError:
                    continue
                if age > self.lock_stale_s:
                    with contextlib.suppress(FileNotFoundError):
                        lock.unlink()
                else:
                    self._sleep(LOCK_POLL_S)
                continue
            with os.fdopen(fd, "w") as f:
                f.write(str(os.getpid()))
            return

    @staticmethod
    def _release(lock: Path) -> None:
        with contextlib.suppress(OSError):
            lock.unlink(missing_ok=True)

    # --- eviction

    def evict(self, rels) -> None:
        """Delete these entries (names from `rel`) and their sidecars; one that is open elsewhere, or that cannot be removed, is kept."""
        for rel in rels:
            self._delete(self._key(rel))

    def _delete(self, key: Path) -> bool:
        try:
            key.unlink(missing_ok=True)
        except OSError:                                 # PermissionError: open in another process; keep the entry, sidecar included
            return False
        with contextlib.suppress(OSError):
            self._sidecar(key).unlink(missing_ok=True)
        parent = key.parent
        while parent != self.root:                       # a day folder left empty goes with its last entry
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent
        return True

    def _entries(self) -> list[tuple[float, int, Path]]:
        """(mtime, size, path) of every raw entry. Partial copies older than lock_stale_s (a dead process's) are removed."""
        out = []
        now = self._time()
        for p in self.root.rglob("*"):
            try:
                if not p.is_file():
                    continue
                st = p.stat()
            except OSError:
                continue
            if p.name.endswith((SIDECAR, LOCK)):
                continue
            if PART.search(p.name):
                if now - st.st_mtime > self.lock_stale_s:
                    with contextlib.suppress(OSError):
                        p.unlink()
                continue
            out.append((st.st_mtime, st.st_size, p))
        return out

    def _enforce(self, keep: Path) -> None:
        """Delete the least recently touched entries until the total is within max_bytes; never `keep`."""
        entries = sorted(self._entries(), key=lambda e: e[0])
        total = sum(size for _, size, _ in entries)
        for _, size, p in entries:
            if total <= self.max_bytes:
                break
            if p == keep:
                continue
            if self._delete(p):
                total -= size
