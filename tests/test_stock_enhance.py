"""Unit tests for bio_automation.stock_enhance (hermetic -- topaz.enhance and
disk_guard.wait_for_output are mocked; no real GPU/model inference)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from bio_automation import stock_enhance


def _make_jpgs(src_dir: Path, names: list[str]) -> list[Path]:
    src_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in names:
        p = src_dir / name
        p.write_bytes(b"not a real jpeg, just a fixture")
        paths.append(p)
    return paths


def _recording_enhance(monkeypatch, *, write_dst: bool = True):
    """A recording stub (rule 4): records every call instead of doing real
    OpenVINO inference, so tests can assert exactly what was/wasn't called.
    write_dst=True mimics the real topaz.enhance's effect of producing a
    file at dst on success, which is what stock_enhance's resume check
    (dst.exists()) actually depends on."""
    calls = []

    def _fake_enhance(src, dst, mode, cache, min_free_gb):
        calls.append({"src": src, "dst": dst, "mode": mode, "min_free_gb": min_free_gb})
        if write_dst:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(b"sharpened fixture bytes")

    monkeypatch.setattr(stock_enhance.topaz, "enhance", _fake_enhance)
    return calls


def _always_room(monkeypatch):
    monkeypatch.setattr(stock_enhance.disk_guard, "wait_for_output", lambda *a, **k: True)


class TestRun:
    def test_sharpens_unenhanced_file_into_out_dir_without_touching_source(self, tmp_path, monkeypatch):
        src_dir, out_dir = tmp_path / "src", tmp_path / "out"
        calls = _recording_enhance(monkeypatch)
        _always_room(monkeypatch)
        [jpg] = _make_jpgs(src_dir, ["a.jpg"])
        original_bytes = jpg.read_bytes()

        result = stock_enhance.run(src_dir, out_dir=out_dir, min_free_gb=7.0)

        assert result == {"done": 1, "skipped": 0, "failed": 0, "total": 1}
        assert calls == [{"src": jpg, "dst": out_dir / "a.jpg", "mode": "stock", "min_free_gb": 7.0}]
        assert jpg.read_bytes() == original_bytes  # the source is NEVER touched

    def test_skips_file_already_present_in_out_dir(self, tmp_path, monkeypatch):
        """Positive twin to the sharpen test above, on the same kind of
        fixture: a file already present in out_dir must be left alone --
        enhance must never be called for it (rule 4: the stub that must not
        be called records its calls; the list must be empty)."""
        src_dir, out_dir = tmp_path / "src", tmp_path / "out"
        calls = _recording_enhance(monkeypatch)
        _always_room(monkeypatch)
        _make_jpgs(src_dir, ["a.jpg"])
        out_dir.mkdir(parents=True)
        (out_dir / "a.jpg").write_bytes(b"already enhanced")

        result = stock_enhance.run(src_dir, out_dir=out_dir, min_free_gb=7.0)

        assert result == {"done": 0, "skipped": 1, "failed": 0, "total": 1}
        assert calls == []

    def test_failed_enhance_leaves_no_partial_file_in_out_dir(self, tmp_path, monkeypatch):
        src_dir, out_dir = tmp_path / "src", tmp_path / "out"
        _always_room(monkeypatch)

        def _raising_enhance(src, dst, mode, cache, min_free_gb):
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(b"corrupt partial write")  # simulates a crash mid-save
            raise RuntimeError("CUDA DLL folders not found")

        monkeypatch.setattr(stock_enhance.topaz, "enhance", _raising_enhance)
        _make_jpgs(src_dir, ["a.jpg"])

        result = stock_enhance.run(src_dir, out_dir=out_dir, min_free_gb=7.0)

        assert result == {"done": 0, "skipped": 0, "failed": 1, "total": 1}
        assert not (out_dir / "a.jpg").exists()  # the partial write is cleaned up, not left to look "done"

    def test_disk_guard_refusal_stops_before_any_enhance_call(self, tmp_path, monkeypatch):
        src_dir, out_dir = tmp_path / "src", tmp_path / "out"
        calls = _recording_enhance(monkeypatch)
        monkeypatch.setattr(stock_enhance.disk_guard, "wait_for_output", lambda *a, **k: False)
        _make_jpgs(src_dir, ["a.jpg", "b.jpg"])

        result = stock_enhance.run(src_dir, out_dir=out_dir, min_free_gb=7.0)

        assert result == {"done": 0, "skipped": 0, "failed": 0, "total": 2}
        assert calls == []

    def test_limit_caps_the_number_of_files_processed(self, tmp_path, monkeypatch):
        src_dir, out_dir = tmp_path / "src", tmp_path / "out"
        calls = _recording_enhance(monkeypatch)
        _always_room(monkeypatch)
        _make_jpgs(src_dir, ["a.jpg", "b.jpg", "c.jpg"])

        result = stock_enhance.run(src_dir, out_dir=out_dir, min_free_gb=7.0, limit=2)

        assert result == {"done": 2, "skipped": 0, "failed": 0, "total": 2}
        assert len(calls) == 2

    def test_uses_passed_in_cache_instead_of_opening_its_own(self, tmp_path, monkeypatch):
        """Positive twin confirming the caller-supplied cache path actually
        reaches topaz.enhance, exact object identity, not just 'some cache'."""
        from bio_automation import topaz

        src_dir, out_dir = tmp_path / "src", tmp_path / "out"
        sentinel_cache = topaz.ModelCache()
        seen = []

        def _fake_enhance(src, dst, mode, cache, min_free_gb):
            seen.append(cache)

        monkeypatch.setattr(stock_enhance.topaz, "enhance", _fake_enhance)
        _always_room(monkeypatch)
        _make_jpgs(src_dir, ["a.jpg"])

        stock_enhance.run(src_dir, out_dir=out_dir, min_free_gb=7.0, cache=sentinel_cache)

        assert seen == [sentinel_cache]


class _RecordingQueue:
    """A recording stub (rule 4): records every publish() call instead of
    writing to a real queue.sqlite3, so tests can assert exactly what was/
    wasn't published."""

    def __init__(self):
        self.calls = []

    def publish(self, topic, payload):
        self.calls.append((topic, payload))
        return {"id": len(self.calls)}


class TestPublishReview:
    def test_links_new_file_into_review_folder_and_publishes_reindex(self, tmp_path):
        out_dir, gallery_dir = tmp_path / "out", tmp_path / "gallery"
        out_dir.mkdir()
        (out_dir / "a.jpg").write_bytes(b"enhanced fixture bytes")
        queue = _RecordingQueue()

        result = stock_enhance.publish_review(out_dir, gallery_dir=gallery_dir, review_name="x", queue=queue)

        dst = gallery_dir / "_review" / "x" / "a.jpg"
        assert result == {"linked": 1, "already_there": 0, "total": 1, "reindex_message": {"id": 1}}
        assert dst.exists()
        assert os.path.samefile(out_dir / "a.jpg", dst)  # a real hardlink, not a copy
        assert queue.calls == [("gallery.reindex", {"gallery_dir": str(gallery_dir), "folders": ["_review/x"]})]

    def test_rerun_skips_already_linked_file_and_does_not_republish(self, tmp_path):
        """Positive twin to the link test above, on the same kind of fixture:
        calling publish_review again with nothing new must link nothing and
        must never call queue.publish (rule 4: empty calls list)."""
        out_dir, gallery_dir = tmp_path / "out", tmp_path / "gallery"
        out_dir.mkdir()
        (out_dir / "a.jpg").write_bytes(b"enhanced fixture bytes")
        queue = _RecordingQueue()
        stock_enhance.publish_review(out_dir, gallery_dir=gallery_dir, review_name="x", queue=queue)

        result = stock_enhance.publish_review(out_dir, gallery_dir=gallery_dir, review_name="x", queue=queue)

        assert result == {"linked": 0, "already_there": 1, "total": 1, "reindex_message": None}
        assert queue.calls == [("gallery.reindex", {"gallery_dir": str(gallery_dir), "folders": ["_review/x"]})]  # only the first run's call

    def test_runs_without_a_queue(self, tmp_path):
        out_dir, gallery_dir = tmp_path / "out", tmp_path / "gallery"
        out_dir.mkdir()
        (out_dir / "a.jpg").write_bytes(b"enhanced fixture bytes")

        result = stock_enhance.publish_review(out_dir, gallery_dir=gallery_dir, review_name="x")

        assert result == {"linked": 1, "already_there": 0, "total": 1, "reindex_message": None}
        assert (gallery_dir / "_review" / "x" / "a.jpg").exists()
