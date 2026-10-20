"""Unit + live-HTTP tests for bio_automation.review_console.

Pure functions (list_library, folder_assets, _resolve_folder,
_find_raw_path) run against a real on-disk fixture (ensure_layout/
save_assets), same convention as tests/test_pipeline.py. HTTP tests spin up
a real ThreadingHTTPServer on an ephemeral port (same pattern as
tests/test_sync_receiver.py) so routing/serialization/host-checking is
exercised for real, not mocked; the one external dependency
(exif.extract_preview_bytes) is monkeypatched, same as tests/test_render.py.
"""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from bio_automation import review_console
from bio_automation.models import Asset
from bio_automation.paths import make_observation_path
from bio_automation.pipeline import ensure_layout, save_assets, scan_assets
from bio_automation.review_console import (
    ReviewConsoleHandler,
    UnknownFolderError,
    _find_raw_path,
    _resolve_folder,
    folder_assets,
    list_library,
    preview_jpeg_bytes,
)


def _seed_folder(base_dir, obsdate, slug, assets):
    folder = make_observation_path(base_dir, obsdate, slug)
    ensure_layout(folder)
    for a in assets:
        (folder / a.file).parent.mkdir(parents=True, exist_ok=True)
        (folder / a.file).write_bytes(b"raw")
    save_assets(folder, assets)
    return folder


class TestListLibrary:
    def test_counts_originals_by_status_per_folder(self, tmp_path):
        _seed_folder(tmp_path, "2026-01-05", "spider", [
            Asset(file="source/raw/DSC_0001.NEF", role="original", status="kept", selected=True),
            Asset(file="source/raw/DSC_0002.NEF", role="original", status="rejected", selected=False),
            Asset(file="source/raw/DSC_0003.NEF", role="original", status="ingested", selected=False),
        ])
        _seed_folder(tmp_path, "2026-02-09", "fungus", [
            Asset(file="source/raw/DSC_0004.NEF", role="original", status="kept", selected=True),
        ])

        rows = list_library(tmp_path)

        assert sorted(r["folder"] for r in rows) == [
            "2026/2026-01-05/obs-20260105-spider", "2026/2026-02-09/obs-20260209-fungus",
        ]
        spider = next(r for r in rows if r["folder"].endswith("spider"))
        assert spider["total"] == 3
        assert spider["counts"] == {"kept": 1, "rejected": 1, "ingested": 1}
        assert spider["cropped"] == 0

    def test_folders_with_zero_original_assets_are_omitted(self, tmp_path):
        """Positive twin to the counting test: a folder whose only assets
        are non-original (e.g. a stack_output) must not show up at all,
        not show up with total=0."""
        folder = make_observation_path(tmp_path, "2026-01-05", "spider")
        ensure_layout(folder)
        save_assets(folder, [
            Asset(file="work/FS-OutputImage.tiff", role="stack_output", status="ready", selected=True),
        ])

        assert list_library(tmp_path) == []

    def test_cropped_count_reflects_non_none_crop(self, tmp_path):
        _seed_folder(tmp_path, "2026-01-05", "spider", [
            Asset(file="source/raw/DSC_0001.NEF", role="original", status="kept", selected=True,
                  crop=[0.1, 0.1, 0.9, 0.9]),
            Asset(file="source/raw/DSC_0002.NEF", role="original", status="kept", selected=True),
        ])

        rows = list_library(tmp_path)

        assert rows[0]["cropped"] == 1


class TestFolderAssets:
    def test_returns_basename_status_and_crop_for_original_assets_only(self, tmp_path):
        folder = _seed_folder(tmp_path, "2026-01-05", "spider", [
            Asset(file="source/raw/DSC_0001.NEF", role="original", status="kept", selected=True,
                  crop=[0.0, 0.0, 1.0, 1.0]),
        ])
        save_assets(folder, scan_assets(folder) + [
            Asset(file="work/FS-OutputImage.tiff", role="stack_output", status="ready", selected=True),
        ])

        rel = str(folder.relative_to(tmp_path)).replace("\\", "/")
        rows = folder_assets(tmp_path, rel)

        assert rows == [{"file": "DSC_0001.NEF", "status": "kept", "crop": [0.0, 0.0, 1.0, 1.0],
                          "cache_ready": False}]

    def test_cache_ready_reflects_a_real_lightcraft_cache_file(self, tmp_path):
        """Positive twin: cache_ready must flip True once the LightCraft
        auto-adjusted preview actually exists on disk, not just stay False."""
        folder = _seed_folder(tmp_path, "2026-01-05", "spider", [
            Asset(file="source/raw/DSC_0001.NEF", role="original", status="kept", selected=True),
        ])
        from bio_automation.lightcraft_cache import cache_path_for
        cache_path = cache_path_for(folder, "DSC_0001.NEF")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(b"cached-jpeg")

        rel = str(folder.relative_to(tmp_path)).replace("\\", "/")
        rows = folder_assets(tmp_path, rel)

        assert rows[0]["cache_ready"] is True


class TestResolveFolder:
    def test_resolves_a_real_folder_by_its_relative_path(self, tmp_path):
        folder = _seed_folder(tmp_path, "2026-01-05", "spider", [
            Asset(file="source/raw/DSC_0001.NEF", role="original", status="kept", selected=True),
        ])
        rel = str(folder.relative_to(tmp_path)).replace("\\", "/")

        assert _resolve_folder(tmp_path, rel) == folder

    def test_rejects_an_unknown_path_including_traversal_attempts(self, tmp_path):
        """Covers both a plain typo'd path and a deliberate '..' traversal
        attempt -- both must be refused the same way, by checking against
        the real discovered folder list rather than trusting the input."""
        _seed_folder(tmp_path, "2026-01-05", "spider", [
            Asset(file="source/raw/DSC_0001.NEF", role="original", status="kept", selected=True),
        ])

        with pytest.raises(UnknownFolderError, match="no observation folder matches"):
            _resolve_folder(tmp_path, "2026/2026-01-05/obs-nonexistent")
        with pytest.raises(UnknownFolderError, match="no observation folder matches"):
            _resolve_folder(tmp_path, "../../../../etc")


class TestFindRawPath:
    def test_finds_the_real_path_of_an_original_asset_by_basename(self, tmp_path):
        folder = _seed_folder(tmp_path, "2026-01-05", "spider", [
            Asset(file="source/raw/DSC_0001.NEF", role="original", status="kept", selected=True),
        ])

        assert _find_raw_path(folder, "DSC_0001.NEF") == folder / "source/raw/DSC_0001.NEF"

    def test_raises_for_a_name_matching_no_original_asset(self, tmp_path):
        folder = _seed_folder(tmp_path, "2026-01-05", "spider", [
            Asset(file="source/raw/DSC_0001.NEF", role="original", status="kept", selected=True),
        ])

        with pytest.raises(FileNotFoundError, match="no original asset matches 'DSC_9999.NEF'"):
            _find_raw_path(folder, "DSC_9999.NEF")


class TestPreviewJpegBytes:
    def test_reads_the_resolved_raws_embedded_preview(self, tmp_path, monkeypatch):
        folder = _seed_folder(tmp_path, "2026-01-05", "spider", [
            Asset(file="source/raw/DSC_0001.NEF", role="original", status="kept", selected=True),
        ])
        calls = []

        def fake_extract(path):
            calls.append(path)
            return b"jpegbytes"

        monkeypatch.setattr(review_console, "extract_preview_bytes", fake_extract)
        rel = str(folder.relative_to(tmp_path)).replace("\\", "/")

        data = preview_jpeg_bytes(tmp_path, rel, "DSC_0001.NEF")

        assert data == b"jpegbytes"
        assert calls == [folder / "source/raw/DSC_0001.NEF"]

    def test_prefers_the_lightcraft_cache_when_present(self, tmp_path, monkeypatch):
        """Positive twin to the fallback test above: once a LightCraft
        auto-adjusted cache file exists, it must be served instead of the
        flat embedded preview -- extract_preview_bytes must not even be
        called."""
        from bio_automation.lightcraft_cache import cache_path_for

        folder = _seed_folder(tmp_path, "2026-01-05", "spider", [
            Asset(file="source/raw/DSC_0001.NEF", role="original", status="kept", selected=True),
        ])
        cache_path = cache_path_for(folder, "DSC_0001.NEF")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(b"auto-adjusted-jpeg")
        monkeypatch.setattr(review_console, "extract_preview_bytes",
                             lambda p: (_ for _ in ()).throw(AssertionError("must not use flat preview")))
        rel = str(folder.relative_to(tmp_path)).replace("\\", "/")

        data = preview_jpeg_bytes(tmp_path, rel, "DSC_0001.NEF")

        assert data == b"auto-adjusted-jpeg"


class _LiveServer:
    """Real ThreadingHTTPServer on an ephemeral port, base_dir set per test
    -- same pattern as tests/test_sync_receiver.py's live-HTTP fixture."""

    def __init__(self, base_dir: Path):
        class _Handler(ReviewConsoleHandler):
            pass

        _Handler.base_dir = base_dir
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def live(tmp_path, monkeypatch):
    monkeypatch.setattr(review_console, "extract_preview_bytes", lambda p: b"preview-bytes")
    srv = _LiveServer(tmp_path)
    yield srv
    srv.close()


def _get(srv, path):
    try:
        with urllib.request.urlopen(srv.url(path), timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _get_raw(srv, path):
    try:
        with urllib.request.urlopen(srv.url(path), timeout=10) as r:
            return r.status, r.read(), r.headers.get("Content-Type")
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers.get("Content-Type")


def _post(srv, path, payload):
    req = urllib.request.Request(
        srv.url(path), data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


class TestLiveLibraryAndFolder:
    def test_api_library_real_roundtrip(self, live, tmp_path):
        _seed_folder(tmp_path, "2026-01-05", "spider", [
            Asset(file="source/raw/DSC_0001.NEF", role="original", status="kept", selected=True),
        ])

        status, body = _get(live, "/api/library")

        assert status == 200
        assert body["folders"][0]["counts"] == {"kept": 1}

    def test_api_folder_unknown_path_returns_404(self, live):
        status, body = _get(live, "/api/folder?folder=nope")
        assert status == 404
        assert "no observation folder matches" in body["error"]

    def test_api_folder_missing_param_returns_400(self, live):
        status, body = _get(live, "/api/folder")
        assert status == 400
        assert body["error"] == "folder is required"


class TestLiveImg:
    def test_img_real_roundtrip_returns_jpeg_bytes(self, live, tmp_path):
        folder = _seed_folder(tmp_path, "2026-01-05", "spider", [
            Asset(file="source/raw/DSC_0001.NEF", role="original", status="kept", selected=True),
        ])
        rel = str(folder.relative_to(tmp_path)).replace("\\", "/")

        status, data, ctype = _get_raw(live, f"/img?folder={rel}&file=DSC_0001.NEF")

        assert status == 200
        assert data == b"preview-bytes"
        assert ctype == "image/jpeg"

    def test_img_unknown_file_returns_404(self, live, tmp_path):
        folder = _seed_folder(tmp_path, "2026-01-05", "spider", [
            Asset(file="source/raw/DSC_0001.NEF", role="original", status="kept", selected=True),
        ])
        rel = str(folder.relative_to(tmp_path)).replace("\\", "/")

        status, body = _get(live, f"/img?folder={rel}&file=DSC_9999.NEF")
        assert status == 404


class TestLiveKeepReject:
    def test_keep_really_updates_status_on_disk(self, live, tmp_path):
        folder = _seed_folder(tmp_path, "2026-01-05", "spider", [
            Asset(file="source/raw/DSC_0001.NEF", role="original", status="ingested", selected=False),
        ])
        rel = str(folder.relative_to(tmp_path)).replace("\\", "/")

        status, body = _post(live, "/api/keep", {"folder": rel, "files": ["DSC_0001.NEF"]})

        assert status == 200
        assert body["assets"] == [{"file": "DSC_0001.NEF", "status": "kept", "crop": None,
                                    "cache_ready": False}]
        assert scan_assets(folder)[0].status == "kept"
        assert scan_assets(folder)[0].selected is True

    def test_reject_really_updates_status_on_disk(self, live, tmp_path):
        """Positive twin to the keep test: the other status, same wiring."""
        folder = _seed_folder(tmp_path, "2026-01-05", "spider", [
            Asset(file="source/raw/DSC_0001.NEF", role="original", status="ingested", selected=False),
        ])
        rel = str(folder.relative_to(tmp_path)).replace("\\", "/")

        status, body = _post(live, "/api/reject", {"folder": rel, "files": ["DSC_0001.NEF"]})

        assert status == 200
        assert scan_assets(folder)[0].status == "rejected"
        assert scan_assets(folder)[0].selected is False

    def test_keep_missing_files_returns_400(self, live, tmp_path):
        folder = _seed_folder(tmp_path, "2026-01-05", "spider", [
            Asset(file="source/raw/DSC_0001.NEF", role="original", status="ingested", selected=False),
        ])
        rel = str(folder.relative_to(tmp_path)).replace("\\", "/")

        status, body = _post(live, "/api/keep", {"folder": rel, "files": []})
        assert status == 400


class TestLiveCrop:
    def test_crop_set_then_clear_real_roundtrip(self, live, tmp_path):
        folder = _seed_folder(tmp_path, "2026-01-05", "spider", [
            Asset(file="source/raw/DSC_0001.NEF", role="original", status="kept", selected=True),
        ])
        rel = str(folder.relative_to(tmp_path)).replace("\\", "/")

        status, body = _post(live, "/api/crop",
                              {"folder": rel, "file": "DSC_0001.NEF", "crop": [0.1, 0.2, 0.8, 0.9]})
        assert status == 200
        assert body["crop"] == [0.1, 0.2, 0.8, 0.9]
        assert scan_assets(folder)[0].crop == [0.1, 0.2, 0.8, 0.9]

        status, body = _post(live, "/api/crop", {"folder": rel, "file": "DSC_0001.NEF", "crop": None})
        assert status == 200
        assert body["crop"] is None
        assert scan_assets(folder)[0].crop is None

    def test_crop_bad_shape_returns_400(self, live, tmp_path):
        folder = _seed_folder(tmp_path, "2026-01-05", "spider", [
            Asset(file="source/raw/DSC_0001.NEF", role="original", status="kept", selected=True),
        ])
        rel = str(folder.relative_to(tmp_path)).replace("\\", "/")

        status, body = _post(live, "/api/crop", {"folder": rel, "file": "DSC_0001.NEF", "crop": [0.1, 0.2]})
        assert status == 400
        assert "crop must be" in body["error"]

    def test_crop_unknown_file_returns_404(self, live, tmp_path):
        folder = _seed_folder(tmp_path, "2026-01-05", "spider", [
            Asset(file="source/raw/DSC_0001.NEF", role="original", status="kept", selected=True),
        ])
        rel = str(folder.relative_to(tmp_path)).replace("\\", "/")

        status, body = _post(live, "/api/crop", {"folder": rel, "file": "DSC_9999.NEF", "crop": None})
        assert status == 404


class TestLiveCull:
    def test_cull_deletes_only_rejected_raws_real_filesystem(self, live, tmp_path):
        folder = _seed_folder(tmp_path, "2026-01-05", "spider", [
            Asset(file="source/raw/DSC_0001.NEF", role="original", status="rejected", selected=False),
            Asset(file="source/raw/DSC_0002.NEF", role="original", status="kept", selected=True),
        ])
        rel = str(folder.relative_to(tmp_path)).replace("\\", "/")

        status, body = _post(live, "/api/cull", {"folder": rel})

        assert status == 200
        assert body["deleted"] == ["DSC_0001.NEF"]
        assert not (folder / "source/raw/DSC_0001.NEF").exists()
        assert (folder / "source/raw/DSC_0002.NEF").exists()  # kept raw untouched

    def test_cull_unknown_folder_returns_404(self, live):
        status, body = _post(live, "/api/cull", {"folder": "nope"})
        assert status == 404


class TestLiveHostCheck:
    def test_wrong_host_header_is_refused(self, live):
        req = urllib.request.Request(live.url("/api/library"), headers={"Host": "evil.example.com"})
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req, timeout=10)
        assert exc_info.value.code == 403

    def test_root_page_is_served_as_html(self, live):
        with urllib.request.urlopen(live.url("/"), timeout=10) as r:
            assert r.status == 200
            assert r.headers.get("Content-Type").startswith("text/html")
            assert b"<html>" in r.read()


class TestLivePrepare:
    def test_starts_a_background_build_and_reports_cache_building(self, live, tmp_path, monkeypatch):
        folder = _seed_folder(tmp_path, "2026-01-05", "spider", [
            Asset(file="source/raw/DSC_0001.NEF", role="original", status="kept", selected=True),
        ])
        rel = str(folder.relative_to(tmp_path)).replace("\\", "/")
        release = threading.Event()
        started = threading.Event()

        def slow_build(folder_arg, raw_names, **kwargs):
            started.set()
            release.wait(timeout=10)
            return {"rendered": list(raw_names), "skipped": [], "failed": []}

        monkeypatch.setattr(review_console, "build_folder_cache", slow_build)

        status, body = _post(live, "/api/prepare", {"folder": rel})
        assert status == 200
        assert body == {"started": True, "already_running": False}

        assert started.wait(timeout=5)
        # While the build is in flight, /api/folder must report cache_building: True
        # and a second /api/prepare for the SAME folder must not start a second build.
        status, folder_body = _get(live, f"/api/folder?folder={rel}")
        assert folder_body["cache_building"] is True

        status, body2 = _post(live, "/api/prepare", {"folder": rel})
        assert body2 == {"started": False, "already_running": True}

        release.set()

    def test_unknown_folder_returns_404(self, live):
        status, body = _post(live, "/api/prepare", {"folder": "nope"})
        assert status == 404

    def test_missing_folder_returns_400(self, live):
        status, body = _post(live, "/api/prepare", {})
        assert status == 400


def _raw_post(srv, path, body: bytes, headers):
    req = urllib.request.Request(srv.url(path), data=body, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


class TestCsrfGuard:
    def _folder_with_rejected(self, tmp_path):
        folder = _seed_folder(tmp_path, "2026-01-05", "spider", [
            Asset(file="source/raw/DSC_0001.NEF", role="original", status="rejected", selected=False),
        ])
        return folder, str(folder.relative_to(tmp_path)).replace("\\", "/")

    def test_text_plain_cross_site_cull_is_refused_and_deletes_nothing(self, live, tmp_path):
        folder, rel = self._folder_with_rejected(tmp_path)

        status, resp = _raw_post(live, "/api/cull", json.dumps({"folder": rel}).encode(),
                                 {"Content-Type": "text/plain"})

        assert status == 403
        assert resp == {"error": "cross-site or non-JSON request refused"}
        assert (folder / "source/raw/DSC_0001.NEF").exists()

    def test_foreign_origin_with_json_content_type_is_refused(self, live, tmp_path):
        folder, rel = self._folder_with_rejected(tmp_path)

        status, _ = _raw_post(live, "/api/cull", json.dumps({"folder": rel}).encode(),
                              {"Content-Type": "application/json", "Origin": "http://evil.example"})

        assert status == 403
        assert (folder / "source/raw/DSC_0001.NEF").exists()

    def test_same_origin_json_cull_still_works(self, live, tmp_path):
        folder, rel = self._folder_with_rejected(tmp_path)

        status, resp = _raw_post(live, "/api/cull", json.dumps({"folder": rel}).encode(), {
            "Content-Type": "application/json; charset=utf-8",
            "Origin": f"http://127.0.0.1:{live.port}"})

        assert status == 200
        assert resp == {"deleted": ["DSC_0001.NEF"]}
        assert not (folder / "source/raw/DSC_0001.NEF").exists()


class TestCropValidationOverHttp:
    @pytest.mark.parametrize("bad", [["a", 0, 1, 1], [0.6, 0, 0.4, 1], [0, 0, 2, 1]])
    def test_malformed_crop_returns_400_and_is_not_stored(self, live, tmp_path, bad):
        folder = _seed_folder(tmp_path, "2026-01-05", "spider", [
            Asset(file="source/raw/DSC_0001.NEF", role="original", status="kept", selected=True),
        ])
        rel = str(folder.relative_to(tmp_path)).replace("\\", "/")

        status, body = _post(live, "/api/crop", {"folder": rel, "file": "DSC_0001.NEF", "crop": bad})

        assert status == 400
        assert "crop" in body["error"]
        assert scan_assets(folder)[0].crop is None
