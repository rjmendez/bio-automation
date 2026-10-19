from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from bio_automation import sync_receiver as sr


class TestCheckSafeName:
    @pytest.mark.parametrize("name", ["DSC_0001.NEF", "manifest.json", "20261019T173423-abc123"])
    def test_accepts_real_filenames_and_stack_ids(self, name):
        assert sr.check_safe_name(name) is None  # returns cleanly, no exception

    @pytest.mark.parametrize("name", ["../../etc/passwd", "a/b", "a\\b", "..", ""])
    def test_rejects_path_traversal_and_separators(self, name):
        with pytest.raises(sr.UnsafeNameError):
            sr.check_safe_name(name)


class TestSaveUpload:
    def test_writes_exact_bytes_into_staging_dir(self, tmp_path):
        dest = sr.save_upload("stack1", "DSC_0001.NEF", b"REALRAWDATA", sync_dir=tmp_path)
        assert dest == tmp_path / ".incoming-stack1" / "DSC_0001.NEF"
        assert dest.read_bytes() == b"REALRAWDATA"

    def test_rejects_unsafe_filename_negative_twin(self, tmp_path):
        with pytest.raises(sr.UnsafeNameError):
            sr.save_upload("stack1", "../../evil.txt", b"x", sync_dir=tmp_path)

    def test_rejects_unsafe_stack_id(self, tmp_path):
        with pytest.raises(sr.UnsafeNameError):
            sr.save_upload("../escape", "file.NEF", b"x", sync_dir=tmp_path)


class TestCompleteStack:
    def test_raises_when_no_manifest_received_yet(self, tmp_path):
        (tmp_path / ".incoming-stack1").mkdir()
        with pytest.raises(sr.IncompleteStackError, match="no manifest.json"):
            sr.complete_stack("stack1", sync_dir=tmp_path)

    def test_raises_when_declared_file_missing_negative_twin(self, tmp_path):
        staging = tmp_path / ".incoming-stack1"
        staging.mkdir()
        (staging / "manifest.json").write_text(json.dumps({"files": ["DSC_0001.NEF", "DSC_0002.NEF"]}))
        (staging / "DSC_0001.NEF").write_bytes(b"x")
        with pytest.raises(sr.IncompleteStackError, match=r"DSC_0002\.NEF"):
            sr.complete_stack("stack1", sync_dir=tmp_path)
        assert staging.exists()  # never renamed on an incomplete stack

    def test_renames_atomically_when_all_files_present_positive_twin(self, tmp_path):
        staging = tmp_path / ".incoming-stack1"
        staging.mkdir()
        (staging / "manifest.json").write_text(json.dumps({"files": ["DSC_0001.NEF"]}))
        (staging / "DSC_0001.NEF").write_bytes(b"realbytes")

        final_dir = sr.complete_stack("stack1", sync_dir=tmp_path)

        assert final_dir == tmp_path / "stack1"
        assert final_dir.exists()
        assert not staging.exists()
        assert (final_dir / "DSC_0001.NEF").read_bytes() == b"realbytes"


class TestLiveHttpServer:
    """Real server, real socket, real HTTP requests -- not a mocked handler."""

    @pytest.fixture
    def running_server(self, tmp_path):
        token = "test-token-xyz"
        sr.SyncReceiverHandler.sync_dir = tmp_path
        sr.SyncReceiverHandler.token = token
        server = sr.ThreadingHTTPServer(("127.0.0.1", 0), sr.SyncReceiverHandler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        yield port, token
        server.shutdown()
        server.server_close()

    def _put(self, port, path, data, token):
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}{path}", data=data, method="PUT",
            headers={"Authorization": f"Bearer {token}"},
        )
        return urllib.request.urlopen(req, timeout=5)

    def _post(self, port, path, token):
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}{path}", data=b"", method="POST",
            headers={"Authorization": f"Bearer {token}"},
        )
        return urllib.request.urlopen(req, timeout=5)

    def test_upload_without_token_is_rejected(self, running_server, tmp_path):
        port, _token = running_server
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/upload/stack1/DSC_0001.NEF", data=b"x", method="PUT",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req, timeout=5)
        assert exc_info.value.code == 401
        assert not (tmp_path / ".incoming-stack1").exists()

    def test_upload_with_wrong_token_is_rejected_negative_twin(self, running_server, tmp_path):
        port, _token = running_server
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            self._put(port, "/upload/stack1/DSC_0001.NEF", b"x", token="wrong-token")
        assert exc_info.value.code == 401
        assert not (tmp_path / ".incoming-stack1").exists()

    def test_real_upload_and_complete_end_to_end(self, running_server, tmp_path):
        port, token = running_server
        manifest = json.dumps({"files": ["DSC_0001.NEF", "DSC_0002.NEF"]}).encode()

        resp = self._put(port, "/upload/realstack/manifest.json", manifest, token)
        assert resp.status == 200
        resp = self._put(port, "/upload/realstack/DSC_0001.NEF", b"REAL_NEF_BYTES_1", token)
        assert resp.status == 200
        resp = self._put(port, "/upload/realstack/DSC_0002.NEF", b"REAL_NEF_BYTES_2", token)
        assert resp.status == 200

        resp = self._post(port, "/complete/realstack", token)
        assert resp.status == 200

        final_dir = tmp_path / "realstack"
        assert final_dir.exists()
        assert (final_dir / "DSC_0001.NEF").read_bytes() == b"REAL_NEF_BYTES_1"
        assert (final_dir / "DSC_0002.NEF").read_bytes() == b"REAL_NEF_BYTES_2"
        assert not (tmp_path / ".incoming-realstack").exists()

    def test_complete_before_all_files_uploaded_fails_negative_twin(self, running_server, tmp_path):
        port, token = running_server
        manifest = json.dumps({"files": ["DSC_0001.NEF", "DSC_0002.NEF"]}).encode()
        self._put(port, "/upload/partial/manifest.json", manifest, token)
        self._put(port, "/upload/partial/DSC_0001.NEF", b"only one file", token)

        with pytest.raises(urllib.error.HTTPError) as exc_info:
            self._post(port, "/complete/partial", token)
        assert exc_info.value.code == 409
        assert not (tmp_path / "partial").exists()
        assert (tmp_path / ".incoming-partial").exists()  # still staged, nothing lost

    def test_unsafe_filename_rejected_over_real_http_negative_twin(self, running_server, tmp_path):
        port, token = running_server
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/upload/stack1/..%2F..%2Fevil.txt", data=b"x", method="PUT",
            headers={"Authorization": f"Bearer {token}"},
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req, timeout=5)
        assert exc_info.value.code == 400
