"""download_server.py over real HTTP on an ephemeral 127.0.0.1 port: auth, allowlist, path rules, ranges, read-only."""
from __future__ import annotations

import base64
import http.client
import threading

import pytest

from bio_automation import download_server as ds
from bio_automation.download_server import is_allowed, load_or_create_password, make_server, parse_range

PW = "correct horse"
DATA = bytes(range(256)) * 4          # 1024 distinct-looking bytes
AUTH = {"Authorization": "Basic " + base64.b64encode(f"owner:{PW}".encode()).decode()}


@pytest.fixture(autouse=True)
def _no_failure_delay(monkeypatch):
    monkeypatch.setattr(ds, "_FAIL_DELAY_S", 0)


@pytest.fixture
def folder(tmp_path):
    d = tmp_path / "dl"
    d.mkdir()
    (d / "inat-3photos.zip").write_bytes(DATA)
    (d / "inat-3photos.zip.sha256").write_bytes(("ab" * 32 + "  inat-3photos.zip\n").encode("ascii"))
    (d / "secret.txt").write_text("not a zip", encoding="utf-8")
    (d / "sub").mkdir()
    (d / "sub" / "nested.zip").write_bytes(b"nested")
    return d


def serve(folder, allow=ds.DEFAULT_ALLOW, password=PW):
    srv = make_server(folder, "127.0.0.1", 0, password, allow=allow)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


@pytest.fixture
def srv(folder):
    s = serve(folder)
    yield s
    s.shutdown(); s.server_close()


def req(srv, method, path, headers=None):
    c = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=10)
    c.request(method, path, headers=headers or {})
    r = c.getresponse()
    body = r.read()
    out = (r.status, dict((k.lower(), v) for k, v in r.getheaders()), body)
    c.close()
    return out


class TestAuth:
    def test_no_credentials_is_401_with_a_basic_challenge(self, srv):
        status, h, _ = req(srv, "GET", "/inat-3photos.zip")
        assert (status, h["www-authenticate"]) == (401, 'Basic realm="bio-downloads"')

    def test_wrong_password_is_401(self, srv):
        bad = {"Authorization": "Basic " + base64.b64encode(b"owner:wrong").decode()}
        assert req(srv, "GET", "/inat-3photos.zip", bad)[0] == 401

    def test_wrong_user_is_401(self, srv):
        bad = {"Authorization": "Basic " + base64.b64encode(f"admin:{PW}".encode()).decode()}
        assert req(srv, "GET", "/inat-3photos.zip", bad)[0] == 401

    def test_garbage_authorization_header_is_401(self, srv):
        assert req(srv, "GET", "/inat-3photos.zip", {"Authorization": "Basic !!!not-base64!!!"})[0] == 401

    def test_correct_credentials_get_the_exact_bytes(self, srv):
        status, h, body = req(srv, "GET", "/inat-3photos.zip", AUTH)
        assert (status, body) == (200, DATA)
        assert h["content-length"] == "1024"
        assert h["content-disposition"] == 'attachment; filename="inat-3photos.zip"'
        assert h["accept-ranges"] == "bytes"
        assert h["content-type"] == "application/zip"


class TestAllowlist:
    def test_is_allowed_boundaries(self):
        cidrs = ds.DEFAULT_ALLOW
        assert [is_allowed(ip, cidrs) for ip in
                ("127.0.0.1", "10.1.2.3", "172.15.255.255", "172.16.0.1", "172.31.255.255", "172.32.0.0", "192.168.1.1",
                 "100.64.0.1", "100.127.255.255", "100.128.0.1", "8.8.8.8", "::1", "::ffff:192.168.1.5", "::ffff:8.8.8.8")] == \
               [True, True, False, True, True, False, True, True, True, False, False, False, True, False]

    def test_client_outside_the_allowlist_is_403_even_with_the_right_password(self, folder):
        s = serve(folder, allow=("10.0.0.0/8",))
        try:
            assert req(s, "GET", "/inat-3photos.zip", AUTH)[0] == 403
        finally:
            s.shutdown(); s.server_close()

    def test_client_inside_the_allowlist_is_served(self, folder):    # positive twin of the 403 test: same request, loopback allowed
        s = serve(folder, allow=("127.0.0.0/8",))
        try:
            assert req(s, "GET", "/inat-3photos.zip", AUTH)[0] == 200
        finally:
            s.shutdown(); s.server_close()

    def test_a_typo_in_the_allowlist_fails_at_start_up(self, folder):
        with pytest.raises(ValueError, match="does not appear to be an IPv4 or IPv6 network"):
            make_server(folder, "127.0.0.1", 0, PW, allow=("not-a-cidr",))


class TestPaths:
    @pytest.mark.parametrize("path", ["/secret.txt", "/sub/nested.zip", "/sub", "/..%2f..%2fsecret.txt", "/../secret.txt",
                                      "/%2e%2e/secret.txt", "/missing.zip", "/inat-3photos.zip/extra", "/.hidden.zip"])
    def test_anything_but_a_top_level_zip_is_404(self, srv, path):
        assert req(srv, "GET", path, AUTH)[0] == 404

    def test_the_sha256_sidecar_is_served(self, srv):
        status, h, body = req(srv, "GET", "/inat-3photos.zip.sha256", AUTH)
        assert (status, body) == (200, ("ab" * 32 + "  inat-3photos.zip\n").encode())
        assert h["content-type"] == "text/plain; charset=utf-8"

    def test_query_string_is_ignored(self, srv):
        assert req(srv, "GET", "/inat-3photos.zip?x=1", AUTH)[2] == DATA

    def test_index_lists_only_zips_with_size_and_sha(self, srv):
        status, _, body = req(srv, "GET", "/", AUTH)
        assert (status, body.decode()) == (200, "size_bytes\tsha256\tname\n1024\t" + "ab" * 32 + "\tinat-3photos.zip\n")

    def test_index_with_no_zips_is_just_the_header(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        s = serve(d)
        try:
            assert req(s, "GET", "/", AUTH)[2] == b"size_bytes\tsha256\tname\n"
        finally:
            s.shutdown(); s.server_close()


class TestRanges:
    @pytest.mark.parametrize("header,start,end", [("bytes=0-9", 0, 9), ("bytes=1000-", 1000, 1023), ("bytes=-4", 1020, 1023),
                                                   ("bytes=10-5000", 10, 1023), ("bytes=-5000", 0, 1023)])
    def test_single_ranges_return_exact_slices(self, srv, header, start, end):
        status, h, body = req(srv, "GET", "/inat-3photos.zip", {**AUTH, "Range": header})
        assert status == 206
        assert body == DATA[start:end + 1]
        assert h["content-range"] == f"bytes {start}-{end}/1024"
        assert h["content-length"] == str(end - start + 1)

    @pytest.mark.parametrize("header", ["bytes=1024-", "bytes=5000-6000", "bytes=-0", "bytes=9-3"])
    def test_unsatisfiable_ranges_are_416_with_the_size(self, srv, header):
        status, h, _ = req(srv, "GET", "/inat-3photos.zip", {**AUTH, "Range": header})
        assert (status, h["content-range"]) == (416, "bytes */1024")

    def test_unsupported_range_forms_fall_back_to_the_whole_file(self, srv):
        assert req(srv, "GET", "/inat-3photos.zip", {**AUTH, "Range": "bytes=0-1,5-6"})[1:] [1] == DATA

    def test_parse_range_is_pure(self):
        assert parse_range("bytes=2-5", 100) == (2, 5) and parse_range("", 100) is None and parse_range("bytes=-", 100) is None


class TestReadOnly:
    def test_head_has_headers_and_no_body(self, srv):
        status, h, body = req(srv, "HEAD", "/inat-3photos.zip", AUTH)
        assert (status, h["content-length"], body) == (200, "1024", b"")

    @pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "PATCH"])
    def test_write_methods_are_405_and_change_nothing(self, srv, folder, method):
        status, h, _ = req(srv, method, "/inat-3photos.zip", AUTH)
        assert (status, h["allow"]) == (405, "GET, HEAD")
        assert (folder / "inat-3photos.zip").read_bytes() == DATA


class TestPasswordFile:
    def test_created_once_then_reused(self, tmp_path):
        f = tmp_path / "sub" / "pw.txt"
        first = load_or_create_password(f)
        assert len(first) >= 32 and f.read_text(encoding="utf-8").strip() == first
        assert load_or_create_password(f) == first

    def test_empty_password_file_is_refused(self, tmp_path):
        f = tmp_path / "pw.txt"
        f.write_text("  \n", encoding="utf-8")
        with pytest.raises(ValueError, match="is empty: refusing to serve with an empty password"):
            load_or_create_password(f)

    def test_empty_password_cannot_start_a_server(self, folder):
        with pytest.raises(ValueError, match="refusing to serve with an empty password"):
            make_server(folder, "127.0.0.1", 0, "")

    def test_missing_folder_cannot_start_a_server(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="download folder does not exist"):
            make_server(tmp_path / "nope", "127.0.0.1", 0, PW)
