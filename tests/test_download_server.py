"""download_server.py over real HTTP: tailnet-identity auth, loopback-only guard, path rules, ranges, read-only."""
from __future__ import annotations

import base64
import http.client
import socket
import threading

import pytest

from bio_automation import download_server as ds
from bio_automation.download_server import IDENTITY_HEADER, DownloadServer, is_loopback, make_server, parse_range

LOGIN = "owner@example.com"
DATA = bytes(range(256)) * 4          # 1024 distinct-looking bytes
ID = {IDENTITY_HEADER: LOGIN}


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


def start(srv):
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


@pytest.fixture
def srv(folder):
    s = start(make_server(folder, 0, [LOGIN]))
    yield s
    s.shutdown(); s.server_close()


def req(srv, method, path, headers=None, host="127.0.0.1"):
    c = http.client.HTTPConnection(host, srv.server_address[1], timeout=10)
    c.request(method, path, headers=headers or {})
    r = c.getresponse()
    body = r.read()
    out = (r.status, {k.lower(): v for k, v in r.getheaders()}, body)
    c.close()
    return out


def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))          # no packet is sent: this only picks the outbound interface address
        return s.getsockname()[0]
    finally:
        s.close()


class TestIdentity:
    def test_no_identity_header_is_401(self, srv):
        status, _, body = req(srv, "GET", "/inat-3photos.zip")
        assert (status, body) == (401, b"authentication required: open this through your tailnet address\n")

    def test_other_tailnet_user_is_403(self, srv):
        assert req(srv, "GET", "/inat-3photos.zip", {IDENTITY_HEADER: "someone@example.com"})[0] == 403

    def test_allowed_login_gets_the_exact_bytes_and_headers(self, srv):
        status, h, body = req(srv, "GET", "/inat-3photos.zip", ID)
        assert (status, body) == (200, DATA)
        assert h["content-length"] == "1024"
        assert h["content-disposition"] == 'attachment; filename="inat-3photos.zip"'
        assert h["accept-ranges"] == "bytes" and h["content-type"] == "application/zip"

    def test_login_match_ignores_case_and_surrounding_whitespace(self, srv):
        assert req(srv, "GET", "/inat-3photos.zip", {IDENTITY_HEADER: "  Owner@Example.COM "})[0] == 200

    def test_every_allowed_login_works(self, folder):
        s = start(make_server(folder, 0, [LOGIN, "second@example.com"]))
        try:
            assert [req(s, "GET", "/inat-3photos.zip", {IDENTITY_HEADER: x})[0] for x in (LOGIN, "second@example.com", "third@example.com")] == [200, 200, 403]
        finally:
            s.shutdown(); s.server_close()

    def test_configured_login_case_does_not_matter_either(self, folder):
        s = start(make_server(folder, 0, ["Owner@Example.COM"]))                    # configured in mixed case, sent lower case
        try:
            assert req(s, "GET", "/inat-3photos.zip", {IDENTITY_HEADER: "owner@example.com"})[0] == 200
            assert s.logins == frozenset({"owner@example.com"})
        finally:
            s.shutdown(); s.server_close()

    def test_basic_auth_and_cookies_are_not_accepted(self, srv):
        basic = {"Authorization": "Basic " + base64.b64encode(b"owner:anything").decode(), "Cookie": "session=abc"}
        assert req(srv, "GET", "/inat-3photos.zip", basic)[0] == 401
        assert req(srv, "GET", "/inat-3photos.zip", {**basic, **ID})[0] == 200      # positive twin: the identity header is what counts


class TestLoopbackOnly:
    def test_is_loopback_boundaries(self):
        # '::ffff:127.0.0.1' needs our own IPv4-mapped unwrap on Python 3.12 (stdlib is_loopback is False there; 3.13+ handles it): requires-python is >=3.12
        assert [is_loopback(x) for x in ("127.0.0.1", "127.255.255.254", "::1", "::ffff:127.0.0.1", "128.0.0.1", "10.0.0.1", "192.168.1.1",
                                         "100.73.200.19", "::ffff:8.8.8.8")] == [True, True, True, True, False, False, False, False, False]

    def test_non_loopback_peer_is_403_even_with_a_valid_identity_header(self, folder):
        wide = start(DownloadServer(("0.0.0.0", 0), folder, [LOGIN]))               # built directly: make_server would refuse this bind
        try:
            ip = lan_ip()
            assert not is_loopback(ip), f"test machine has no non-loopback address ({ip})"
            status, _, body = req(wide, "GET", "/inat-3photos.zip", ID, host=ip)
            assert (status, body) == (403, b"forbidden: not the local tailscale proxy\n")
            assert req(wide, "GET", "/inat-3photos.zip", ID, host="127.0.0.1")[0] == 200     # twin: same request over loopback
        finally:
            wide.shutdown(); wide.server_close()

    @pytest.mark.parametrize("bind", ["0.0.0.0", "172.16.100.11", "100.73.200.19", "::"])
    def test_make_server_refuses_to_bind_anything_but_loopback(self, folder, bind):
        with pytest.raises(ValueError, match="refusing to bind .*: the identity header is only trustworthy behind the local tailscale proxy"):
            make_server(folder, 0, [LOGIN], bind=bind)

    def test_make_server_accepts_loopback_binds(self, folder):
        s = make_server(folder, 0, [LOGIN], bind="127.0.0.1")
        try:
            assert s.server_address[0] == "127.0.0.1"
        finally:
            s.server_close()

    @pytest.mark.parametrize("logins", [[], [""], ["  "], None])
    def test_no_allowed_login_means_no_server(self, folder, logins):
        with pytest.raises(ValueError, match="refusing to start without at least one allowed tailnet login"):
            make_server(folder, 0, logins)

    def test_missing_folder_cannot_start_a_server(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="download folder does not exist"):
            make_server(tmp_path / "nope", 0, [LOGIN])


class TestPaths:
    @pytest.mark.parametrize("path", ["/secret.txt", "/sub/nested.zip", "/sub", "/..%2f..%2fsecret.txt", "/../secret.txt",
                                      "/%2e%2e/secret.txt", "/missing.zip", "/inat-3photos.zip/extra", "/.hidden.zip"])
    def test_anything_but_a_top_level_zip_is_404(self, srv, path):
        assert req(srv, "GET", path, ID)[0] == 404

    def test_the_sha256_sidecar_is_served(self, srv):
        status, h, body = req(srv, "GET", "/inat-3photos.zip.sha256", ID)
        assert (status, body) == (200, ("ab" * 32 + "  inat-3photos.zip\n").encode())
        assert h["content-type"] == "text/plain; charset=utf-8"

    def test_query_string_is_ignored(self, srv):
        assert req(srv, "GET", "/inat-3photos.zip?x=1", ID)[2] == DATA

    def test_index_lists_only_zips_with_size_and_sha(self, srv):
        status, _, body = req(srv, "GET", "/", ID)
        assert (status, body.decode()) == (200, "size_bytes\tsha256\tname\n1024\t" + "ab" * 32 + "\tinat-3photos.zip\n")

    def test_index_with_no_zips_is_just_the_header(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        s = start(make_server(d, 0, [LOGIN]))
        try:
            assert req(s, "GET", "/", ID)[2] == b"size_bytes\tsha256\tname\n"
        finally:
            s.shutdown(); s.server_close()


class TestRanges:
    @pytest.mark.parametrize("header,start_,end", [("bytes=0-9", 0, 9), ("bytes=1000-", 1000, 1023), ("bytes=-4", 1020, 1023),
                                                    ("bytes=10-5000", 10, 1023), ("bytes=-5000", 0, 1023)])
    def test_single_ranges_return_exact_slices(self, srv, header, start_, end):
        status, h, body = req(srv, "GET", "/inat-3photos.zip", {**ID, "Range": header})
        assert status == 206
        assert body == DATA[start_:end + 1]
        assert h["content-range"] == f"bytes {start_}-{end}/1024"
        assert h["content-length"] == str(end - start_ + 1)

    @pytest.mark.parametrize("header", ["bytes=1024-", "bytes=5000-6000", "bytes=-0", "bytes=9-3"])
    def test_unsatisfiable_ranges_are_416_with_the_size(self, srv, header):
        status, h, _ = req(srv, "GET", "/inat-3photos.zip", {**ID, "Range": header})
        assert (status, h["content-range"]) == (416, "bytes */1024")

    def test_unsupported_range_forms_fall_back_to_the_whole_file(self, srv):
        status, _, body = req(srv, "GET", "/inat-3photos.zip", {**ID, "Range": "bytes=0-1,5-6"})
        assert (status, body) == (200, DATA)

    def test_parse_range_is_pure(self):
        assert parse_range("bytes=2-5", 100) == (2, 5) and parse_range("", 100) is None and parse_range("bytes=-", 100) is None


class TestReadOnly:
    def test_head_has_headers_and_no_body(self, srv):
        status, h, body = req(srv, "HEAD", "/inat-3photos.zip", ID)
        assert (status, h["content-length"], body) == (200, "1024", b"")

    @pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "PATCH"])
    def test_write_methods_are_405_and_change_nothing(self, srv, folder, method):
        status, h, _ = req(srv, method, "/inat-3photos.zip", ID)
        assert (status, h["allow"]) == (405, "GET, HEAD")
        assert (folder / "inat-3photos.zip").read_bytes() == DATA
