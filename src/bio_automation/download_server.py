"""Read-only HTTP download endpoint for the flat zips built by ``bio export-zip``.

Scope is deliberately tiny: it serves ``*.zip`` and ``*.zip.sha256`` files that sit directly in ONE folder, and nothing else (no listing of
other files, no sub-folders, no uploads, no CGI). Guardrails:
  * HTTP Basic auth, constant-time compare, password read from a file the operator controls (created on first start, 24 random bytes);
  * a source-address allowlist (default: loopback, RFC1918 private ranges and the Tailscale 100.64.0.0/10 range), checked BEFORE auth;
  * file names must match a strict pattern, so ``..``, slashes and anything that is not a zip cannot be requested;
  * single-range ``Range`` requests, so a 1 GB+ download can resume after a dropped Wi-Fi link;
  * binds to the address you give it (default 127.0.0.1): LAN exposure is an explicit ``--bind``.
Not provided: TLS. Basic auth over plain HTTP is readable by anyone who can sniff the LAN segment; put it behind a TLS front door if that matters.
"""
from __future__ import annotations

import base64
import hmac
import ipaddress
import re
import secrets
import sys
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DEFAULT_ALLOW = ("127.0.0.0/8", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "100.64.0.0/10")
_SERVABLE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.zip(\.sha256)?$")
_RANGE = re.compile(r"^bytes=(\d*)-(\d*)$")
_CHUNK = 1 << 20
_FAIL_DELAY_S = 0.5


def is_allowed(ip: str, cidrs) -> bool:
    """True when *ip* falls inside one of *cidrs* (IPv4-mapped IPv6 addresses are unwrapped first)."""
    addr = ipaddress.ip_address(ip)
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped:
        addr = addr.ipv4_mapped
    return any(addr in ipaddress.ip_network(c, strict=False) for c in cidrs)


def load_or_create_password(path: Path) -> str:
    """The password stored in *path*; created with 24 random bytes (urlsafe) when the file does not exist. An empty file is an error."""
    path = Path(path)
    if path.exists():
        pw = path.read_text(encoding="utf-8").strip()
        if not pw:
            raise ValueError(f"{path} is empty: refusing to serve with an empty password")
        return pw
    path.parent.mkdir(parents=True, exist_ok=True)
    pw = secrets.token_urlsafe(24)
    path.write_text(pw + "\n", encoding="utf-8")
    return pw


def _sha_of(zip_path: Path) -> str:
    side = zip_path.with_name(zip_path.name + ".sha256")
    try:
        return side.read_text(encoding="ascii").split()[0]
    except (OSError, IndexError):
        return "-"


def parse_range(header: str, size: int):
    """(start, end) inclusive for a single ``bytes=`` range, or None for no/unsupported header. Raises ValueError when unsatisfiable."""
    m = _RANGE.match(header.strip()) if header else None
    if not m or (m.group(1) == "" and m.group(2) == ""):
        return None
    a, b = m.groups()
    if a == "":                       # suffix: the last b bytes
        n = int(b)
        if n == 0:
            raise ValueError("empty suffix range")
        return max(size - n, 0), size - 1
    start = int(a)
    end = int(b) if b != "" else size - 1
    if start >= size or end < start:
        raise ValueError("range not satisfiable")
    return start, min(end, size - 1)


class _Handler(BaseHTTPRequestHandler):
    server_version = "bio-downloads/1"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):            # never log the Authorization header; one line per request on stderr
        sys.stderr.write("%s %s\n" % (self.client_address[0], fmt % args))

    def _deny(self, status: HTTPStatus, extra: dict | None = None, body: bytes = b""):
        self.send_response(status)
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        if body and self.command != "HEAD":
            self.wfile.write(body)

    def _authorised(self) -> bool:
        hdr = self.headers.get("Authorization", "")
        if not hdr.startswith("Basic "):
            return False
        try:
            given = base64.b64decode(hdr[6:], validate=True)
        except Exception:
            return False
        want = f"{self.server.user}:{self.server.password}".encode()
        return hmac.compare_digest(given, want)

    def _serve(self):
        if not is_allowed(self.client_address[0], self.server.allow):
            return self._deny(HTTPStatus.FORBIDDEN, body=b"forbidden\n")
        if not self._authorised():
            time.sleep(_FAIL_DELAY_S)
            return self._deny(HTTPStatus.UNAUTHORIZED, {"WWW-Authenticate": 'Basic realm="bio-downloads"'}, b"authentication required\n")
        path = self.path.split("?", 1)[0]
        if path == "/":
            return self._index()
        name = path[1:]
        if not _SERVABLE.match(name):
            return self._deny(HTTPStatus.NOT_FOUND, body=b"not found\n")
        f = self.server.directory / name
        if not f.is_file():
            return self._deny(HTTPStatus.NOT_FOUND, body=b"not found\n")
        size = f.stat().st_size
        try:
            rng = parse_range(self.headers.get("Range", ""), size)
        except ValueError:
            return self._deny(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, {"Content-Range": f"bytes */{size}"})
        start, end = rng if rng else (0, size - 1)
        length = end - start + 1 if size else 0
        self.send_response(HTTPStatus.PARTIAL_CONTENT if rng else HTTPStatus.OK)
        self.send_header("Content-Type", "application/zip" if name.endswith(".zip") else "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Disposition", f'attachment; filename="{name}"')
        if rng:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if self.command == "HEAD" or not size:
            return
        try:
            with open(f, "rb") as fh:
                fh.seek(start)
                left = length
                while left > 0:
                    block = fh.read(min(_CHUNK, left))
                    if not block:
                        break
                    self.wfile.write(block)
                    left -= len(block)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def _index(self):
        rows = []
        for p in sorted(self.server.directory.iterdir(), key=lambda q: q.name.lower()):
            if p.is_file() and _SERVABLE.match(p.name) and p.name.endswith(".zip"):
                rows.append(f"{p.stat().st_size}\t{_sha_of(p)}\t{p.name}")
        body = ("size_bytes\tsha256\tname\n" + "\n".join(rows) + ("\n" if rows else "")).encode()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    do_GET = do_HEAD = _serve

    def _no(self):
        self._deny(HTTPStatus.METHOD_NOT_ALLOWED, {"Allow": "GET, HEAD"}, b"read-only\n")

    do_POST = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = _no


class DownloadServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, directory: Path, password: str, allow, user: str):
        super().__init__(addr, _Handler)
        self.directory = Path(directory)
        self.password = password
        self.allow = tuple(allow)
        self.user = user


def make_server(directory: Path, bind: str, port: int, password: str, allow=DEFAULT_ALLOW, user: str = "owner") -> DownloadServer:
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"download folder does not exist: {directory}")
    if not password:
        raise ValueError("refusing to serve with an empty password")
    for c in allow:
        ipaddress.ip_network(c, strict=False)          # fail at start-up on a typo, not at the first request
    return DownloadServer((bind, port), directory, password, allow, user)
