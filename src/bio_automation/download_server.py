"""Read-only HTTP download endpoint for the flat zips built by ``bio export-zip``, meant to sit behind ``tailscale serve``.

Authentication is the tailnet, not a password: ``tailscale serve`` terminates TLS, authenticates the calling device through WireGuard, strips any
client-supplied ``Tailscale-User-Login`` header and sets it to the real login of the caller. This server therefore
  * listens on loopback ONLY (it refuses to bind anything else, because the header is only trustworthy behind the local proxy),
  * accepts a connection only from a loopback peer (the proxy), and
  * serves a request only when ``Tailscale-User-Login`` is one of the allowed logins (nothing else is accepted: no cookies, no Basic auth).
Anyone who can already run code on this machine can forge the header from loopback; that is inside the machine's trust boundary.

Scope is deliberately tiny: ``*.zip`` and ``*.zip.sha256`` files directly in ONE folder, nothing else (no other files, no sub-folders, no uploads).
File names must match a strict pattern (no ``..``, no slashes). Single-range ``Range`` requests let a 1 GB+ download resume. GET/HEAD only.
"""
from __future__ import annotations

import ipaddress
import re
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

IDENTITY_HEADER = "Tailscale-User-Login"
_SERVABLE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.zip(\.sha256)?$")
_RANGE = re.compile(r"^bytes=(\d*)-(\d*)$")
_CHUNK = 1 << 20


def is_loopback(ip: str) -> bool:
    """True for 127.0.0.0/8, ::1 and IPv4-mapped loopback (::ffff:127.x)."""
    addr = ipaddress.ip_address(ip)
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped:
        addr = addr.ipv4_mapped
    return addr.is_loopback


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
    server_version = "bio-downloads/2"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):            # one line per request on stderr, with the authenticated login when there is one
        who = self.headers.get(IDENTITY_HEADER, "-")
        sys.stderr.write("%s %s %s\n" % (self.client_address[0], who, fmt % args))

    def _deny(self, status: HTTPStatus, body: bytes = b"", extra: dict | None = None):
        self.send_response(status)
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        if body and self.command != "HEAD":
            self.wfile.write(body)

    def _serve(self):
        if not is_loopback(self.client_address[0]):
            return self._deny(HTTPStatus.FORBIDDEN, b"forbidden: not the local tailscale proxy\n")
        login = self.headers.get(IDENTITY_HEADER, "").strip().lower()
        if not login:
            return self._deny(HTTPStatus.UNAUTHORIZED, b"authentication required: open this through your tailnet address\n")
        if login not in self.server.logins:
            return self._deny(HTTPStatus.FORBIDDEN, b"forbidden: this tailnet user is not allowed\n")
        path = self.path.split("?", 1)[0]
        if path == "/":
            return self._index()
        name = path[1:]
        if not _SERVABLE.match(name):
            return self._deny(HTTPStatus.NOT_FOUND, b"not found\n")
        f = self.server.directory / name
        if not f.is_file():
            return self._deny(HTTPStatus.NOT_FOUND, b"not found\n")
        size = f.stat().st_size
        try:
            rng = parse_range(self.headers.get("Range", ""), size)
        except ValueError:
            return self._deny(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, extra={"Content-Range": f"bytes */{size}"})
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
        self._deny(HTTPStatus.METHOD_NOT_ALLOWED, b"read-only\n", {"Allow": "GET, HEAD"})

    do_POST = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = _no


class DownloadServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, directory: Path, logins):
        super().__init__(addr, _Handler)
        self.directory = Path(directory)
        self.logins = frozenset(x.strip().lower() for x in logins)


def make_server(directory: Path, port: int, logins, bind: str = "127.0.0.1") -> DownloadServer:
    """A loopback-only server. Refuses a non-loopback *bind* and an empty login list."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"download folder does not exist: {directory}")
    logins = [x for x in (logins or []) if x and x.strip()]
    if not logins:
        raise ValueError("refusing to start without at least one allowed tailnet login")
    if not is_loopback(bind):
        raise ValueError(f"refusing to bind {bind}: the identity header is only trustworthy behind the local tailscale proxy, so this server listens on loopback only")
    return DownloadServer((bind, port), directory, logins)
