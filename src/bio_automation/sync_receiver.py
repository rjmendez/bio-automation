"""sync_receiver.py -- a narrow, loopback-by-default HTTP service that
accepts focus-stack uploads from the NikonStacker Pi and lands them in
D:\\nikon_stack_sync for stack_sync.py to pick up.

Deliberately replaces an earlier SSH-key-based design: that would have
given the Pi standing shell access to this machine to run rsync + a
remote rename command, far broader than this feature needs. This service
exposes exactly two capabilities -- upload one file into a staging area,
and atomically complete a stack once every declared file has arrived --
nothing else. A leaked token lets someone write files into one staging
directory; it cannot run a command here.

Also deliberately shaped for an easy later swap to an S3-backed queue (per
the owner's stated long-term plan): the two operations below (PUT one
object under a key, then a "finalize" call) map directly onto an S3
multipart-upload-then-complete flow, so migrating the transport later is a
backend swap behind the same two verbs, not a redesign.
"""
from __future__ import annotations

import json
import re
import shutil
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import settings

SYNC_DIR = settings.SYNC_DIR
TOKEN_FILE = Path.home() / ".nikon_sync_token"

# Filenames and stack ids both come from the network -- only allow the
# characters a real manifest/filename would ever contain, so nothing can
# path-traverse out of the intended staging directory.
_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


class UnsafeNameError(ValueError):
    """A stack_id or filename from the network failed the safe-name check."""


class AuthError(PermissionError):
    """Missing or wrong bearer token."""


class IncompleteStackError(RuntimeError):
    """complete_stack() was called before every manifest-declared file arrived."""


def load_token(token_file: Path = TOKEN_FILE) -> str:
    return token_file.read_text().strip()


def check_safe_name(name: str) -> None:
    # The character-class regex alone lets "." and ".." through (both are
    # made entirely of allowed characters), but as a COMPLETE name either
    # one IS the traversal token -- "d / '..'" escapes to d's parent. Must
    # reject them explicitly, not just rely on excluding "/" and "\".
    if name in (".", "..") or not _SAFE_NAME.match(name):
        raise UnsafeNameError(f"unsafe name: {name!r}")


def staging_dir(stack_id: str, sync_dir: Path = SYNC_DIR) -> Path:
    check_safe_name(stack_id)
    return sync_dir / f".incoming-{stack_id}"


def save_upload(stack_id: str, filename: str, data: bytes, *,
                 sync_dir: Path = SYNC_DIR) -> Path:
    """Write *data* into the stack's staging directory under *filename*.
    Only ever writes inside staging_dir(stack_id) -- check_safe_name()
    rejects anything that could escape it (path separators, "..", etc.)."""
    check_safe_name(filename)
    d = staging_dir(stack_id, sync_dir)
    d.mkdir(parents=True, exist_ok=True)
    dest = d / filename
    dest.write_bytes(data)
    return dest


def complete_stack(stack_id: str, *, sync_dir: Path = SYNC_DIR) -> Path:
    """Atomically move a stack from its hidden staging name to its real,
    visible name -- only once every file its own manifest.json declares is
    actually present in staging. A watcher polling *sync_dir* therefore
    never sees a stack under its real name until it is genuinely complete.

    Raises IncompleteStackError (not a silent partial landing) if
    manifest.json is missing or any declared file hasn't arrived yet.
    """
    check_safe_name(stack_id)
    staging = staging_dir(stack_id, sync_dir)
    manifest_path = staging / "manifest.json"
    if not manifest_path.exists():
        raise IncompleteStackError(f"{stack_id}: no manifest.json received yet")

    manifest = json.loads(manifest_path.read_text())
    missing = [f for f in manifest["files"] if not (staging / f).exists()]
    if missing:
        raise IncompleteStackError(f"{stack_id}: missing files {missing}")

    final_dir = sync_dir / stack_id
    staging.rename(final_dir)
    return final_dir


class SyncReceiverHandler(BaseHTTPRequestHandler):
    sync_dir: Path = SYNC_DIR
    token: str = ""

    def _authorized(self) -> bool:
        auth = self.headers.get("Authorization", "")
        return auth == f"Bearer {self.token}"

    def _reject(self, status: int, message: str) -> None:
        body = message.encode()
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_PUT(self):
        if not self._authorized():
            return self._reject(HTTPStatus.UNAUTHORIZED, "bad token")
        parts = self.path.strip("/").split("/")
        if len(parts) != 3 or parts[0] != "upload":
            return self._reject(HTTPStatus.NOT_FOUND, "expected /upload/<stack_id>/<filename>")
        _, stack_id, filename = parts
        length = int(self.headers.get("Content-Length", 0))
        data = self.rfile.read(length)
        try:
            save_upload(stack_id, filename, data, sync_dir=self.sync_dir)
        except UnsafeNameError as e:
            return self._reject(HTTPStatus.BAD_REQUEST, str(e))
        self._reject(HTTPStatus.OK, "ok")

    def do_POST(self):
        if not self._authorized():
            return self._reject(HTTPStatus.UNAUTHORIZED, "bad token")
        parts = self.path.strip("/").split("/")
        if len(parts) != 2 or parts[0] != "complete":
            return self._reject(HTTPStatus.NOT_FOUND, "expected /complete/<stack_id>")
        _, stack_id = parts
        try:
            complete_stack(stack_id, sync_dir=self.sync_dir)
        except (UnsafeNameError, IncompleteStackError) as e:
            return self._reject(HTTPStatus.CONFLICT, str(e))
        self._reject(HTTPStatus.OK, "ok")

    def log_message(self, fmt, *args):
        pass  # quiet by default; real logging added when this moves off loopback


def serve(host: str = "127.0.0.1", port: int = 8799, sync_dir: Path = SYNC_DIR,
          token_file: Path = TOKEN_FILE) -> None:
    sync_dir.mkdir(parents=True, exist_ok=True)
    SyncReceiverHandler.sync_dir = sync_dir
    SyncReceiverHandler.token = load_token(token_file)
    server = ThreadingHTTPServer((host, port), SyncReceiverHandler)
    server.serve_forever()


if __name__ == "__main__":
    serve()
