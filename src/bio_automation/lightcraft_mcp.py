"""A headless LightCraft engine on the persistent library (settings.LIGHTCRAFT_LIBRARY), driven over its MCP stdio protocol.

Every render/import script goes through open_library(), which refuses to work on the wrong catalog: a Claude session's own `lightcraft`
MCP server can be a non-persistent scratch catalog (ids from 1, gone on restart), and a desktop app holding the same library open would be a
second writer. Both are checked before the first command runs.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path

from . import settings


class LightCraftLibraryError(RuntimeError):
    """The engine is not on the persistent library this pipeline expects, or another writer may have it open."""


def check_library_info(info: dict, expected: Path) -> None:
    """Raise unless library.info describes the persistent catalog at `expected`."""
    if not isinstance(info, dict):
        raise LightCraftLibraryError(f"library.info returned {info!r}, not a library description")
    if info.get("persistent") is not True:
        raise LightCraftLibraryError(f"library at {info.get('path')!r} is not persistent: edits and imports would be lost")
    actual = info.get("path")
    if actual is None or os.path.normcase(os.path.normpath(actual)) != os.path.normcase(os.path.normpath(str(expected))):
        raise LightCraftLibraryError(f"engine opened library {actual!r}, expected {str(expected)!r}")
    damaged = (info.get("load") or {}).get("damaged")
    if damaged:
        raise LightCraftLibraryError(f"library at {actual!r} loaded damaged: {damaged!r}")


def gui_listening(address: str = settings.LIGHTCRAFT_GUI, timeout: float = 0.5) -> bool:
    host, _, port = address.rpartition(":")
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def _text(result: dict):
    for c in result.get("content", []):
        if c.get("type") == "text":
            try:
                return json.loads(c["text"])
            except ValueError:
                return c["text"]
    return None


class LightCraftLibrary:
    """One headless engine process. call(tool, args) returns the tool's decoded text result; close() ends the process."""

    def __init__(self, library: Path = settings.LIGHTCRAFT_LIBRARY, cli: Path = settings.LIGHTCRAFT_CLI, *, allow_gui: bool = False,
                 gpu: bool = settings.LIGHTCRAFT_GPU):
        if not allow_gui and gui_listening():
            raise LightCraftLibraryError(f"the LightCraft app is listening on {settings.LIGHTCRAFT_GUI}; a headless engine on the same library "
                                         "would be a second writer (close the app, or pass allow_gui=True if it has another library open)")
        self.library = Path(library)
        self._p = subprocess.Popen([str(cli), "mcp", "--library", str(self.library)], env=dict(os.environ, LIGHTCRAFT_GPU="1" if gpu else "0"),
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding="utf-8", bufsize=1)
        self._n = 1
        self._send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                    "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "bio-automation", "version": "1"}}})
        self._recv(1)
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        self.info = self.call("cmd_library_info", {})
        check_library_info(self.info, self.library)

    def _send(self, obj: dict) -> None:
        self._p.stdin.write(json.dumps(obj) + "\n")
        self._p.stdin.flush()

    def _recv(self, msg_id: int) -> dict:
        while True:
            line = self._p.stdout.readline()
            if not line:
                raise LightCraftLibraryError(f"lightcraft engine exited (code {self._p.poll()}) before answering request {msg_id}")
            d = json.loads(line)
            if d.get("id") == msg_id:
                return d

    def call(self, tool: str, args: dict):
        self._n += 1
        self._send({"jsonrpc": "2.0", "id": self._n, "method": "tools/call", "params": {"name": tool, "arguments": args}})
        r = self._recv(self._n)
        if "error" in r:
            raise LightCraftLibraryError(f"{tool}: {r['error']}")
        return _text(r.get("result", {}))

    def close(self) -> None:
        if self._p.poll() is None:
            self._p.stdin.close()
            try:
                self._p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self._p.kill()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def has_live_edits(inspect: dict) -> bool:
    """True when the photo carries edits right now. LightCraft's 'edited' field is only a last-edit timestamp and stays set after a Reset, so
    live edits = the history is not empty and does not end in 'Reset'."""
    hist = (inspect or {}).get("history") or []
    return bool(hist) and hist[-1] != "Reset"


EXPORT_SETTINGS = {"format": "jpeg", "quality": 95, "longEdge": 0, "colorSpace": "srgb", "metadata": "all", "removeLocation": True,
                   "conflict": "overwrite"}     # the Stock render: full size, q95, sRGB, metadata kept, location removed


def render_retouched(lib, photo_id: int, out: Path, *, wait_s: float = 420, poll_s: float = 3, sleep=None, clock=None) -> dict:
    """Export `photo_id` retouched to `out`: the owner's own edits as they are, else develop.auto (reset afterwards, so the library keeps
    no edit this made). Returns {'owner_edited', 'iso'}; raises LightCraftLibraryError if no non-empty file appears within wait_s
    (a headless export blocks until the file is written; the poll is only a fallback)."""
    import time
    sleep, clock = sleep or time.sleep, clock or time.monotonic
    ins = lib.call("cmd_photo_inspect", {"id": photo_id}) or {}
    owner_edited = has_live_edits(ins)
    touched = False
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not owner_edited:
            lib.call("select_photos", {"ids": [photo_id], "active": photo_id})
            lib.call("command_run", {"id": "develop.auto", "params": {}})
            touched = True
        res = lib.call("export", {"id": photo_id, "path": str(out), **EXPORT_SETTINGS})
        t0 = clock()
        while not (out.is_file() and out.stat().st_size > 0) and clock() - t0 < wait_s:   # headless exports block: normally already written
            sleep(poll_s)
        if not (out.is_file() and out.stat().st_size > 0):
            raise LightCraftLibraryError(f"photo {photo_id}: no export at {out} after {wait_s:.0f} s ({str(res)[:160]})")
    finally:
        if touched:
            lib.call("command_run", {"id": "develop.reset", "params": {"ids": [photo_id]}})
    return {"owner_edited": owner_edited, "iso": ((ins.get("meta") or {}).get("iso") if isinstance(ins, dict) else None) or 0}


@dataclass(frozen=True)
class Imported:
    ids: dict            # {path: photo id} for every path asked for
    duplicates: dict     # {path: library path of the photo it duplicates} -- LightCraft matches by content, so that path can differ


def import_files(lib: LightCraftLibrary, paths: list[str]) -> Imported:
    """Reference `paths` in place (mode add); every path gets an id, new or the existing photo LightCraft says it duplicates.
    A path that failed, or whose duplicate is only in Recently Deleted, raises: a frame without an id cannot be rendered."""
    r = lib.call("import", {"paths": list(paths), "mode": "add"})
    ids = list(r.get("imported", []))
    dups = {d["path"]: d for d in r.get("duplicates", [])}
    failed = {(f["path"] if isinstance(f, dict) else f): f for f in r.get("failed", [])}
    if failed:
        raise LightCraftLibraryError(f"{len(failed)} files failed to import: {list(failed.values())[:3]}")
    new_paths = [p for p in paths if p not in dups]
    if len(new_paths) != len(ids):
        raise LightCraftLibraryError(f"import returned {len(ids)} ids for {len(new_paths)} new files: cannot map ids to paths")
    out = dict(zip(new_paths, ids))
    for p, d in dups.items():
        if d.get("existingDeleted") or d.get("existing") is None:
            raise LightCraftLibraryError(f"{p} is in the library only as a deleted photo ({d})")
        out[p] = int(d["existing"])
    return Imported(ids=out, duplicates={p: d.get("existingPath") or d.get("reason", "") for p, d in dups.items()})
