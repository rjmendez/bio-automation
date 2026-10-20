"""Local web console: sort (keep/reject), non-destructively crop, and cull
raws across the whole library -- every observation folder under a base
dir, not just one engine's queue.

This is a browser frontend for existing, already-reviewed mechanics, not a
new code path: "keep"/"reject" call pipeline.set_asset_status() exactly the
way `bio keep`/`bio reject` do, "crop" calls pipeline.set_asset_crop()
(applied later by render.py's export functions -- the original raw in
source/raw/ is never touched), and "cull" calls
pipeline.delete_rejected_raws() exactly the way `bio cull` does. This
module only ever decides WHICH folder/files a call targets; the actual
status-change/crop/delete logic lives in pipeline.py and is reused as-is.

Loopback only (127.0.0.1) by default, same as pipeline/stages/inat_console.py
-- reached from outside through a reverse proxy (Caddy's
photos.lab.floppydicks.net site) that restricts by source IP, not by this
process binding a public interface itself.
"""
from __future__ import annotations

import json
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .exif import extract_preview_bytes
from .lightcraft_cache import build_folder_cache, cache_path_for
from .models import Asset
from .pipeline import (
    delete_rejected_raws,
    discover_observation_folders,
    scan_assets,
    set_asset_crop,
    validate_crop,
    set_asset_status,
)

DEFAULT_PORT = 8780


def _rel_folder(base_dir: Path, folder: Path) -> str:
    return str(folder.relative_to(base_dir)).replace("\\", "/")


def list_library(base_dir: Path) -> list[dict]:
    """One row per observation folder under *base_dir*: its path (relative
    to base_dir, the only form the frontend/API ever sees or accepts -- see
    _resolve_folder), and a status breakdown over its role=="original"
    assets. Folders with zero original assets are omitted; an empty library
    is an empty list, not an error.
    """
    base_dir = Path(base_dir)
    rows = []
    for folder in discover_observation_folders(base_dir):
        originals = [a for a in scan_assets(folder) if a.role == "original"]
        if not originals:
            continue
        counts: dict[str, int] = {}
        for a in originals:
            counts[a.status] = counts.get(a.status, 0) + 1
        rows.append({
            "folder": _rel_folder(base_dir, folder),
            "total": len(originals),
            "counts": counts,
            "cropped": sum(1 for a in originals if a.crop is not None),
        })
    return rows


def folder_assets(base_dir: Path, folder_rel: str) -> list[dict]:
    """role=="original" assets for one folder (by its base_dir-relative
    path, as list_library() returns it), each as {file, status, crop,
    cache_ready} -- file is the asset's own basename, the only identifier
    the frontend needs (set_asset_status/set_asset_crop both accept a
    basename). cache_ready reflects whether a LightCraft auto-adjusted
    preview (see lightcraft_cache.py) already exists on disk for this raw
    -- it does not trigger rendering; see trigger_cache_build() for that.
    """
    folder = _resolve_folder(base_dir, folder_rel)
    return [
        {"file": Path(a.file).name, "status": a.status, "crop": a.crop,
         "cache_ready": cache_path_for(folder, Path(a.file).name).exists()}
        for a in scan_assets(folder) if a.role == "original"
    ]


_BUILD_LOCK = threading.Lock()
_BUILDING: set[str] = set()  # base_dir-relative folder paths with a build thread in flight


def trigger_cache_build(base_dir: Path, folder_rel: str) -> dict:
    """Start a background thread rendering LightCraft auto-adjusted
    previews for every role=="original" raw in this folder that doesn't
    already have one (see lightcraft_cache.build_folder_cache). Returns
    immediately -- {"started": True} the first time for a given folder,
    {"started": False, "already_running": True} if a build for this exact
    folder is already in flight (never launches a second overlapping one).
    Real failures (e.g. lightcraft-cli missing) surface per-file in the
    next /api/folder poll via cache_ready staying False forever, not as an
    exception here -- the caller only asked to kick off a background job.
    """
    folder = _resolve_folder(base_dir, folder_rel)
    with _BUILD_LOCK:
        if folder_rel in _BUILDING:
            return {"started": False, "already_running": True}
        _BUILDING.add(folder_rel)

    raw_names = [Path(a.file).name for a in scan_assets(folder) if a.role == "original"]

    def _run():
        try:
            build_folder_cache(folder, raw_names)
        finally:
            with _BUILD_LOCK:
                _BUILDING.discard(folder_rel)

    threading.Thread(target=_run, daemon=True).start()
    return {"started": True, "already_running": False}


class UnknownFolderError(ValueError):
    """Raised when a folder path from a request does not match a real
    observation folder under base_dir -- see _resolve_folder's docstring
    for why this is checked against the real discovered list rather than
    just path-joined and trusted."""


def _resolve_folder(base_dir: Path, folder_rel: str) -> Path:
    """Resolve a base_dir-relative folder path from a request to a real
    observation folder, by checking it against discover_observation_folders()
    rather than just Path(base_dir, folder_rel): a bare join would let a
    crafted folder_rel (e.g. containing "..") read or write assets.jsonl
    anywhere on disk the process can reach, not just under base_dir. Raises
    UnknownFolderError if *folder_rel* isn't one of the real folders found.
    """
    base_dir = Path(base_dir)
    wanted = folder_rel.replace("\\", "/")
    for folder in discover_observation_folders(base_dir):
        if _rel_folder(base_dir, folder) == wanted:
            return folder
    raise UnknownFolderError(f"no observation folder matches {folder_rel!r} under {base_dir}")


def _find_raw_path(folder: Path, file_name: str) -> Path:
    """The real on-disk path of the role=="original" asset named *file_name*
    (basename or folder-relative path) in *folder* -- raises FileNotFoundError
    if no such asset exists, so /img can never be made to read an arbitrary
    path: it only ever opens a file this folder's own assets.jsonl already
    named.
    """
    wanted = file_name.replace("\\", "/")
    for asset in scan_assets(folder):
        if asset.role == "original" and (asset.file == wanted or Path(asset.file).name == wanted):
            return folder / asset.file
    raise FileNotFoundError(f"no original asset matches {file_name!r} in {folder}")


def preview_jpeg_bytes(base_dir: Path, folder_rel: str, file_name: str) -> bytes:
    """A real, judgeable preview of the raw, uncropped -- the console's own
    crop editor draws the crop-selection box over this full image in the
    browser; the stored crop is only ever applied at export time (see
    render.py), not pre-applied here.

    Prefers the LightCraft auto-adjusted cache (see lightcraft_cache.py) if
    it's already been rendered for this file -- the flat embedded camera
    preview has no exposure/contrast/detail to judge sharpness by, which is
    useless for actually culling bracket frames. Falls back to the flat
    embedded preview when no cache entry exists yet (e.g. the build hasn't
    finished, or lightcraft-cli isn't installed on this machine) so the
    console still works, just with a worse starting image, rather than
    failing outright.
    """
    folder = _resolve_folder(base_dir, folder_rel)
    cached = cache_path_for(folder, file_name)
    if cached.exists():
        return cached.read_bytes()
    raw_path = _find_raw_path(folder, file_name)
    return extract_preview_bytes(raw_path)


class ReviewConsoleHandler(BaseHTTPRequestHandler):
    base_dir: Path = Path(".")  # set per-instance by serve() via a subclass

    def log_message(self, *a):
        pass

    def _send(self, code: int, body, ctype: str = "application/json") -> None:
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _host_ok(self) -> bool:
        return (self.headers.get("Host") or "").split(":")[0] in ("127.0.0.1", "localhost")

    def do_GET(self) -> None:
        if not self._host_ok():
            return self._send(403, {"error": "host"})
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        if u.path == "/":
            return self._send(200, PAGE, "text/html; charset=utf-8")
        if u.path == "/api/library":
            return self._send(200, {"folders": list_library(self.base_dir)})
        if u.path == "/api/folder":
            try:
                folder_rel = q["folder"][0]
                assets = folder_assets(self.base_dir, folder_rel)
            except KeyError:
                return self._send(400, {"error": "folder is required"})
            except UnknownFolderError as exc:
                return self._send(404, {"error": str(exc)})
            with _BUILD_LOCK:
                building = folder_rel in _BUILDING
            return self._send(200, {"assets": assets, "cache_building": building})
        if u.path == "/img":
            try:
                data = preview_jpeg_bytes(self.base_dir, q["folder"][0], q["file"][0])
            except KeyError:
                return self._send(400, {"error": "folder and file are required"})
            except (UnknownFolderError, FileNotFoundError) as exc:
                return self._send(404, {"error": str(exc)})
            return self._send(200, data, "image/jpeg")
        return self._send(404, {"error": "not found"})

    def _origin_ok(self) -> bool:
        """CSRF guard: a cross-site page can fire a simple (no-preflight)
        POST whose Host is 127.0.0.1, so also require a JSON Content-Type
        (which forces a CORS preflight the console never answers) and, when
        the browser sent an Origin, that it is this console itself."""
        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if ctype != "application/json":
            return False
        origin = self.headers.get("Origin")
        return origin is None or origin == f"http://{self.headers.get('Host')}"

    def do_POST(self) -> None:
        if not self._host_ok():
            return self._send(403, {"error": "host"})
        if not self._origin_ok():
            return self._send(403, {"error": "cross-site or non-JSON request refused"})
        n = int(self.headers.get("Content-Length") or 0)
        try:
            d = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return self._send(400, {"error": "invalid JSON body"})
        if self.path == "/api/keep":
            return self._status_action(d, "kept", True)
        if self.path == "/api/reject":
            return self._status_action(d, "rejected", False)
        if self.path == "/api/crop":
            return self._crop_action(d)
        if self.path == "/api/cull":
            return self._cull_action(d)
        if self.path == "/api/prepare":
            return self._prepare_action(d)
        return self._send(404, {"error": "not found"})

    def _prepare_action(self, d: dict) -> None:
        folder_rel = d.get("folder")
        if not folder_rel:
            return self._send(400, {"error": "folder is required"})
        try:
            return self._send(200, trigger_cache_build(self.base_dir, folder_rel))
        except UnknownFolderError as exc:
            return self._send(404, {"error": str(exc)})

    def _status_action(self, d: dict, status: str, selected: bool) -> None:
        folder_rel, files = d.get("folder"), d.get("files")
        if not folder_rel or not isinstance(files, list) or not files:
            return self._send(400, {"error": "folder and a non-empty files list are required"})
        try:
            folder = _resolve_folder(self.base_dir, folder_rel)
        except UnknownFolderError as exc:
            return self._send(404, {"error": str(exc)})
        set_asset_status(folder, files, status, selected)
        return self._send(200, {"assets": folder_assets(self.base_dir, folder_rel)})

    def _crop_action(self, d: dict) -> None:
        folder_rel, file_name, crop = d.get("folder"), d.get("file"), d.get("crop")
        if not folder_rel or not file_name:
            return self._send(400, {"error": "folder and file are required"})
        if crop is not None and (not isinstance(crop, list) or len(crop) != 4):
            return self._send(400, {"error": "crop must be [x0, y0, x1, y1] or null"})
        if crop is not None:
            try:
                validate_crop(crop)
            except ValueError as exc:
                return self._send(400, {"error": str(exc)})
        try:
            folder = _resolve_folder(self.base_dir, folder_rel)
            asset = set_asset_crop(folder, file_name, crop)
        except UnknownFolderError as exc:
            return self._send(404, {"error": str(exc)})
        except ValueError as exc:
            return self._send(404, {"error": str(exc)})
        return self._send(200, {"file": Path(asset.file).name, "crop": asset.crop})

    def _cull_action(self, d: dict) -> None:
        folder_rel = d.get("folder")
        if not folder_rel:
            return self._send(400, {"error": "folder is required"})
        try:
            folder = _resolve_folder(self.base_dir, folder_rel)
        except UnknownFolderError as exc:
            return self._send(404, {"error": str(exc)})
        deleted = delete_rejected_raws(folder)
        return self._send(200, {"deleted": [p.name for p in deleted]})


PAGE = (Path(__file__).parent / "review_console.html").read_text(encoding="utf-8")


def serve(base_dir: Path, host: str = "127.0.0.1", port: int = DEFAULT_PORT) -> None:
    base_dir = Path(base_dir)

    class _Handler(ReviewConsoleHandler):
        pass

    _Handler.base_dir = base_dir
    print(f"review console on http://{host}:{port}/  | base_dir {base_dir}", flush=True)
    ThreadingHTTPServer((host, port), _Handler).serve_forever()
