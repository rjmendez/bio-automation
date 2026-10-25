"""Credential hygiene for the posting workflow: redaction before anything is written, and a scan that finds what slipped through.

redact(value) walks dicts/lists/strings and replaces secret material with REDACTED:
  * header/field names: Authorization, Proxy-Authorization, DPoP, Cookie, access/refresh/id tokens, api keys, passwords, client secrets;
  * the private members of a JWK (d, p, q, dp, dq, qi, k) in any dict that has a "kty";
  * inside strings: JWTs, "Bearer|DPoP|Basic <token>", "Authorization: <scheme> <token>", PEM private key blocks, JSON `"d": "..."` members,
    Cookie / Set-Cookie header values, and `key=value` / `key: value` / `"key": "value"` pairs (query strings, error text) for access_token,
    refresh_token, id_token, password, client_secret, api_key and the like, plus `?token=` / `&token=`;
  * bytes are decoded and redacted as text; any other object (an exception, a Path) is redacted as str(value) and comes back as a str.
scan_paths(paths) reports files holding token-shaped strings, with the kinds split into STRONG (a real secret shape: JWT, PEM, bearer, key) and weak
(cookie / key=value forms, which also match harmless prose); it returns paths and kinds only, never the matched text, and reads at most
MAX_SCAN_FILES files / MAX_SCAN_FILE_BYTES per file (the tail of a bigger file) / MAX_SCAN_TOTAL_BYTES in all.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, NamedTuple

REDACTED = "[REDACTED]"

SECRET_KEYS = frozenset({
    "authorization", "proxy-authorization", "dpop", "cookie", "set-cookie", "x-api-key", "api-key", "api_key", "apikey",
    "access_token", "refresh_token", "id_token", "token", "password", "passwd", "secret", "client_secret", "private_key",
    "privatekey", "jwk_private", "app_password"})
JWK_PRIVATE = frozenset({"d", "p", "q", "dp", "dq", "qi", "k"})

_B64 = r"[A-Za-z0-9._~+/=-]"
_STRING_PATTERNS: tuple[tuple[str, re.Pattern, str], ...] = (
    ("pem", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(?:-----END [A-Z ]*PRIVATE KEY-----|\Z)", re.S), REDACTED),
    ("authorization-header", re.compile(r"(?i)(authorization['\"]?\s*[:=]\s*['\"]?)[A-Za-z]+\s+" + _B64 + r"{8,}"), r"\1" + REDACTED),
    ("bearer", re.compile(r"(?i)\b(bearer|dpop)\s+" + _B64 + r"{16,}"), r"\1 " + REDACTED),
    ("jwt", re.compile(r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"), REDACTED),
    ("jwk-private", re.compile(r'("(?:d|p|q|dp|dq|qi)"\s*:\s*")[A-Za-z0-9_-]{20,}(")'), r"\1" + REDACTED + r"\2"),
    ("api-key", re.compile(r"\b(?:sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{30,})\b"), REDACTED),
    ("cookie", re.compile(r"(?i)\b((?:set-)?cookie['\"]?\s*:\s*['\"]?)(?!\[REDACTED\])[^\r\n'\"]{4,}"), r"\1" + REDACTED),
    ("secret-pair", re.compile(r"(?i)\b((?:access|refresh|id)[_-]?token|password|passwd|client[_-]?secret|app[_-]?password|api[_-]?key|apikey|private[_-]?key)(['\"]?\s*[:=]\s*['\"]?)(?!\[REDACTED\])[^\s&'\",;]{4,}"),
     r"\1\2" + REDACTED),
    ("query-token", re.compile(r"(?i)([?&]token=)(?!\[REDACTED\])[^\s&'\"]{4,}"), r"\1" + REDACTED),
)
STRONG_KINDS = frozenset({"pem", "authorization-header", "bearer", "jwt", "jwk-private", "api-key"})


def redact_text(text: str) -> str:
    for _, pat, repl in _STRING_PATTERNS:
        text = pat.sub(repl, text)
    return text


def redact(value):
    """A copy of `value` with secret material replaced; input is not modified."""
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, (bytes, bytearray)):
        return redact_text(bytes(value).decode("utf-8", "replace"))
    if isinstance(value, dict):
        is_jwk = "kty" in value
        out = {}
        for k, v in value.items():
            name = str(k).lower()
            if name in SECRET_KEYS or (is_jwk and name in JWK_PRIVATE):
                out[k] = REDACTED
            else:
                out[k] = redact(v)
        return out
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return redact_text(str(value))                      # an exception, a Path...: its text may hold a token


# ------------------------------------------------------------------------------------------------------------------------ scan

class Finding(NamedTuple):
    path: Path
    kinds: tuple[str, ...]


_SCAN_KINDS = tuple((n, p) for n, p, _ in _STRING_PATTERNS)
MAX_SCAN_FILES = 5000
MAX_SCAN_FILE_BYTES = 4 * 1024 * 1024           # a bigger file (a growing log) is scanned by its last 4 MB
MAX_SCAN_TOTAL_BYTES = 256 * 1024 * 1024
SKIP_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".gif", ".nef", ".dng", ".onnx", ".npy", ".pt", ".sqlite3", ".db"})


def scan_text(text: str) -> tuple[str, ...]:
    return tuple(name for name, pat in _SCAN_KINDS if pat.search(text))


def _files(paths: Iterable[Path]):
    for root in paths:
        root = Path(root)
        if root.is_file():
            yield root
        elif root.is_dir():
            for dp, _, names in os.walk(root):
                for n in names:
                    yield Path(dp) / n


class ScanResult(NamedTuple):
    findings: list
    files: int          # files read
    capped: bool        # the file or byte cap stopped the scan early


def _read_tail(f: Path, limit: int) -> str:
    size = f.stat().st_size
    with open(f, "rb") as fh:
        if size > limit:
            fh.seek(size - limit)
        return fh.read(limit).decode("utf-8", "replace")


def scan(paths: Iterable[Path], *, max_files: int = MAX_SCAN_FILES, max_file_bytes: int = MAX_SCAN_FILE_BYTES,
         max_total_bytes: int = MAX_SCAN_TOTAL_BYTES) -> ScanResult:
    """Files under `paths` (files or folders) with token-shaped strings; bounded by the three caps (never reads a whole huge file)."""
    out, n, total, capped = [], 0, 0, False
    for f in _files(paths):
        if n >= max_files or total >= max_total_bytes:
            capped = True
            break
        try:
            if f.suffix.lower() in SKIP_SUFFIXES:
                continue
            text = _read_tail(f, max_file_bytes)
        except OSError:
            continue
        n += 1
        total += len(text)
        kinds = scan_text(text)
        if kinds:
            out.append(Finding(f, kinds))
    return ScanResult(out, n, capped)


def scan_paths(paths: Iterable[Path], **caps) -> list[Finding]:
    """scan(...).findings. Unreadable files are skipped; no matched text is returned."""
    return scan(paths, **caps).findings


# Key material that must never be copied into a backup.
KEY_FILE_NAMES = ("client_key.jwk.json", "sessions.json", "states.json")


def is_key_material(path: Path, protected_dirs: Iterable[Path] = (), *, max_entries: int = 5000) -> bool:
    """True for a key file, anything under a protected dir, or a directory that contains any of those (walked, at most max_entries entries).
    make_backup applies it to every source as defence in depth for custom `sources`; the production sources are seven fixed non-key files, so the
    check that covers the production path is the selfcheck's secret-scan of the backup folder."""
    p = Path(path)
    if p.is_dir():
        seen = 0
        for dp, _, names in os.walk(p):
            for n in names:
                seen += 1
                if seen > max_entries or is_key_material(Path(dp) / n, protected_dirs):
                    return True
        return False
    if p.name.lower() in KEY_FILE_NAMES or p.name.lower().endswith((".jwk.json", ".pem")):
        return True
    try:
        rp = p.resolve()
        for d in protected_dirs:
            if rp == Path(d).resolve() or Path(d).resolve() in rp.parents:
                return True
    except OSError:
        pass
    return False
