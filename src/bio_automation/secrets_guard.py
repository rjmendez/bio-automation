"""Credential hygiene for the posting workflow: redaction before anything is written, and a scan that finds what slipped through.

redact(value) walks dicts/lists/strings and replaces secret material with REDACTED:
  * header/field names: Authorization, Proxy-Authorization, DPoP, Cookie, access/refresh/id tokens, api keys, passwords, client secrets;
  * the private members of a JWK (d, p, q, dp, dq, qi, k) in any dict that has a "kty";
  * inside strings: JWTs, "Bearer|DPoP|Basic <token>", "Authorization: <scheme> <token>", PEM private key blocks, JSON `"d": "..."` members.
scan_paths(paths) reports files holding token-shaped strings. It returns paths and kinds only, never the matched text.
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
)


def redact_text(text: str) -> str:
    for _, pat, repl in _STRING_PATTERNS:
        text = pat.sub(repl, text)
    return text


def redact(value):
    """A copy of `value` with secret material replaced; input is not modified."""
    if isinstance(value, str):
        return redact_text(value)
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
    return value


# ------------------------------------------------------------------------------------------------------------------------ scan

class Finding(NamedTuple):
    path: Path
    kinds: tuple[str, ...]


_SCAN_KINDS = tuple((n, p) for n, p, _ in _STRING_PATTERNS)
MAX_SCAN_BYTES = 64 * 1024 * 1024
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


def scan_paths(paths: Iterable[Path]) -> list[Finding]:
    """Files under `paths` (files or folders) that contain token-shaped strings. Unreadable files are skipped; no matched text is returned."""
    out = []
    for f in _files(paths):
        try:
            if f.suffix.lower() in SKIP_SUFFIXES or f.stat().st_size > MAX_SCAN_BYTES:
                continue
            text = f.read_bytes().decode("utf-8", "replace")
        except OSError:
            continue
        kinds = scan_text(text)
        if kinds:
            out.append(Finding(f, kinds))
    return out


# Key material that must never be copied into a backup.
KEY_FILE_NAMES = ("client_key.jwk.json", "sessions.json", "states.json")


def is_key_material(path: Path, protected_dirs: Iterable[Path] = ()) -> bool:
    p = Path(path)
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
