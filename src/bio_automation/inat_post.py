"""Posts console-approved observations to iNaturalist.

Ports legacy pipeline/stages/inat_post.py's safety properties and its two-
step observation-then-photos posting shape, not its HTTP plumbing:
dry-run by default (never posts for real unless the caller explicitly asks
for it), idempotent via the posted ledger (a group already "posted" or
"partial" is never re-submitted), and stop-on-first-problem (one bad post
halts the batch instead of being retried automatically) -- but, matching
the legacy script exactly, a failure does not raise: it is caught, written
to the posted ledger as "partial" (an observation was created but at least
one photo failed to attach) or "failed" (no observation was created at
all), and *that* row's non-"posted" status is what stops the batch.

The real HTTP transport (POST to api.inaturalist.org/v1/observations, then
one POST per photo to /observation_photos) is isolated behind *create_fn*/
*attach_fn* -- functions this module calls, never implements inline, so
callers (and tests) can swap them for a recording stub without touching
the logic above them.
"""
from __future__ import annotations

import json
import mimetypes
import urllib.error
import urllib.request
import uuid as uuid_module
from typing import Callable

from .inat import INAT_API_BASE, RateLimiter
from .inat_queue import QueueItem

# Matches legacy inat_post.py's UA exactly (see its module docstring).
USER_AGENT = "bio-automation-poster/0.1 (personal uploader)"

DEFAULT_PACE_SECONDS = 1.1  # legacy inat_post.py's pace(): 1.1 - elapsed, i.e. <= ~1 req/s with headroom
DEFAULT_LIMIT = 10


class MissingCredentialError(Exception):
    """dry_run=False was requested with no jwt -- legacy inat_post.py's
    `sys.exit('INAT_JWT is not set ...')` as a real exception instead."""


class InatAuthError(Exception):
    """The API rejected the request's credentials (e.g. an expired
    INAT_JWT)."""


class AlreadyPostedError(Exception):
    """post_one() was asked to post a group the posted ledger already has
    a "posted" or "partial" row for. post_approved() checks the ledger
    itself and skips these before ever calling post_one() (that's the
    whole point of the ledger); this only fires when post_one() is called
    directly on an already-done group."""


def _is_done(posted: dict[int, dict], group: int) -> bool:
    row = posted.get(group)
    return row is not None and row.get("status") in ("posted", "partial")


def _default_create_observation(item: QueueItem, jwt: str) -> dict:
    """Real transport, step 1: POST the observation itself. This is the
    network boundary -- tests pass their own create_fn instead of
    exercising this against the live API."""
    body = json.dumps({"observation": {}}).encode("utf-8")
    request = urllib.request.Request(
        f"{INAT_API_BASE}/observations", data=body,
        headers={"User-Agent": USER_AGENT, "Authorization": jwt,
                 "Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise InatAuthError(f"iNaturalist rejected credentials (HTTP {exc.code})") from exc
        raise
    result = payload["results"][0] if isinstance(payload.get("results"), list) else payload
    return {"observation_id": result.get("id"), "uuid": result.get("uuid")}


def _multipart(fields: dict, filename: str, data: bytes) -> tuple[bytes, str]:
    boundary = uuid_module.uuid4().hex
    parts = []
    for key, value in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode())
    content_type = mimetypes.guess_type(filename)[0] or "image/jpeg"
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f'Content-Type: {content_type}\r\n\r\n'.encode()
    )
    parts.append(data)
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _default_attach_photo(observation_id: int, path: str, jwt: str) -> None:
    """Real transport, step 2: POST one photo. Network boundary, same as
    _default_create_observation."""
    with open(path, "rb") as handle:
        file_bytes = handle.read()
    body, content_type = _multipart({"observation_photo[observation_id]": observation_id},
                                     path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1], file_bytes)
    request = urllib.request.Request(
        f"{INAT_API_BASE}/observation_photos", data=body,
        headers={"User-Agent": USER_AGENT, "Authorization": jwt, "Content-Type": content_type},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120):
            pass
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise InatAuthError(f"iNaturalist rejected credentials (HTTP {exc.code})") from exc
        raise


def post_one(item: QueueItem, *, jwt: str | None, posted: dict[int, dict],
             create_fn: Callable[[QueueItem, str], dict] | None = None,
             attach_fn: Callable[[int, str, str], None] | None = None,
             dry_run: bool = True) -> dict:
    """Post (or dry-run) a single approved QueueItem. Raises
    AlreadyPostedError if item.group already has a "posted"/"partial" row
    in *posted* -- callers doing a batch should check the ledger
    themselves (see post_approved) and never reach this for a done group;
    this is the loud failure path for a caller that calls post_one()
    directly anyway.

    Never raises for an actual posting failure (network error, rejected
    credentials, ...): that is caught and reflected in the returned row's
    "status" ("partial" if the observation was created before the failure,
    "failed" if not), matching legacy inat_post.py's per-item try/except.
    """
    if _is_done(posted, item.group):
        done_row = posted[item.group]
        raise AlreadyPostedError(
            f"group {item.group} was already {done_row['status']} "
            f"(observation_id={done_row.get('observation_id')})"
        )
    if dry_run:
        return {"group": item.group, "status": "dry_run", "observation_id": None,
                "uuid": None, "photos": 0, "error": None}

    create_fn = create_fn or _default_create_observation
    attach_fn = attach_fn or _default_attach_photo
    observation_id = None
    uuid_value = None
    photos_sent = 0
    try:
        created = create_fn(item, jwt)
        observation_id = created["observation_id"]
        uuid_value = created.get("uuid")
        for path in item.photos:
            attach_fn(observation_id, path, jwt)
            photos_sent += 1
        return {"group": item.group, "status": "posted", "observation_id": observation_id,
                "uuid": uuid_value, "photos": photos_sent, "error": None}
    except Exception as exc:
        status = "partial" if observation_id is not None else "failed"
        return {"group": item.group, "status": status, "observation_id": observation_id,
                "uuid": uuid_value, "photos": photos_sent, "error": f"{type(exc).__name__}: {exc}"}


def post_approved(items: list[QueueItem], decisions: dict[int, dict], posted: dict[int, dict], *,
                   jwt: str | None = None,
                   create_fn: Callable[[QueueItem, str], dict] | None = None,
                   attach_fn: Callable[[int, str, str], None] | None = None,
                   dry_run: bool = True,
                   limit: int = DEFAULT_LIMIT,
                   rate_limiter: RateLimiter | None = None) -> list[dict]:
    """Post every item decisions marks "approved", in queue order, up to
    *limit* (legacy default 10), skipping any already "posted"/"partial"
    in *posted* (idempotent dedupe -- create_fn/attach_fn are never called
    for those; a "failed" row IS retried, since no observation exists yet
    for it). Stops at the first item whose result status isn't "posted":
    that row is still recorded (both into *posted*, mutated in place, and
    into the returned list), but nothing after it is attempted, matching
    legacy inat_post.py's "stopping after the first problem" behavior.

    Raises MissingCredentialError immediately, before attempting anything,
    if dry_run=False and no jwt was given -- legacy inat_post.py's
    `sys.exit('INAT_JWT is not set ...')` check, ported as a real
    exception instead of a process exit.
    """
    if not dry_run and not jwt:
        raise MissingCredentialError(
            "INAT_JWT is not set (get it from "
            "https://www.inaturalist.org/users/api_token while logged in)"
        )

    limiter = rate_limiter if rate_limiter is not None else RateLimiter(min_interval=DEFAULT_PACE_SECONDS)
    todo = [item for item in items
            if decisions.get(item.group, {}).get("status") == "approved" and not _is_done(posted, item.group)]

    results: list[dict] = []
    for item in todo[:limit]:
        limiter.wait()
        row = post_one(item, jwt=jwt, posted=posted, create_fn=create_fn, attach_fn=attach_fn, dry_run=dry_run)
        results.append(row)
        if not dry_run:
            posted[item.group] = row
        if row["status"] not in ("posted", "dry_run"):
            break
    return results
