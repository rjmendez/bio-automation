"""`bio social-audit-live`: re-run the privacy gates against what is ACTUALLY public on Bluesky.

The gates (social_gates) screen a draft before it is sent; this reads the owner's recent posted records back from the public AppView
(api.bsky.app, unauthenticated GETs only, nothing is ever written to Bluesky) and applies the same checks to the published text and image alt
texts, so a gate that was too weak when the post went out, or a place that became hidden afterwards, is found.

Per post, by URI:
  coordinates / street address   in the post text or any image alt text (social_gates.alt_leaks)
  hidden place name              only for a post whose draft is on record (ledger posted uri -> draft_id -> drafts/<id>/draft.json): the place
                                 words of an observation that hides its location (geoprivacy / taxon geoprivacy / no public place)
  image floor                    the published image is too small, black, grossly under/over-exposed or far too blurry (social_gates.image_problem
                                 on the CDN's full-size image; --no-images skips the downloads), GPS tags in the file, more than 4 images, a missing alt text
Reports and audit records name the post URI and the KIND of problem, never the text that leaked (the audit log, social.log, the digest and
the backups are less private than the post's author thinks). Audit events: live_audit (every run), live_audit_problem (a run with problems),
live_audit_failed (the AppView could not be read).

Exit code: 0 clean, 1 problems found, 2 could not run. `--weekly` skips (exit 0) when the last live_audit record is under 7 days old, so the
watch script can call it every evening."""
from __future__ import annotations

import datetime as dt
import io
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable

from . import settings, social_gates
from .social_rules import Audit
from .social_style import is_bot_post

WEEK = dt.timedelta(days=7)
MAX_IMAGES = 4
PAGE = 100


class LiveAuditError(Exception):
    pass


ERRORS = (LiveAuditError,)

# Problem kinds beyond the gates' own (coordinates, street address, hidden place name).
P_GPS = "GPS tags in a public image"
P_MANY = f"more than {MAX_IMAGES} images"
P_NO_ALT = "image without alt text"


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "bio-automation/social-audit-live", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def _get_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "bio-automation/social-audit-live"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read(20_000_000)


def fetch_recent(handle: str, limit: int, *, get_json: Callable[[str], dict] | None = None) -> list[dict]:
    """The account's own recent top-level posts (reposts and replies left out), newest first, at most `limit`: AppView post views."""
    get_json = get_json or _get_json                              # looked up at call time (a default argument would freeze the first value)
    out: list[dict] = []
    cursor = None
    while len(out) < limit:
        q = {"actor": handle, "limit": min(PAGE, limit - len(out) + 20), "filter": "posts_no_replies", **({"cursor": cursor} if cursor else {})}
        url = f"{settings.BSKY_AUDIT_APPVIEW}/xrpc/app.bsky.feed.getAuthorFeed?{urllib.parse.urlencode(q)}"
        try:
            page = get_json(url)
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise LiveAuditError(f"the public AppView could not be read ({type(e).__name__}: {e})") from e
        items = page.get("feed") or []
        for item in items:
            post = item.get("post") or {}
            if item.get("reason") or not post.get("uri"):
                continue                                          # a repost (or a pinned marker) is someone else's text
            out.append(post)
        cursor = page.get("cursor")
        if not cursor or not items:
            break
    return out[:limit]


def posted_drafts(ledger_path: Path | None = None) -> dict[str, str]:
    """{post uri: draft id} from the ledger's posted records."""
    path = Path(ledger_path or settings.SOCIAL_LEDGER)
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if isinstance(r, dict) and r.get("event") == "posted" and r.get("uri") and r.get("draft_id"):
            out[r["uri"]] = r["draft_id"]
    return out


def _observations(draft_id: str, drafts_dir: Path) -> list[dict] | None:
    try:
        draft = json.loads((drafts_dir / draft_id / "draft.json").read_text(encoding="utf-8"))
        return list(social_gates.draft_observations(draft).values())
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _has_gps(data: bytes) -> bool:
    from PIL import Image

    try:
        with Image.open(io.BytesIO(data)) as im:
            return bool(im.getexif().get_ifd(0x8825))
    except (OSError, ValueError):
        return False


def _image_kinds(data: bytes) -> list[str]:
    from PIL import UnidentifiedImageError

    kinds = []
    try:
        problem = social_gates.image_problem(social_gates.measure_image(io.BytesIO(data)))
    except (OSError, UnidentifiedImageError, ValueError):
        problem = "image could not be decoded"
    if problem:
        kinds.append("image floor: " + problem.split(" (")[0])      # the reason without its numbers
    if _has_gps(data):
        kinds.append(P_GPS)
    return kinds


def check_post(post: dict, observations: list[dict] | None, *, get_bytes: Callable[[str], bytes] | None) -> list[str]:
    """The KINDS of problem of one public post ([] = clean)."""
    record = post.get("record") or {}
    text = record.get("text") or ""
    images = ((post.get("embed") or {}).get("images")) or []
    kinds: list[str] = []

    def add(kind: str) -> None:
        if kind not in kinds:
            kinds.append(kind)

    def leaks(blob: str, where: str) -> None:
        if not blob:
            return
        # the gate runs once without an observation (coordinates, address) and once per known observation (its place words too)
        for obs in observations or [{}]:
            for kind in social_gates.alt_leaks(blob, obs):
                add(f"{kind} in {where}")

    leaks(text, "post text")
    for i, img in enumerate(images, 1):
        leaks(img.get("alt") or "", f"alt of image {i}")
        if not (img.get("alt") or "").strip() and not is_bot_post(text):          # the old inat_skeet bot posts never had alt text; the cleanup removes them
            add(f"{P_NO_ALT} {i}")
    if len(images) > MAX_IMAGES:
        add(P_MANY)
    if get_bytes is not None:
        for i, img in enumerate(images, 1):
            url = img.get("fullsize") or img.get("thumb")
            if not url:
                continue
            try:
                data = get_bytes(url)
            except (urllib.error.URLError, OSError):
                add(f"image {i} could not be fetched")
                continue
            for kind in _image_kinds(data):
                add(f"{kind} (image {i})")
    return kinds


def last_audit_at(audit_path: Path | None = None) -> dt.datetime | None:
    path = Path(audit_path or settings.SOCIAL_AUDIT_LOG)
    if not path.is_file():
        return None
    latest = None
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
            if r.get("event") != "live_audit":
                continue
            at = dt.datetime.fromisoformat(r["at"])
        except (ValueError, KeyError, AttributeError):
            continue
        at = at if at.tzinfo else at.replace(tzinfo=settings.LOCAL_TZ)
        latest = at if latest is None or at > latest else latest
    return latest


def audit_live(handle: str, *, limit: int = 50, images: bool = True, get_json: Callable[[str], dict] | None = None,
               get_bytes: Callable[[str], bytes] | None = None, ledger_path: Path | None = None, drafts_dir: Path | None = None,
               audit: Audit | None = None) -> dict:
    """{"checked", "problem_posts", "place_unchecked", "problems": [{"uri", "kinds"}]}; audited. Raises LiveAuditError when the AppView fails."""
    audit = audit or Audit()
    drafts_dir = Path(drafts_dir or settings.SOCIAL_DIR / "drafts")
    try:
        posts = fetch_recent(handle, limit, get_json=get_json)
    except LiveAuditError as e:
        audit.log("live_audit_failed", reason=str(e))
        raise
    by_uri = posted_drafts(ledger_path)
    problems, unchecked = [], 0
    for post in posts:
        draft_id = by_uri.get(post["uri"])
        obs = _observations(draft_id, drafts_dir) if draft_id else None
        if obs is None:
            unchecked += 1
        kinds = check_post(post, obs, get_bytes=(get_bytes or _get_bytes) if images else None)
        if kinds:
            problems.append({"uri": post["uri"], "kinds": kinds})
    report = {"checked": len(posts), "problem_posts": len(problems), "place_unchecked": unchecked, "problems": problems}
    audit.log("live_audit", checked=len(posts), problem_posts=len(problems), place_unchecked=unchecked, handle=handle)
    if problems:
        audit.log("live_audit_problem",
                  reason=f"{len(problems)} public post(s) fail the privacy gates: " + ", ".join(p["uri"] for p in problems[:3]),
                  problems=[f"{p['uri']}: {'; '.join(p['kinds'])}" for p in problems[:5]], uris=[p["uri"] for p in problems])
    return report


def cmd_social_audit_live(args) -> int:
    if args.weekly:
        last = last_audit_at()
        now = dt.datetime.now(settings.LOCAL_TZ)
        if last is not None and now - last < WEEK:
            print(f"skipped: the last live audit was {last.isoformat(timespec='seconds')}, under 7 days ago")
            return 0
    try:
        report = audit_live(args.handle, limit=args.limit, images=not args.no_images)
    except LiveAuditError as e:
        print(f"bio social-audit-live: {e}", file=sys.stderr)
        return 2
    print(f"checked {report['checked']} public posts of {args.handle}; {report['place_unchecked']} had no draft on record "
          "(place names not checked for those)")
    for p in report["problems"]:
        print(f"PROBLEM {p['uri']}: {'; '.join(p['kinds'])}")
    if not report["problems"]:
        print("no problems")
    return 1 if report["problems"] else 0


def add_parsers(sub) -> None:
    p = sub.add_parser("social-audit-live", help="Re-run the privacy gates on the owner's recent PUBLIC posts (unauthenticated reads); reports by post URI "
                                                "only. Exit 0 clean, 1 problems, 2 could not read the AppView.")
    p.add_argument("--handle", default=settings.BSKY_HANDLE)
    p.add_argument("--limit", type=int, default=50, help="how many recent posts (default 50)")
    p.add_argument("--no-images", action="store_true", help="do not download the published images (text and alt only)")
    p.add_argument("--weekly", action="store_true", help="do nothing when the last live audit is under 7 days old (for the watch script)")


COMMANDS = {"social-audit-live": cmd_social_audit_live}
