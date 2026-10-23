"""Post verification: after a live post, read it back from Bluesky's PUBLIC AppView (no login) and prove it is what was sent.

Checked: the record exists, its author is the DID in the URI, the text is identical to the request text, the number of images and every
alt text (in order) match, and every link and tag of the draft's text segments is present as a facet over exactly its own text.
The post_send hook (rule verify_posts) audits post_verified or post_verify_failed; it never raises and never undoes a post.
`bio social-verify [URI | --draft ID | --last]` runs the same check on an existing post."""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable

from . import settings, social_hooks
from .social import Ledger, SocialError, load_draft, segments_text, strip_review_claim
from .social_rules import Audit

APPVIEW_URL = "https://api.bsky.app/xrpc/app.bsky.feed.getPosts"
USER_AGENT = "bio-automation-post-verify/0.1"
ATTEMPTS = 6                      # the AppView indexes a new post a little after it is created
DELAY_SECONDS = 3
LINK_TYPE = "app.bsky.richtext.facet#link"
TAG_TYPE = "app.bsky.richtext.facet#tag"


class VerifyError(SocialError):
    pass


def http_get(url: str) -> dict:
    """GET `url` unauthenticated and return the parsed JSON (an HTTP or network error propagates)."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode("utf-8"))


def posts_url(uri: str) -> str:
    return APPVIEW_URL + "?" + urllib.parse.urlencode({"uris": uri})


def uri_did(uri: str) -> str:
    """The DID of an at:// post URI."""
    if not uri.startswith("at://"):
        raise VerifyError(f"not an at:// post URI: {uri!r}")
    return uri[len("at://"):].split("/", 1)[0]


def expected_facets(segments: list[list[str]]) -> list[tuple[str, str, str]]:
    """(facet type, value, covered text) for each link and tag segment, in order."""
    out = []
    for seg in segments:
        if seg[0] == "link":
            out.append((LINK_TYPE, seg[2], seg[1]))
        elif seg[0] == "tag":
            out.append((TAG_TYPE, seg[2], seg[1]))
    return out


def _facet_problems(record: dict, segments: list[list[str]]) -> list[str]:
    text_bytes = (record.get("text") or "").encode("utf-8")
    found = []
    for facet in record.get("facets") or []:
        idx = facet.get("index") or {}
        for feat in facet.get("features") or []:
            found.append((feat.get("$type"), feat.get("uri") if feat.get("$type") == LINK_TYPE else feat.get("tag"),
                          text_bytes[idx.get("byteStart", 0):idx.get("byteEnd", 0)].decode("utf-8", "replace")))
    problems = []
    for kind, value, shown in expected_facets(segments):
        label = "link" if kind == LINK_TYPE else "tag"
        same = [f for f in found if f[0] == kind and f[1] == value]
        if not same:
            problems.append(f"missing {label} facet {value}")
        elif all(f[2] != shown for f in same):
            problems.append(f"{label} facet {value} covers {same[0][2]!r} instead of {shown!r}")
    return problems


def compare(post: dict, *, uri: str, text: str, alts: list[str], segments: list[list[str]]) -> tuple[list[str], list[str]]:
    """(checks that passed, problems) for one AppView post against what was sent."""
    checks: list[str] = []
    problems: list[str] = []
    record = post.get("record") or {}
    author = (post.get("author") or {}).get("did")
    if author == uri_did(uri):
        checks.append("author")
    else:
        problems.append(f"author is {author!r}, the URI says {uri_did(uri)!r}")
    if record.get("text") == text:
        checks.append("text")
    else:
        problems.append(f"text differs: sent {text!r}, read back {record.get('text')!r}")
    images = (record.get("embed") or {}).get("images") or []
    if len(images) == len(alts):
        checks.append("image_count")
        wrong = [f"image {n + 1} alt differs: sent {a!r}, read back {img.get('alt')!r}"
                 for n, (a, img) in enumerate(zip(alts, images)) if img.get("alt") != a]
        problems.extend(wrong)
        if not wrong:
            checks.append("alts")
    else:
        problems.append(f"image count differs: sent {len(alts)}, read back {len(images)}")
    facet_problems = _facet_problems(record, segments)
    problems.extend(facet_problems)
    if not facet_problems:
        checks.append("facets")
    return checks, problems


def verify_post(uri: str, *, text: str, alts: list[str], segments: list[list[str]], get: Callable[[str], dict] | None = None,
                sleep: Callable[[float], None] | None = None, attempts: int = ATTEMPTS, delay: float = DELAY_SECONDS) -> dict:
    """Read `uri` back from the AppView (up to `attempts` times, `delay` seconds apart, while it is not there yet or the call fails) and compare.
    -> {"ok", "checks", "problems", "attempts"}. Never raises on network or data errors; they become problems."""
    get = get or http_get
    sleep = sleep or time.sleep
    url = posts_url(uri)
    why = "no attempt was made"
    used = 0
    for used in range(1, attempts + 1):
        if used > 1:
            sleep(delay)
        try:
            posts = get(url).get("posts") or []
        except Exception as e:  # noqa: BLE001 -- any failure to read is a problem to report, never a crash
            why = f"the AppView call failed ({type(e).__name__}: {e})"[:300]
            continue
        if not posts:
            why = "the AppView does not have the post"
            continue
        try:
            checks, problems = compare(posts[0], uri=uri, text=text, alts=alts, segments=segments)
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "checks": [], "problems": [f"the AppView answer could not be read ({type(e).__name__}: {e})"[:300]], "attempts": used}
        return {"ok": not problems, "checks": checks, "problems": problems, "attempts": used}
    return {"ok": False, "checks": [], "problems": [f"{why} after {used} attempts"], "attempts": used}


def make_hook(*, get=None, sleep=None, audit: Audit | None = None, attempts: int = ATTEMPTS, delay: float = DELAY_SECONDS):
    """The post_send hook with injectable network, clock and audit log."""

    def verify_hook(ctx: dict) -> None:
        log = audit or Audit()
        who = {"draft_id": ctx.get("draft_id"), "uri": ctx.get("uri")}
        try:
            request = ctx["request"]
            segments = (ctx.get("draft") or {}).get("text")
            result = verify_post(ctx["uri"], text=request["text"], alts=[i["alt"] for i in request["images"]],
                                 segments=segments if isinstance(segments, list) else [], get=get, sleep=sleep, attempts=attempts, delay=delay)
            if result["ok"]:
                log.log("post_verified", checks=result["checks"], attempts=result["attempts"], **who)
            else:
                log.log("post_verify_failed", problems=result["problems"], attempts=result["attempts"], **who)
        except Exception as e:  # noqa: BLE001 -- verification never disturbs a post that exists
            try:
                log.log("post_verify_failed", problems=[f"verification could not run ({type(e).__name__}: {e})"[:300]], attempts=0, **who)
            except OSError:
                pass
        return None

    return verify_hook


def register_hooks() -> None:
    social_hooks.register("verify_posts", "post_send", make_hook(), rule="verify_posts", priority=10)


# --------------------------------------------------------------------------------------------------- bio social-verify

def _expected_for_draft(draft_id: str) -> tuple[str, list[str], list[list[str]], str | None]:
    """(text, alts, segments, uri) of a draft as it would have been sent, from the ledger and the draft folder."""
    ledger = Ledger()
    draft = ledger.drafts().get(draft_id)
    if draft is None:
        raise VerifyError(f"no draft {draft_id!r} in the ledger")
    uri = None
    for rec in ledger.records():
        if rec["draft_id"] == draft_id and rec["event"] == "posted":
            uri = rec.get("uri")
    ddir = settings.SOCIAL_DIR / "drafts" / draft_id
    if not (ddir / "draft.json").is_file():
        raise VerifyError(f"the draft folder of {draft_id} is missing ({ddir})")
    data = load_draft(ddir)
    return segments_text(data["text"]), [i["alt"] for i in data["images"]], data["text"], uri


def cmd_social_verify(args) -> int:
    ledger = Ledger()
    draft_id = args.draft
    uri = args.uri
    if args.last:
        posted = [r for r in ledger.records() if r["event"] == "posted" and r.get("uri")]
        if not posted:
            raise VerifyError("the ledger has no posted draft")
        draft_id, uri = posted[-1]["draft_id"], posted[-1]["uri"]
    if draft_id is None and uri is None:
        raise VerifyError("give a post URI, --draft ID or --last")
    if draft_id is None:
        for rec in ledger.records():
            if rec["event"] == "posted" and rec.get("uri") == uri:
                draft_id = rec["draft_id"]
    if draft_id is None:
        raise VerifyError(f"{uri} is not a post in the ledger; nothing to compare it with")
    text, alts, segments, ledger_uri = _expected_for_draft(draft_id)
    uri = uri or ledger_uri
    if not uri:
        raise VerifyError(f"draft {draft_id} has no posted URI")
    result = verify_post(uri, text=text, alts=alts, segments=segments, attempts=1)
    if not result["ok"] and any(p.startswith("text differs") for p in result["problems"]):
        stripped = strip_review_claim(segments)                 # approval removes the legacy review-claim line before sending
        if stripped != segments:
            again = verify_post(uri, text=segments_text(stripped), alts=alts, segments=stripped, attempts=1)
            if again["ok"]:
                result = again
    print(f"{'OK  ' if result['ok'] else 'FAIL'} {draft_id} {uri}")
    print("  passed: " + (", ".join(result["checks"]) or "nothing"))
    for p in result["problems"]:
        print("  problem: " + p)
    return 0 if result["ok"] else 1


def add_parsers(sub) -> None:
    p = sub.add_parser("social-verify", help="Read a posted Bluesky post back from the public AppView and compare it with the draft that was sent.")
    p.add_argument("uri", nargs="?", help="at:// URI of the post (it must be in the ledger)")
    p.add_argument("--draft", help="a posted draft id")
    p.add_argument("--last", action="store_true", help="the most recently posted draft")


COMMANDS = {"social-verify": cmd_social_verify}
ERRORS = (VerifyError,)
