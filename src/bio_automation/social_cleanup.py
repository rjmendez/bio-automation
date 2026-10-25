"""Removing the old inat_skeet bot posts from the owner's Bluesky account: a plan the owner approves, then one run.

  build_plan   from live public data: every post in inat_skeet's template (social_style.is_bot_post) is a candidate; the ones with
               replies from people are KEPT (deleting them would orphan the conversation), and so are any the AppView no longer
               reports (engagement unknown). The plan is saved with an id = hash of the URIs it would delete.
  run_plan     DRY RUN unless told otherwise: it reports what it would delete and touches nothing. Deleting needs BOTH the plan's id and
               --confirm-count N equal to the plan's size (the owner's approval of that list and of how long it is). It deletes exactly that plan through the "cleanup" bsky-oauth profile
               (scope: delete posts, nothing else) in batches of one session each. Every result goes to a ledger next to the plan,
               so a rerun resumes; a "failed" delete is re-checked against the public record and counted as done when it is gone.
               A rate limit or a lost session stops the run; rerun later.
               Every run and every single deletion result is also an audit record (cleanup_dry_run, cleanup_start, cleanup_delete, cleanup_end;
               the post URI and status only, never its text). Only app.bsky.feed.post URIs of the plan's own account are ever sent.

Take an archive first (the account's repository export, `com.atproto.sync.getRepo`); deletions cannot be undone.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable

from . import settings
from .social import SocialError
from .social_rules import Audit
from .social_style import _get_json, fetch_posts, is_bot_post, resolve_account

RULE = "inat_skeet template posts with no replies"
CHUNK = 50                      # deletes per helper call (one session, one token refresh each)


def cleanup_profile():
    from bsky_oauth import Profile

    return Profile(name="cleanup", client_id=settings.BSKY_CLEANUP_CLIENT_ID, redirect_uri=settings.BSKY_REDIRECT_URI,
                   scope=settings.BSKY_CLEANUP_SCOPE, state_dir=Path(settings.BSKY_CLEANUP_OAUTH_DIR),
                   client_name="bio-automation cleanup (delete old bot posts)")


def cleanup_account():
    from bsky_oauth import Account

    return Account(cleanup_profile())


def engagement(uris: list[str], *, get_json: Callable[[str], dict] = _get_json) -> dict[str, dict]:
    """{uri: {likes, reposts, replies, quotes}} from the public AppView (25 per request); URIs it does not return are absent."""
    out = {}
    for i in range(0, len(uris), 25):
        q = urllib.parse.urlencode([("uris", u) for u in uris[i:i + 25]])
        for p in get_json(f"{settings.BSKY_APPVIEW}/xrpc/app.bsky.feed.getPosts?{q}").get("posts", []):
            out[p["uri"]] = {"likes": p.get("likeCount", 0), "reposts": p.get("repostCount", 0), "replies": p.get("replyCount", 0),
                             "quotes": p.get("quoteCount", 0)}
    return out


def plan_id(uris: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(uris)).encode()).hexdigest()[:12]


def build_plan(handle: str, *, get_json: Callable[[str], dict] = _get_json, now: dt.datetime | None = None) -> dict:
    did, pds = resolve_account(handle, get_json=get_json)
    bots = [r for r in fetch_posts(did, pds, get_json=get_json) if is_bot_post(r.get("value", {}).get("text", ""))]
    eng = engagement([r["uri"] for r in bots], get_json=get_json)
    delete, keep = [], []
    for r in sorted(bots, key=lambda r: r["value"].get("createdAt", "")):
        row = {"uri": r["uri"], "created_at": r["value"].get("createdAt"), "text": r["value"].get("text", "").split("\n")[0][:80],
               "engagement": eng.get(r["uri"])}
        if row["engagement"] is None:
            keep.append({**row, "reason": "engagement unknown (not returned by the AppView)"})
        elif row["engagement"]["replies"] > 0:
            keep.append({**row, "reason": f"{row['engagement']['replies']} replies from people"})
        else:
            delete.append(row)
    return {"kind": "bsky_cleanup_plan", "rule": RULE, "handle": handle, "did": did, "pds": pds,
            "built_at": (now or dt.datetime.now(dt.timezone.utc)).isoformat(timespec="seconds"),
            "plan_id": plan_id([d["uri"] for d in delete]), "delete": delete, "keep": keep}


def save_plan(plan: dict, archive_dir: Path | None = None) -> Path:
    path = Path(archive_dir or settings.BSKY_ARCHIVE_DIR) / f"cleanup_plan_{plan['plan_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=1, ensure_ascii=False), encoding="utf-8")
    return path


def summary(plan: dict, samples: int = 5) -> dict:
    d = plan["delete"]
    kept = {"has replies": sum(1 for k in plan["keep"] if "replies" in k["reason"]),
            "engagement unknown": sum(1 for k in plan["keep"] if "unknown" in k["reason"])}
    return {"plan_id": plan["plan_id"], "rule": plan["rule"], "delete": len(d), "keep": len(plan["keep"]), "kept_because": kept,
            "date_range": [d[0]["created_at"][:10], d[-1]["created_at"][:10]] if d else None,
            "likes_on_deleted": sum(x["engagement"]["likes"] for x in d),
            "samples": [f"{x['created_at'][:10]} {x['text']}" for x in d[:: max(1, len(d) // samples)][:samples]]}


def record_exists(uri: str, pds: str, *, get_json: Callable[[str], dict] = _get_json) -> bool:
    """Whether the record is still on the PDS (public getRecord; RecordNotFound -> False)."""
    repo, collection, rkey = uri[len("at://"):].split("/")
    q = urllib.parse.urlencode({"repo": repo, "collection": collection, "rkey": rkey})
    try:
        get_json(f"{pds}/xrpc/com.atproto.repo.getRecord?{q}")
        return True
    except urllib.error.HTTPError as e:
        if e.code == 400 and b"RecordNotFound" in e.read():
            return False
        raise


def _ledger_done(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {json.loads(l)["uri"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip() and json.loads(l)["status"] == "deleted"}


def run_plan(plan_path: Path, confirm: str | None = None, *, confirm_count: int | None = None, account=None,
             exists: Callable[[str], bool] | None = None, chunk: int = CHUNK, pause_ms: int = 300, audit: Audit | None = None) -> dict:
    """Delete the plan's posts, or (without both `confirm` and `confirm_count`) only report what would be deleted. confirm must equal the plan's
    id (recomputed from its URIs, so an edited plan is refused) and confirm_count its number of posts."""
    from bsky_oauth import BatchInterrupted, NothingSent

    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    uris = [d["uri"] for d in plan["delete"]]
    if plan_id(uris) != plan["plan_id"]:
        raise SocialError(f"{plan_path} was edited after it was built (its URIs no longer hash to {plan['plan_id']}): build a new plan")
    own = f"at://{plan['did']}/app.bsky.feed.post/" if plan.get("did") else None
    foreign = [u for u in uris if not (u.startswith(own) if own else "/app.bsky.feed.post/" in u)]
    if foreign:
        raise SocialError(f"{len(foreign)} URI(s) of the plan are not app.bsky.feed.post records of the plan's account (first: {foreign[0]}): "
                          "nothing is deleted; build a new plan")
    if (confirm is None) != (confirm_count is None):
        raise SocialError("deleting needs both --confirm <plan id> and --confirm-count <number of posts in the plan>; give neither for a dry run")
    execute = confirm is not None
    if execute and confirm != plan["plan_id"]:
        raise SocialError(f"--confirm {confirm!r} does not match plan {plan['plan_id']}: review the plan, then confirm with its id")
    if execute and confirm_count != len(uris):
        raise SocialError(f"--confirm-count {confirm_count} does not match the plan, which deletes {len(uris)} posts: review the plan, then confirm its size")
    ledger = Path(plan_path).with_suffix(".ledger.jsonl")
    done = _ledger_done(ledger)
    todo = [u for u in uris if u not in done]
    audit = audit or Audit()
    if not execute:
        audit.log("cleanup_dry_run", plan_id=plan["plan_id"], would_delete=len(todo), already_done=len(done))
        return {"plan_id": plan["plan_id"], "dry_run": True, "plan_size": len(uris), "already_done": len(done), "would_delete": len(todo),
                "deleted": 0, "failed": 0, "stopped": None, "remaining": len(todo)}
    audit.log("cleanup_start", plan_id=plan["plan_id"], plan_size=len(uris), todo=len(todo), already_done=len(done))   # an unwritable audit log stops the run before anything is deleted
    acct = account or cleanup_account()
    exists = exists or (lambda u: record_exists(u, plan["pds"]))
    report = {"plan_id": plan["plan_id"], "dry_run": False, "already_done": len(done), "deleted": 0, "failed": 0, "stopped": None}

    def log(uri: str, status: str, error: str | None = None) -> None:
        with open(ledger, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps({"at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "uri": uri, "status": status,
                                **({"error": error} if error else {})}) + "\n")
        audit.log("cleanup_delete", plan_id=plan["plan_id"], uri=uri, status=status, **({"error": error} if error else {}))

    def settle(results: list[dict]) -> bool:
        """Log results; True when the run must stop (rate limit)."""
        for r in results:
            status = r["status"]
            if status == "failed" and not exists(r["uri"]):
                status = "deleted"                       # the record is gone: the delete took effect (or it was already deleted)
            log(r["uri"], status, r.get("error"))
            report[status] = report.get(status, 0) + 1
            if status == "rate_limited":
                report["stopped"] = "rate limited: rerun later to continue"
                return True
        return False

    try:
        for i in range(0, len(todo), chunk):
            try:
                results = acct.delete_many(todo[i:i + chunk], pause_ms=pause_ms)
            except NothingSent as e:
                report["stopped"] = f"nothing sent: {e}"
                break
            except BatchInterrupted as e:
                settle(e.results)
                report["stopped"] = f"interrupted: {e}; rerun to continue (each remaining post is re-checked)"
                break
            if settle(results):
                break
    finally:
        report["remaining"] = len(uris) - len(_ledger_done(ledger))
        audit.log("cleanup_end", plan_id=plan["plan_id"], deleted=report["deleted"], failed=report["failed"], remaining=report["remaining"],
                  stopped=report["stopped"])
    return report
