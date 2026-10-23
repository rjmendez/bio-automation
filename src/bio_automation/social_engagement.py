"""Engagement analytics for the Bluesky posts: which themes, days and hours work.

`bio social-engagement [--days N] [--refresh] [--report]` takes the posted URIs of the last N days from the ledger, asks the PUBLIC AppView
(app.bsky.feed.getPosts, unauthenticated, 25 URIs per request, one request a second) for their counts and appends a snapshot per post to
settings.SOCIAL_ENGAGEMENT_LOG. Only counts are stored: no other user's text, handle or reply ever reaches the file. The report aggregates the
latest snapshot of each post by theme, weekday, slot hour and (when the taxonomy cache knows the observation) kingdom and class.
engagement_by_theme() is the entry the digest builder (social_alerts) imports."""
from __future__ import annotations

import datetime as dt
import json
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable

from . import settings
from .social import Ledger

APPVIEW = "https://api.bsky.app/xrpc/app.bsky.feed.getPosts"
BATCH = 25                          # getPosts takes at most 25 URIs
PAUSE_S = 1.0                       # one request a second
DEFAULT_DAYS = 30
MIN_POSTS = 2                       # a group needs this many posts before a suggestion is built on it
COUNTS = ("likeCount", "repostCount", "replyCount", "quoteCount")
WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
USER_AGENT = "bio-automation-engagement/1 (read-only public counts)"


class EngagementError(RuntimeError):
    pass


ERRORS = (EngagementError,)


# --------------------------------------------------------------------------------------------------- the posts and the fetch

def posted(ledger: Ledger, *, days: int, now: dt.datetime) -> list[dict]:
    """The posts of the last `days` days (a post exactly `days` days old is in), oldest first: {uri, draft_id, theme, weekday, hour, at, obs_id}.
    A draft posted twice counts its latest post; a post without a uri is skipped."""
    cutoff = now - dt.timedelta(days=days)
    drafts = ledger.drafts()
    latest: dict[str, dict] = {}
    for r in ledger.records():
        if r["event"] == "posted" and r.get("uri"):
            latest[r["draft_id"]] = r
    out = []
    for draft_id, r in latest.items():
        at = dt.datetime.fromisoformat(r["at"])
        if at < cutoff:
            continue
        d = drafts.get(draft_id, {})
        ids = d.get("obs_ids") or [r.get("obs_id") or d.get("obs_id")]
        out.append({"uri": r["uri"], "draft_id": draft_id, "theme": d.get("theme") or "single", "weekday": WEEKDAYS[at.weekday()],
                    "hour": at.hour, "at": r["at"], "obs_id": ids[0]})
    return sorted(out, key=lambda p: p["at"])


def http_get(url: str, params: list[tuple[str, str]]) -> dict:
    full = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(full, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise EngagementError(f"the Bluesky AppView did not answer ({type(e).__name__}: {e}); nothing more was stored this run") from e


def fetch_counts(uris: list[str], *, get: Callable = http_get, sleep: Callable[[float], None] = time.sleep) -> dict[str, dict]:
    """{uri: {likeCount, repostCount, replyCount, quoteCount}} for the posts the AppView returns (a deleted post is simply absent)."""
    out: dict[str, dict] = {}
    for i in range(0, len(uris), BATCH):
        if i:
            sleep(PAUSE_S)
        reply = get(APPVIEW, [("uris", u) for u in uris[i:i + BATCH]])
        for post in reply.get("posts", []):
            out[post["uri"]] = {k: int(post.get(k) or 0) for k in COUNTS}
    return out


def snapshot(*, days: int = DEFAULT_DAYS, ledger: Ledger | None = None, log: Path | None = None, get: Callable = http_get,
             sleep: Callable[[float], None] = time.sleep, now: dt.datetime | None = None) -> dict:
    """Fetch the counts of the recent posts and append one snapshot line per post found. Returns {"posts", "snapshots", "missing": [uris]}."""
    now = now or dt.datetime.now(settings.LOCAL_TZ)
    log = Path(log or settings.SOCIAL_ENGAGEMENT_LOG)
    posts = posted(ledger or Ledger(), days=days, now=now)
    counts = fetch_counts([p["uri"] for p in posts], get=get, sleep=sleep)
    lines = []
    for p in posts:
        if p["uri"] in counts:
            lines.append({"at": now.isoformat(timespec="seconds"), "uri": p["uri"], "draft_id": p["draft_id"], "theme": p["theme"],
                          "weekday": p["weekday"], "hour": p["hour"], **counts[p["uri"]]})
    if lines:
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a", encoding="utf-8", newline="\n") as f:
            f.write("".join(json.dumps(x, sort_keys=True) + "\n" for x in lines))
    return {"posts": len(posts), "snapshots": len(lines), "missing": [p["uri"] for p in posts if p["uri"] not in counts]}


# --------------------------------------------------------------------------------------------------- aggregation

def read_snapshots(log: Path | None = None) -> list[dict]:
    log = Path(log or settings.SOCIAL_ENGAGEMENT_LOG)
    if not log.is_file():
        return []
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]


def latest_per_post(snaps: list[dict]) -> list[dict]:
    """The newest snapshot of each post (the log is append-only, so the last line of a uri wins)."""
    by_uri: dict[str, dict] = {}
    for s in snaps:
        by_uri[s["uri"]] = s
    return list(by_uri.values())


def score(s: dict) -> int:
    return sum(int(s.get(k) or 0) for k in COUNTS)


def aggregate(snaps: list[dict], key: Callable[[dict], str | None]) -> dict[str, dict]:
    """{group: {posts, mean/median likes, mean reposts, mean replies, mean_score, best, worst}} over the snapshots `key` puts in a group
    (a snapshot whose key is None is left out). best/worst: the draft id and score of the highest and lowest scoring post."""
    groups: dict[str, list[dict]] = {}
    for s in snaps:
        k = key(s)
        if k is not None:
            groups.setdefault(str(k), []).append(s)
    out = {}
    for k, rows in groups.items():
        likes = [int(r.get("likeCount") or 0) for r in rows]
        ranked = sorted(rows, key=score)
        out[k] = {"posts": len(rows), "mean_likes": round(float(statistics.mean(likes)), 2), "median_likes": statistics.median(likes),
                  "mean_reposts": round(float(statistics.mean(int(r.get("repostCount") or 0) for r in rows)), 2),
                  "mean_replies": round(float(statistics.mean(int(r.get("replyCount") or 0) for r in rows)), 2),
                  "mean_score": round(float(statistics.mean(score(r) for r in rows)), 2),
                  "best": {"draft_id": ranked[-1]["draft_id"], "score": score(ranked[-1])},
                  "worst": {"draft_id": ranked[0]["draft_id"], "score": score(ranked[0])}}
    return out


def _lineage(entry: dict | None, rank: str) -> str | None:
    for node in ((entry or {}).get("taxon") or {}).get("lineage") or []:
        if node.get("rank") == rank:
            return node.get("name")
    return None


def taxonomy_keys(ledger: Ledger) -> dict[str, int | None]:
    """{draft_id: lead observation id} so a snapshot can be looked up in the taxonomy cache."""
    out = {}
    for draft_id, d in ledger.drafts().items():
        ids = d.get("obs_ids") or [d.get("obs_id")]
        out[draft_id] = ids[0]
    return out


def load_taxonomy_cache(path: Path | None = None) -> dict:
    """The taxonomy cache as it is on disk ({} when there is none); never fetches."""
    path = Path(path or settings.OBS_TAXONOMY_FILE)
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def engagement_by_theme(days: int = DEFAULT_DAYS, *, log: Path | None = None, now: dt.datetime | None = None) -> dict[str, dict]:
    """{theme: aggregate row} over the newest snapshot of each post taken within the last `days` days ("single" for the one-photo posts).
    The digest builder (social_alerts) imports this."""
    now = now or dt.datetime.now(settings.LOCAL_TZ)
    cutoff = now - dt.timedelta(days=days)
    snaps = [s for s in read_snapshots(log) if dt.datetime.fromisoformat(s["at"]) >= cutoff]
    return aggregate(latest_per_post(snaps), lambda s: s.get("theme") or "single")


def build_report(snaps: list[dict], *, ledger: Ledger | None = None, taxonomy: dict | None = None) -> dict:
    """{"posts", "by_theme", "by_weekday", "by_hour", "by_kingdom" | None, "by_class" | None, "suggestions"}."""
    posts = latest_per_post(snaps)
    report = {"posts": len(posts), "by_theme": aggregate(posts, lambda s: s.get("theme") or "single"),
              "by_weekday": aggregate(posts, lambda s: s.get("weekday")), "by_hour": aggregate(posts, lambda s: s.get("hour")),
              "by_kingdom": None, "by_class": None}
    if taxonomy:
        lead = taxonomy_keys(ledger or Ledger())

        def entry(s):
            return taxonomy.get(str(lead.get(s["draft_id"])))
        report["by_kingdom"] = aggregate(posts, lambda s: _lineage(entry(s), "kingdom"))
        report["by_class"] = aggregate(posts, lambda s: _lineage(entry(s), "class"))
    report["suggestions"] = suggestions(report)
    return report


def suggestions(report: dict) -> list[str]:
    """A few plain sentences; a group needs MIN_POSTS posts before it can be named, and two such groups before one can beat another."""
    out = []
    for label, field, fmt in (("theme", "by_theme", str), ("weekday", "by_weekday", str), ("posting hour", "by_hour", lambda h: f"{int(h):02d}:00")):
        rows = {k: v for k, v in (report.get(field) or {}).items() if v["posts"] >= MIN_POSTS}
        if len(rows) < 2:
            continue
        best = max(rows, key=lambda k: (rows[k]["mean_score"], -rows[k]["posts"]))
        worst = min(rows, key=lambda k: (rows[k]["mean_score"], rows[k]["posts"]))
        if rows[best]["mean_score"] == rows[worst]["mean_score"]:
            continue
        out.append(f"Best {label}: {fmt(best)} averages {rows[best]['mean_score']} engagements over {rows[best]['posts']} posts "
                   f"(weakest: {fmt(worst)} at {rows[worst]['mean_score']} over {rows[worst]['posts']}).")
    if not out:
        out.append(f"Too few posts to suggest anything yet: a theme, day or hour needs {MIN_POSTS} posts, and two of them, to be compared.")
    return out


def format_report(report: dict) -> str:
    lines = [f"Engagement over {report['posts']} posts (score = likes + reposts + replies + quotes)"]
    for title, field in (("theme", "by_theme"), ("weekday", "by_weekday"), ("hour", "by_hour"), ("kingdom", "by_kingdom"), ("class", "by_class")):
        rows = report.get(field)
        if not rows:
            continue
        lines += ["", f"By {title}:"]
        order = sorted(rows, key=(lambda k: WEEKDAYS.index(k)) if field == "by_weekday" else (lambda k: int(k)) if field == "by_hour" else (lambda k: -rows[k]["mean_score"]))
        for k in order:
            r = rows[k]
            lines.append(f"  {k:<24} posts {r['posts']:>3}  likes mean {r['mean_likes']} median {r['median_likes']}  reposts {r['mean_reposts']}  "
                         f"replies {r['mean_replies']}  best {r['best']['draft_id']} ({r['best']['score']})  worst {r['worst']['draft_id']} ({r['worst']['score']})")
    lines += ["", "Suggestions:"] + [f"  - {s}" for s in report["suggestions"]]
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------------- the command

def cmd_social_engagement(args) -> int:
    days = args.days
    if days < 1:
        raise EngagementError(f"--days must be at least 1, not {days}")
    if args.refresh or not args.report:
        result = snapshot(days=days)
        print(f"{result['posts']} posts in the last {days} days; {result['snapshots']} snapshots stored in {settings.SOCIAL_ENGAGEMENT_LOG}")
        for uri in result["missing"]:
            print(f"  not returned by the AppView (deleted or hidden): {uri}")
    if args.report:
        now = dt.datetime.now(settings.LOCAL_TZ)
        cutoff = now - dt.timedelta(days=days)
        snaps = [s for s in read_snapshots() if dt.datetime.fromisoformat(s["at"]) >= cutoff]
        print(format_report(build_report(snaps, taxonomy=load_taxonomy_cache())))
    return 0


def add_parsers(sub) -> None:
    p = sub.add_parser("social-engagement", help="Fetch public like/repost/reply counts of the recent Bluesky posts and report which themes, days and hours work. "
                                                 "Read-only against the public AppView; stores counts only.")
    p.add_argument("--days", type=int, default=DEFAULT_DAYS, help="posts of the last N days (default 30)")
    p.add_argument("--refresh", action="store_true", help="fetch fresh counts (the default unless only --report is given)")
    p.add_argument("--report", action="store_true", help="print the aggregates and suggestions from the stored snapshots")


COMMANDS = {"social-engagement": cmd_social_engagement}
