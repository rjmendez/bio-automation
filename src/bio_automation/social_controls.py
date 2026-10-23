"""Owner controls for the scheduled Bluesky posts: a pause switch and a circuit breaker (rules pause_switch and circuit_breaker).

Pause: `bio social-pause [--reason TEXT]` writes settings.SOCIAL_PAUSE_FILE; while it exists every scheduled post is blocked (a manual post is
not: the pause is for the automation). `bio social-resume` removes it. A pause file that cannot be read still pauses (fail closed).
Remote pause (off by default): set BIO_SOCIAL_PAUSE_SYNC_FILE to a file in a synced folder; creating it from any device (any content; its first
line is the reason) pauses, deleting it resumes, and `social-pause --sync` / `social-resume` write / remove it from here. Only the pre_send hook
blocks: prep (drafting) and the watch/alert scans keep running while paused, because they publish nothing.
Breaker: from the audit log, the scheduled posts that ended in post_failed since the last posted or breaker_reset record; at N (the rule's number)
scheduled posts are blocked until `bio social-breaker --reset` or a successful post (a manual one counts). It also closes itself: once
settings.SOCIAL_BREAKER_COOLDOWN_H hours have passed since the last failure (or the last probe), ONE live scheduled post is let through as a probe
(audit breaker_probe); a posted record closes the breaker, a failure re-opens it and restarts the cool-down. Only the tail of the log is read.

Audit events: paused (action set | cleared | blocked), circuit_open, breaker_probe, breaker_reset."""
from __future__ import annotations

import contextlib
import datetime as dt
import json
import os
from pathlib import Path
from typing import Iterator

from . import settings, social_hooks
from .social_hooks import Verdict
from .social_rules import Audit, load_rules

CHUNK = 64 * 1024
SHOWN_FAILURES = 5


class ControlsError(Exception):
    pass


ERRORS = (ControlsError,)


def _now() -> dt.datetime:
    return dt.datetime.now(settings.LOCAL_TZ)


# --------------------------------------------------------------------------------------------------- the pause switch

_SETTINGS = object()


def _read_marker(path: Path, *, sync: bool) -> dict:
    """{"at", "reason", "source", ["unreadable"]} of a marker that exists. The local file is JSON; the synced one may also be an empty file or
    plain text (what a phone's file app creates): any content pauses, its first line is the reason. Unreadable content still pauses."""
    source = "sync" if sync else "local"
    try:
        raw = path.read_text(encoding="utf-8-sig")
        try:
            doc = json.loads(raw)
        except ValueError:
            if not sync:
                raise
            return {"at": None, "reason": (raw.strip().splitlines() or [""])[0][:200], "source": source}
        if not isinstance(doc, dict):
            if sync:
                return {"at": None, "reason": "", "source": source}
            raise ValueError("not a JSON object")
    except (OSError, ValueError) as e:                          # JSONDecodeError and UnicodeDecodeError are ValueErrors
        return {"at": None, "source": source, "unreadable": True,
                "reason": f"{path} is unreadable ({type(e).__name__}: {e}); remove it with `bio social-resume`"}
    return {"at": doc.get("at"), "reason": str(doc.get("reason") or ""), "source": source}


def sync_marker_path(sync_path=_SETTINGS) -> Path | None:
    path = settings.SOCIAL_PAUSE_SYNC_FILE if sync_path is _SETTINGS else sync_path
    return Path(path) if path else None


def read_pause(path: Path | None = None, sync_path=_SETTINGS) -> dict | None:
    """None when not paused, else {"at", "reason", "source": "local" | "sync"}; the local file wins when both exist. A marker that is not readable
    still pauses, with `unreadable` set. The synced marker counts only when settings.SOCIAL_PAUSE_SYNC_FILE (or `sync_path`) is configured."""
    path = Path(path or settings.SOCIAL_PAUSE_FILE)
    if path.exists():
        return _read_marker(path, sync=False)
    remote = sync_marker_path(sync_path)
    if remote is not None and remote.exists():
        return _read_marker(remote, sync=True)
    return None


def pause_hook(ctx: dict) -> Verdict | None:
    if ctx.get("manual"):
        return None
    state = read_pause()
    if state is None:
        return None
    since = f" since {state['at']}" if state.get("at") else ""
    why = f" ({state['reason']})" if state.get("reason") else ""
    reason = f"posting is paused{since}{why}; `bio social-resume` lifts it"
    with contextlib.suppress(OSError):
        Audit().log("paused", action="blocked", draft_id=ctx.get("draft_id"), reason=state["reason"], since=state.get("at"),
                    unreadable=bool(state.get("unreadable")), source=state.get("source"), live=ctx.get("live"))
    return Verdict("block", reason)


def cmd_social_pause(args) -> int:
    path = Path(settings.SOCIAL_PAUSE_FILE)
    remote = sync_marker_path()
    if args.sync and remote is None:
        raise ControlsError("--sync needs BIO_SOCIAL_PAUSE_SYNC_FILE (the synced pause marker) to be set")
    if path.exists() and not args.force:
        state = read_pause(path)
        raise ControlsError(f"already paused ({state['reason'] or 'no reason given'}{', since ' + state['at'] if state['at'] else ''}); "
                            "`bio social-resume` first, or use --force to replace the pause")
    at = _now().isoformat(timespec="seconds")
    body = json.dumps({"at": at, "reason": args.reason or ""}, ensure_ascii=False, indent=2) + "\n"
    written = [path] + ([remote] if args.sync else [])
    for target in written:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
    Audit().log("paused", action="set", reason=args.reason or "", forced=bool(args.force), files=[str(t) for t in written])
    print(f"paused: {', '.join(str(t) for t in written)}" + (f" ({args.reason})" if args.reason else ""))
    return 0


def cmd_social_resume(args) -> int:
    """Removes the local marker and, when configured, the synced one (a remote pause that survived a resume would leave the posts blocked)."""
    targets = [p for p in (Path(settings.SOCIAL_PAUSE_FILE), sync_marker_path()) if p is not None and p.exists()]
    if not targets:
        print("not paused")
        return 0
    for p in targets:
        p.unlink()
    Audit().log("paused", action="cleared", files=[str(p) for p in targets])
    print("resumed: " + ", ".join(f"{p} removed" for p in targets))
    return 0


# --------------------------------------------------------------------------------------------------- the circuit breaker

def _reverse_lines(path: Path) -> Iterator[str]:
    """The file's lines, last first, reading CHUNK bytes at a time from the end."""
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        pos = f.tell()
        tail = b""
        while pos > 0:
            step = min(CHUNK, pos)
            pos -= step
            f.seek(pos)
            parts = (f.read(step) + tail).split(b"\n")
            tail = parts[0]                                      # may be the end of a line that began in an earlier chunk
            for raw in reversed(parts[1:]):
                if raw.strip():
                    yield raw.decode("utf-8", errors="replace")
        if tail.strip():
            yield tail.decode("utf-8", errors="replace")


def _parse_at(value) -> dt.datetime | None:
    try:
        when = dt.datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return when if when.tzinfo else when.replace(tzinfo=settings.LOCAL_TZ)


def breaker_state(audit_path: Path | None = None, limit: int | None = None, *, now: dt.datetime | None = None,
                  cooldown_h: float | None = None) -> dict:
    """{"count", "failures" (newest first, at most SHOWN_FAILURES), "limit", "open", "cooldown_h", "next_probe_at", "probe_due"}: scheduled
    post_failed records since the last posted or breaker_reset record. A failure of a manual post (its post_decision says manual) is not
    counted. While open, next_probe_at is the cool-down's end after the newest failure or probe, and probe_due says `now` has reached it."""
    path = Path(audit_path or settings.SOCIAL_AUDIT_LOG)
    cooldown_h = settings.SOCIAL_BREAKER_COOLDOWN_H if cooldown_h is None else cooldown_h
    now = now or _now()
    failures: list[dict] = []
    manual_runs: set[tuple] = set()
    last_probe: str | None = None
    if path.is_file():
        for line in _reverse_lines(path):
            try:
                rec = json.loads(line)
            except ValueError:
                continue                                         # a torn line is not a boundary
            if not isinstance(rec, dict):
                continue
            event = rec.get("event")
            if event in ("posted", "breaker_reset"):
                break
            if event == "post_failed":
                failures.append(rec)
            elif event == "breaker_probe" and last_probe is None:
                last_probe = rec.get("at")
            elif event == "post_decision" and rec.get("manual"):
                manual_runs.add((rec.get("run"), rec.get("draft_id")))
    scheduled = [r for r in failures if (r.get("run"), r.get("draft_id")) not in manual_runs]
    is_open = limit is not None and len(scheduled) >= limit
    next_probe = None
    if is_open:
        marks = [m for m in (_parse_at(scheduled[0].get("at")), _parse_at(last_probe)) if m is not None]
        if marks:
            next_probe = max(marks) + dt.timedelta(hours=cooldown_h)
    return {"count": len(scheduled), "limit": limit,
            "failures": [{"at": r.get("at"), "draft_id": r.get("draft_id"), "outcome": r.get("outcome"), "error": r.get("error")}
                         for r in scheduled[:SHOWN_FAILURES]],
            "open": is_open, "cooldown_h": cooldown_h, "next_probe_at": next_probe.isoformat(timespec="seconds") if next_probe else None,
            "probe_due": bool(next_probe and now >= next_probe)}


def breaker_hook(ctx: dict) -> Verdict | None:
    if ctx.get("manual"):
        return None
    limit = ctx["rules"].value("circuit_breaker")
    now = ctx.get("now")
    state = breaker_state(limit=limit, now=now if isinstance(now, dt.datetime) else None)
    if not state["open"]:
        return None
    if state["probe_due"]:
        if ctx.get("live"):
            draft_id, count, cooldown_h, audit = ctx.get("draft_id"), state["count"], state["cooldown_h"], ctx.get("audit") or Audit()

            def spend_probe() -> None:
                with contextlib.suppress(OSError):
                    audit.log("breaker_probe", draft_id=draft_id, count=count, limit=limit, cooldown_h=cooldown_h)

            ctx.setdefault("on_send", []).append(spend_probe)       # spent when a send actually starts, not here: a later hook may still withdraw this draft
        return None                                              # one probe: success closes the breaker, a failure restarts the cool-down
    after = f"; the next probe is allowed after {state['next_probe_at']}" if state["next_probe_at"] else ""
    reason = (f"circuit breaker open: {state['count']} scheduled posts failed in a row (limit {limit}){after}; "
              "`bio social-breaker --reset` closes it")
    with contextlib.suppress(OSError):
        Audit().log("circuit_open", draft_id=ctx.get("draft_id"), count=state["count"], limit=limit, failures=state["failures"], live=ctx.get("live"),
                    next_probe_at=state["next_probe_at"])
    return Verdict("block", reason)


def cmd_social_breaker(args) -> int:
    rules = load_rules()
    limit = rules.value("circuit_breaker")
    state = breaker_state(limit=limit)
    print(f"circuit breaker: {'OPEN' if state['open'] else 'closed'}; {state['count']} failed scheduled posts in a row (limit {limit})"
          + ("" if rules.on("circuit_breaker") else "; the rule circuit_breaker is OFF, so nothing is blocked"))
    if state["open"]:
        print(f"  next probe: {state['next_probe_at'] or 'unknown'} (cool-down {state['cooldown_h']:g} h)"
              + ("; due now: the next scheduled post goes out as a probe" if state["probe_due"] else ""))
    for f in state["failures"]:
        print(f"  {f['at']}  {f['draft_id']}  {f['outcome']}  {(f['error'] or '')[:120]}")
    if args.reset:
        Audit().log("breaker_reset", count=state["count"], limit=limit)
        print("breaker reset: failures before now no longer count")
    return 0


# --------------------------------------------------------------------------------------------------- wiring

def register_hooks() -> None:
    social_hooks.register("pause_switch", "pre_send", pause_hook, rule="pause_switch", priority=1)
    social_hooks.register("circuit_breaker", "pre_send", breaker_hook, rule="circuit_breaker", priority=5)


def add_parsers(sub) -> None:
    p = sub.add_parser("social-pause", help="Pause every scheduled Bluesky post (manual posts still work) until `bio social-resume`.")
    p.add_argument("--reason", default="", help="why, shown when a run is blocked")
    p.add_argument("--force", action="store_true", help="replace an existing pause")
    p.add_argument("--sync", action="store_true", help="also write the synced marker (BIO_SOCIAL_PAUSE_SYNC_FILE) so other devices see the pause")
    sub.add_parser("social-resume", help="Lift the pause (the local marker and the synced one): scheduled posts go out again.")
    p = sub.add_parser("social-breaker", help="Show the circuit breaker (failed scheduled posts in a row); --reset closes it.")
    p.add_argument("--reset", action="store_true", help="forget the failures so far (audited)")


COMMANDS = {"social-pause": cmd_social_pause, "social-resume": cmd_social_resume, "social-breaker": cmd_social_breaker}
