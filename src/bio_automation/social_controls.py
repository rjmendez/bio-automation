"""Owner controls for the scheduled Bluesky posts: a pause switch and a circuit breaker (rules pause_switch and circuit_breaker).

Pause: `bio social-pause [--reason TEXT]` writes settings.SOCIAL_PAUSE_FILE; while it exists every scheduled post is blocked (a manual post is
not: the pause is for the automation). `bio social-resume` removes it. A pause file that cannot be read still pauses (fail closed).
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

def read_pause(path: Path | None = None) -> dict | None:
    """None when not paused, else {"at", "reason"}; a file that is not readable JSON still pauses, with `unreadable` set."""
    path = Path(path or settings.SOCIAL_PAUSE_FILE)
    if not path.exists():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(doc, dict):
            raise ValueError("not a JSON object")
    except (OSError, ValueError) as e:                          # JSONDecodeError and UnicodeDecodeError are ValueErrors
        return {"at": None, "reason": f"{path} is unreadable ({type(e).__name__}: {e}); remove it with `bio social-resume`", "unreadable": True}
    return {"at": doc.get("at"), "reason": str(doc.get("reason") or "")}


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
                    unreadable=bool(state.get("unreadable")), live=ctx.get("live"))
    return Verdict("block", reason)


def cmd_social_pause(args) -> int:
    path = Path(settings.SOCIAL_PAUSE_FILE)
    if path.exists() and not args.force:
        state = read_pause(path)
        raise ControlsError(f"already paused ({state['reason'] or 'no reason given'}{', since ' + state['at'] if state['at'] else ''}); "
                            "`bio social-resume` first, or use --force to replace the pause")
    at = _now().isoformat(timespec="seconds")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"at": at, "reason": args.reason or ""}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Audit().log("paused", action="set", reason=args.reason or "", forced=bool(args.force))
    print(f"paused: {path}" + (f" ({args.reason})" if args.reason else ""))
    return 0


def cmd_social_resume(args) -> int:
    path = Path(settings.SOCIAL_PAUSE_FILE)
    if not path.exists():
        print("not paused")
        return 0
    path.unlink()
    Audit().log("paused", action="cleared")
    print(f"resumed: {path} removed")
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
            with contextlib.suppress(OSError):
                Audit().log("breaker_probe", draft_id=ctx.get("draft_id"), count=state["count"], limit=limit, cooldown_h=state["cooldown_h"])
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
    sub.add_parser("social-resume", help="Lift the pause: scheduled posts go out again.")
    p = sub.add_parser("social-breaker", help="Show the circuit breaker (failed scheduled posts in a row); --reset closes it.")
    p.add_argument("--reset", action="store_true", help="forget the failures so far (audited)")


COMMANDS = {"social-pause": cmd_social_pause, "social-resume": cmd_social_resume, "social-breaker": cmd_social_breaker}
