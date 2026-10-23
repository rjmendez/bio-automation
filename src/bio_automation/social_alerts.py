"""Alerts and the daily digest for the Bluesky posting workflow (rule: alerts).

A PULL model: nothing here sits in the posting path. `bio social-alerts` reads the audit log, the ledger, the pause file, the social.log mtime,
the theme taxonomy cache and the disks, decides what needs the owner's attention, appends each NEW alert to settings.SOCIAL_ALERTS_LOG and,
when asked (--notify, or settings.SOCIAL_ALERT_COMMAND is set) and the rule `alerts` is on, runs the alert command once per new alert.
`bio social-digest` writes one Markdown page per day to settings.SOCIAL_DIGEST_DIR.

Alert kinds and levels (exit code: 0 none, 1 warnings, 2 errors):
  error    post_failed, post_verify_failed, circuit_open, ledger_corrupt, expected_post_missing, run_silent, disk (C: under 15 GB)
  warning  post_blocked, hook_error, paused, backup_failed, low_stock, queue_empty, disk (D: under 20 GB)
De-duplication: a key already in the alerts log less than 12 hours ago is not new. An alert that stands for ONE audit record (an event: post_failed,
hook_error ...) is new only once, ever, so an old failure still inside the --since window is not announced again after 12 hours.
LOW STOCK counts posts as unclaimed matching observations in the taxonomy cache // the theme's max_photos; "unclaimed" = in no draft that is
not withdrawn. It is an estimate: the cache holds the observations seen so far, not the gallery index.
Every function takes its inputs (paths, clock, runner) as arguments; the defaults come from settings when the command runs."""
from __future__ import annotations

import datetime as dt
import json
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, NamedTuple

from . import settings
from .social_rules import Rules, load_rules

DEDUP_WINDOW = dt.timedelta(hours=12)
LATE_AFTER = dt.timedelta(minutes=60)          # a slot's post is missing when the slot is this old
SILENT_AFTER = dt.timedelta(minutes=30)        # a slot's run left no trace when the slot is this old
LOW_STOCK_POSTS = 10
DISK_MIN_GB = (("C:\\", 15, "error"), ("D:\\", 20, "warning"))
NOTIFY_TIMEOUT_S = 30
ERROR_EVENTS = {"post_failed": "error", "post_verify_failed": "error", "circuit_open": "error", "ledger_corrupt": "error",
                "post_blocked": "warning", "hook_error": "warning", "paused": "warning", "backup_failed": "warning"}
PER_DAY_EVENTS = ("paused", "circuit_open")    # these repeat on every run: one alert a day
LEVELS = {"warning": 1, "error": 2}


class AlertsError(Exception):
    pass


class Alert(NamedTuple):
    key: str
    kind: str
    level: str                                 # warning | error
    message: str
    sticky: bool = False                       # stands for one audit record: announced once, ever


# --------------------------------------------------------------------------------------------------- reading

def social_log_path() -> Path:
    return settings.CACHE_DIR / "logs" / "social.log"


def read_jsonl(path: Path) -> list[dict]:
    """The JSON object lines of `path`; a missing file is empty and a damaged line is skipped (a reader of logs must survive a bad line)."""
    path = Path(path)
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


def parse_time(text, tz=None) -> dt.datetime | None:
    try:
        t = dt.datetime.fromisoformat(str(text))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=tz or settings.LOCAL_TZ)


def parse_since(text: str) -> dt.timedelta:
    m = re.fullmatch(r"\s*(\d+)\s*([mhd])\s*", text or "")
    if not m or int(m.group(1)) < 1:
        raise AlertsError(f"--since {text!r}: expected a number and m, h or d (30m, 24h, 2d)")
    return dt.timedelta(**{{"m": "minutes", "h": "hours", "d": "days"}[m.group(2)]: int(m.group(1))})


def parse_slots(text: str) -> list[dt.time]:
    out = []
    for part in (text or "").split(","):
        if part.strip():
            try:
                out.append(dt.time.fromisoformat(part.strip()))
            except ValueError as e:
                raise AlertsError(f"BIO_SOCIAL_SLOTS {text!r}: {part.strip()!r} is not HH:MM") from e
    return sorted(out)


def _drafts(ledger: list[dict]) -> dict[str, dict]:
    """{draft_id: {state, obs_ids, month, for_date, theme}} from ledger records, the way social.Ledger.drafts() derives it (latest drafted wins)."""
    out: dict[str, dict] = {}
    for r in ledger:
        if r.get("event") == "error" or "draft_id" not in r:
            continue
        d = out.setdefault(r["draft_id"], {})
        if r.get("event") == "drafted":
            d.update(obs_ids=list(r.get("obs_ids") or ([r["obs_id"]] if r.get("obs_id") is not None else [])), month=r.get("month"),
                     for_date=r.get("for_date"), theme=r.get("theme"), images=r.get("images"))
        d["state"] = r.get("event")
    return out


# --------------------------------------------------------------------------------------------------- the pool

def pool_left(taxonomy: dict, drafts: dict[str, dict], themes_list: list) -> dict[str, dict]:
    """{theme key: {"name", "observations", "posts"}}: unclaimed, non-casual observations of the cache that belong to each enabled theme."""
    from . import themes as themes_mod

    claimed = {int(o) for d in drafts.values() if d.get("state") not in (None, "withdrawn") for o in d.get("obs_ids") or []}
    out = {}
    for t in themes_list:
        if not t.enabled:
            continue
        n = sum(1 for k, e in taxonomy.items()
                if isinstance(e, dict) and str(k).isdigit() and int(k) not in claimed and e.get("grade") != "casual" and themes_mod.matches(t, e))
        out[t.key] = {"name": t.name, "observations": n, "posts": n // max(t.max_photos, 1)}
    return out


def _load_pool(taxonomy_file: Path, themes_file, drafts: dict[str, dict]) -> tuple[dict[str, dict] | None, str]:
    """(pool, "") or (None, why it could not be counted)."""
    from . import themes as themes_mod

    if not Path(taxonomy_file).is_file():
        return None, f"taxonomy cache {taxonomy_file} is missing"
    try:
        taxonomy = json.loads(Path(taxonomy_file).read_text(encoding="utf-8"))
        themes_list = themes_mod.load_themes(themes_file) if themes_file else themes_mod.load_themes()
    except (ValueError, OSError, themes_mod.ThemeError) as e:
        return None, f"taxonomy or themes unreadable: {e}"
    return pool_left(taxonomy, drafts, themes_list), ""


# --------------------------------------------------------------------------------------------------- the scan

def scan(now: dt.datetime, *, since: dt.timedelta = dt.timedelta(hours=24), audit_path: Path | None = None, ledger_path: Path | None = None,
         rules: Rules | None = None, slots: list[dt.time] | None = None, pause_file: Path | None = None, log_file: Path | None = None,
         taxonomy_file: Path | None = None, themes_file: Path | None = None,
         disk_free: Callable[[str], int] = lambda p: shutil.disk_usage(p).free) -> tuple[list[Alert], list[str]]:
    """(alerts, notes). Notes say what could not be checked (a missing cache, an unreadable disk); they are not alerts."""
    rules = rules or load_rules()
    audit = read_jsonl(audit_path or settings.SOCIAL_AUDIT_LOG)
    ledger = read_jsonl(ledger_path or settings.SOCIAL_LEDGER)
    slots = parse_slots(settings.SOCIAL_SLOTS) if slots is None else slots
    pause_file = Path(pause_file or settings.SOCIAL_PAUSE_FILE)
    log_file = Path(log_file or social_log_path())
    tz = now.tzinfo
    alerts: dict[str, Alert] = {}
    notes: list[str] = []

    def add(a: Alert) -> None:
        alerts[a.key] = a

    # 1. events in the window
    for r in audit:
        kind, at = r.get("event"), parse_time(r.get("at"), tz)
        if kind not in ERROR_EVENTS or at is None or not (now - since <= at <= now):
            continue
        detail = r.get("reason") or r.get("error") or r.get("status") or ""
        who = r.get("draft_id") or r.get("name") or ""
        text = f"{kind}{' ' + str(who) if who else ''}{': ' + str(detail)[:200] if detail else ''} ({r.get('at')})"
        key = f"{kind}:{at.date().isoformat()}" if kind in PER_DAY_EVENTS else f"{kind}:{who}:{r.get('at')}"
        add(Alert(key, kind, ERROR_EVENTS[kind], text, sticky=True))

    drafts = _drafts(ledger)
    today = now.date()
    cap = rules.value("daily_cap") if rules.on("daily_cap") else None
    manual_ids = {r.get("draft_id") for r in audit if r.get("event") == "posted" and r.get("manual")}
    scheduled = [r for r in ledger if r.get("event") == "posted" and str(r.get("at", ""))[:10] == today.isoformat()
                 and not r.get("manual") and r.get("draft_id") not in manual_ids]
    paused = pause_file.exists() and rules.on("pause_switch")

    def eligible(d: dict) -> bool:
        if d.get("state") not in ("drafted", "approved"):
            return False
        if rules.on("month_posting") and d.get("month") != f"{today.year}-{today.month:02d}":
            return False
        if d.get("for_date") and rules.on("theme_day_only") and d["for_date"] != today.isoformat():
            return False
        return True

    # 2. a slot's post is missing
    due = [s for s in slots if now >= dt.datetime.combine(today, s, tzinfo=tz) + LATE_AFTER]
    if due and not paused and len(scheduled) < len(due) and (cap is None or len(scheduled) < cap) and any(eligible(d) for d in drafts.values()):
        add(Alert(f"expected_post_missing:{today.isoformat()}:{len(due)}", "expected_post_missing", "error",
                  f"{len(scheduled)} scheduled post(s) today, {len(due)} slot(s) passed by 60+ minutes (last {due[-1].strftime('%H:%M')}), "
                  f"cap not reached, an eligible draft is waiting"))

    # 3. a run left no trace
    try:
        log_at = dt.datetime.fromtimestamp(log_file.stat().st_mtime, tz)
    except OSError:
        log_at = None
    for i, s in enumerate(slots):
        start = dt.datetime.combine(today, s, tzinfo=tz)
        end = dt.datetime.combine(today, slots[i + 1], tzinfo=tz) if i + 1 < len(slots) else start + dt.timedelta(days=1)
        if now < start + SILENT_AFTER or start < now - since:
            continue
        seen = any((t := parse_time(r.get("at"), tz)) is not None and start <= t < end for r in audit) or (log_at is not None and start <= log_at < end)
        if not seen:
            add(Alert(f"run_silent:{today.isoformat()}:{s.strftime('%H:%M')}", "run_silent", "error",
                      f"the {s.strftime('%H:%M')} run left no audit record and no social.log write by {now.strftime('%H:%M')}"))

    # 4. stock
    pool, why = _load_pool(taxonomy_file or settings.OBS_TAXONOMY_FILE, themes_file, drafts)
    if pool is None:
        notes.append(f"low stock not checked: {why}")
    else:
        for key, p in sorted(pool.items()):
            if p["posts"] < LOW_STOCK_POSTS:
                add(Alert(f"low_stock:{key}", "low_stock", "warning",
                          f"{p['name']}: {p['posts']} post(s) left ({p['observations']} unclaimed observations), under {LOW_STOCK_POSTS}"))
    if not any(d.get("state") in ("drafted", "approved") for d in drafts.values()):
        add(Alert(f"queue_empty:{today.isoformat()}", "queue_empty", "warning", "no drafted or approved draft is left to post"))

    # 5. disks
    for root, gb, level in DISK_MIN_GB:
        try:
            free = disk_free(root) / 1024 ** 3
        except OSError as e:
            notes.append(f"disk {root} not checked: {e}")
            continue
        if free < gb:
            add(Alert(f"disk:{root[0]}", "disk", level, f"{root} has {free:.1f} GB free, under {gb} GB"))
    return sorted(alerts.values(), key=lambda a: (-LEVELS[a.level], a.key)), notes


def exit_code(alerts: list[Alert]) -> int:
    return max((LEVELS[a.level] for a in alerts), default=0)


# --------------------------------------------------------------------------------------------------- the alerts log and notify

def new_alerts(alerts: list[Alert], log: list[dict], now: dt.datetime) -> list[Alert]:
    """The alerts not announced yet: not in the log within DEDUP_WINDOW (age under 12 h; at exactly 12 h it is new again), and for a sticky alert
    not in the log at all."""
    seen: dict[str, dt.datetime] = {}
    for r in log:
        if r.get("record", "alert") == "alert" and r.get("key") and (t := parse_time(r.get("at"), now.tzinfo)) is not None:
            seen[r["key"]] = max(t, seen.get(r["key"], t))
    out = []
    for a in alerts:
        t = seen.get(a.key)
        if t is None or (not a.sticky and now - t >= DEDUP_WINDOW):
            out.append(a)
    return out


def _append(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, sort_keys=True, ensure_ascii=False) + "\n")


def split_command(text: str) -> list[str]:
    """The alert command as an argument list (shlex, no shell). Windows paths keep their backslashes; surrounding quotes are removed."""
    parts = shlex.split(text, posix=False)
    return [p[1:-1] if len(p) >= 2 and p[0] == p[-1] and p[0] in "\"'" else p for p in parts]


def run_alerts(now: dt.datetime, *, since: dt.timedelta = dt.timedelta(hours=24), notify: bool = False, command: str | None = None,
               rules: Rules | None = None, alerts_log: Path | None = None, runner=subprocess.run, **scan_args) -> dict:
    """Scan, record the new alerts, notify. Returns {"alerts", "new", "notified", "notify_failed", "notes", "exit"}."""
    rules = rules or load_rules()
    alerts_log = Path(alerts_log or settings.SOCIAL_ALERTS_LOG)
    command = settings.SOCIAL_ALERT_COMMAND if command is None else command
    found, notes = scan(now, since=since, rules=rules, **scan_args)
    fresh = new_alerts(found, read_jsonl(alerts_log), now)
    stamp = now.isoformat(timespec="seconds")
    report = {"alerts": found, "new": fresh, "notified": [], "notify_failed": [], "notes": notes, "exit": exit_code(found)}
    for a in fresh:
        _append(alerts_log, {"at": stamp, "record": "alert", "key": a.key, "kind": a.kind, "level": a.level, "message": a.message})
    if not (notify or command.strip()):
        return report
    if not rules.on("alerts"):
        notes.append("rule alerts is off: nothing was sent to the alert command")
        return report
    if not command.strip():
        notes.append("--notify given but no alert command is set (BIO_SOCIAL_ALERT_COMMAND)")
        return report
    for a in fresh:
        try:
            argv = split_command(command)
            res = runner(argv, input=f"[{a.level}] {a.kind}: {a.message}\n", text=True, capture_output=True, timeout=NOTIFY_TIMEOUT_S, shell=False)
            ok, error = res.returncode == 0, f"exit code {res.returncode}: {(res.stderr or '').strip()[:200]}"
        except (OSError, subprocess.SubprocessError, ValueError) as e:
            ok, error = False, f"{type(e).__name__}: {e}"[:300]
        _append(alerts_log, {"at": stamp, "record": "notified" if ok else "notify_failed", "key": a.key, **({} if ok else {"error": error})})
        (report["notified"] if ok else report["notify_failed"]).append(a)
    return report


# --------------------------------------------------------------------------------------------------- the digest

SKIP_EVENTS = ("post_skipped", "post_blocked", "post_failed", "withdrawn", "hook", "gate_rejected", "draft_skipped", "paused", "circuit_open",
               "post_verify_failed", "theme_short", "live_recheck_unavailable")


def _why(r: dict) -> str:
    for k in ("reason", "status", "error", "outcome"):
        if r.get(k):
            return str(r[k])
    return ""


def _engagement(path: Path, uris: dict[str, str]) -> list[str]:
    """Lines for the engagement section, tolerant: any readable line with a uri of today's posts counts; the latest line per uri wins."""
    latest: dict[str, dict] = {}
    for r in read_jsonl(path):
        if r.get("uri") in uris:
            latest[r["uri"]] = r

    def num(r: dict, *names: str) -> int:
        for n in names:
            v = r.get(n)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return int(v)
        return 0

    out = [f"- {uris[u]}: {num(r, 'likes', 'like_count', 'likeCount')} likes, {num(r, 'reposts', 'repost_count', 'repostCount')} reposts, "
           f"{num(r, 'replies', 'reply_count', 'replyCount')} replies, {num(r, 'quotes', 'quote_count', 'quoteCount')} quotes" for u, r in sorted(latest.items(), key=lambda kv: uris[kv[0]])]
    return out or ["- no engagement records for today's posts yet"]


def _rules_line(snap: dict) -> str:
    on = [n + (f"({v['value']})" if "value" in v else "") for n, v in sorted(snap.items()) if v.get("on")]
    off = [n for n, v in sorted(snap.items()) if not v.get("on")]
    return f"- on: {', '.join(on) or 'none'}\n- off: {', '.join(off) or 'none'}"


def _last_snapshot(audit: list[dict], before: dt.datetime | None, tz) -> dict | None:
    snap = None
    for r in audit:
        t = parse_time(r.get("at"), tz)
        if isinstance(r.get("rules"), dict) and t is not None and (before is None or t < before):
            snap = r["rules"]
    return snap


def digest_text(date: dt.date, *, audit: list[dict], ledger: list[dict], alerts_log: list[dict], rules: Rules, pool: dict[str, dict] | None,
                pool_note: str = "", themes_list: list | None = None, engagement_file: Path | None = None, tz=None) -> str:
    """The digest page for `date`. Pure: everything it shows is an argument."""
    from . import themes as themes_mod

    tz = tz or settings.LOCAL_TZ
    day = date.isoformat()
    start = dt.datetime.combine(date, dt.time(), tzinfo=tz)
    end = start + dt.timedelta(days=1)

    def on_day(r: dict) -> bool:
        t = parse_time(r.get("at"), tz)
        return t is not None and start <= t < end

    def hhmm(r: dict) -> str:
        return parse_time(r.get("at"), tz).strftime("%H:%M")

    drafts = _drafts(ledger)
    decisions = {r.get("draft_id"): r for r in audit if r.get("event") == "post_decision" and on_day(r)}
    out = [f"# Bluesky digest {day}", "", "## Posted", ""]
    uris: dict[str, str] = {}
    posted = [r for r in ledger if r.get("event") == "posted" and on_day(r)]
    for r in posted:
        d = drafts.get(r["draft_id"], {})
        images = decisions.get(r["draft_id"], {}).get("images", d.get("images"))
        uris[r.get("uri")] = r["draft_id"]
        out.append(f"- {hhmm(r)} {r['draft_id']} | theme: {d.get('theme') or 'none'} | images: {images if images is not None else 'unknown'} | {r.get('uri')}")
    if not posted:
        out.append("- nothing was posted")
    out += ["", "## Not posted", ""]
    skipped = [r for r in audit if r.get("event") in SKIP_EVENTS and on_day(r)]
    for r in skipped:
        who = r.get("draft_id") or r.get("obs_id") or r.get("name") or ""
        out.append(f"- {hhmm(r)} {r['event']}{' ' + str(who) if who else ''}: {_why(r) or 'no reason recorded'}")
    if not skipped:
        out.append("- nothing was skipped, withdrawn or blocked")
    out += ["", "## Rules in force", "", _rules_line(rules.snapshot()), ""]
    before, after = _last_snapshot(audit, start, tz), _last_snapshot(audit, end, tz)
    if before is None or after is None:
        out.append("Changes since yesterday: no earlier audit record to compare with.")
    else:
        changes = [f"{n}: {_state(before.get(n))} -> {_state(after.get(n))}" for n in sorted(set(before) | set(after)) if before.get(n) != after.get(n)]
        out += ["Changes since yesterday:"] + ([f"- {c}" for c in changes] if changes else ["- none"])
    out += ["", "## Pool remaining", ""]
    if pool is None:
        out.append(f"- not counted: {pool_note}")
    else:
        out += [f"- {p['name']}: {p['posts']} posts ({p['observations']} observations)" for _, p in sorted(pool.items())] or ["- no enabled theme"]
    out += ["", "## Next three days", ""]
    for k in (1, 2, 3):
        nxt = date + dt.timedelta(days=k)
        theme = themes_mod.theme_for(nxt, themes_list) if themes_list is not None else None
        have = [(i, d["state"]) for i, d in drafts.items() if d.get("for_date") == nxt.isoformat() and d.get("state") != "withdrawn"]
        out.append(f"- {nxt.isoformat()} {nxt.strftime('%a')}: {theme.name if theme else 'no theme'}"
                   f"{' (draft ' + have[-1][0] + ': ' + have[-1][1] + ')' if have else ' (no draft yet)' if theme else ''}")
    out += ["", "## Active alerts", ""]
    seen: dict[str, dict] = {}
    for r in alerts_log:
        if r.get("record", "alert") == "alert" and on_day(r):
            seen[r["key"]] = r
    out += [f"- {r['level']} {r['kind']}: {r['message']}" for r in sorted(seen.values(), key=lambda r: (-LEVELS.get(r["level"], 0), r["key"]))] or ["- none"]
    if engagement_file is not None and Path(engagement_file).is_file():
        out += ["", "## Engagement", ""] + _engagement(Path(engagement_file), uris)
    return "\n".join(out) + "\n"


def _state(s) -> str:
    if s is None:
        return "absent"
    return ("on" + (f" ({s['value']})" if "value" in s else "")) if s.get("on") else "off"


def write_digest(date: dt.date, *, rules: Rules | None = None, digest_dir: Path | None = None, audit_path: Path | None = None,
                 ledger_path: Path | None = None, alerts_log: Path | None = None, taxonomy_file: Path | None = None, themes_file=None,
                 engagement_file: Path | None = None) -> Path:
    from . import themes as themes_mod

    rules = rules or load_rules()
    audit, ledger = read_jsonl(audit_path or settings.SOCIAL_AUDIT_LOG), read_jsonl(ledger_path or settings.SOCIAL_LEDGER)
    pool, why = _load_pool(taxonomy_file or settings.OBS_TAXONOMY_FILE, themes_file, _drafts(ledger))
    try:
        themes_list = themes_mod.load_themes(themes_file) if themes_file else themes_mod.load_themes()
    except themes_mod.ThemeError:
        themes_list = None
    text = digest_text(date, audit=audit, ledger=ledger, alerts_log=read_jsonl(alerts_log or settings.SOCIAL_ALERTS_LOG), rules=rules, pool=pool,
                       pool_note=why, themes_list=themes_list, engagement_file=engagement_file or settings.SOCIAL_ENGAGEMENT_LOG)
    path = Path(digest_dir or settings.SOCIAL_DIGEST_DIR) / f"{date.isoformat()}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


# --------------------------------------------------------------------------------------------------- commands

def cmd_social_alerts(args) -> int:
    now = dt.datetime.now(settings.LOCAL_TZ)
    rep = run_alerts(now, since=parse_since(args.since), notify=args.notify)
    for a in rep["alerts"]:
        print(f"{a.level.upper():7} {a.kind}: {a.message}{'' if a in rep['new'] else '  (already announced)'}")
    for n in rep["notes"]:
        print(f"note: {n}")
    for a in rep["notify_failed"]:
        print(f"notify failed for {a.key}: see {settings.SOCIAL_ALERTS_LOG}", file=sys.stderr)
    if not rep["alerts"]:
        print("no alerts")
    return rep["exit"]


def cmd_social_digest(args) -> int:
    try:
        date = dt.date.fromisoformat(args.date) if args.date else dt.datetime.now(settings.LOCAL_TZ).date()
    except ValueError as e:
        raise AlertsError(f"--date {args.date!r}: expected YYYY-MM-DD") from e
    print(write_digest(date))
    return 0


def add_parsers(sub) -> None:
    p = sub.add_parser("social-alerts", help="Scan the posting audit log, ledger, disks and stock for things that need attention; exit 0 none, 1 warnings, 2 errors.")
    p.add_argument("--since", default="24h", help="how far back to read events (30m, 24h, 2d; default 24h)")
    p.add_argument("--notify", action="store_true", help="run the alert command (BIO_SOCIAL_ALERT_COMMAND) once per new alert")
    p = sub.add_parser("social-digest", help="Write the day's Markdown digest (posts, skips, rules, pool, next days, alerts) and print its path.")
    p.add_argument("--date", default=None, help="YYYY-MM-DD (default today)")


COMMANDS = {"social-alerts": cmd_social_alerts, "social-digest": cmd_social_digest}
ERRORS = (AlertsError,)


def _maintenance(ctx: dict) -> None:
    """After each social-draft run: scan and record new alerts (notifies only when an alert command is configured)."""
    run_alerts(dt.datetime.now(settings.LOCAL_TZ), rules=ctx["rules"])


def register_hooks() -> None:
    from . import social_hooks

    social_hooks.register("alerts_scan", "maintenance", _maintenance, rule="alerts", priority=90)
