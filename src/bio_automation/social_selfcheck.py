"""`bio social-selfcheck [--online] [--json]`: one pass over everything the scheduled Bluesky posting run depends on.

Each check answers ok, warn or fail with a one-line message. The exit code is the worst level: 0 all ok, 1 a warning, 2 a failure. A failure
is something that would stop or damage the next post (the rules file does not load, the ledger has a torn line, an approved draft no longer
matches its approval, no Bluesky session); a warning is something that degrades it (a stale gallery index, low disk space, a small photo that
will be withdrawn). The run is written to the audit log as the event "selfcheck" (level, problems). Nothing is posted, nothing is changed;
--online adds two read-only probes (the Bluesky session refresh `social-whoami` does, and an iNat API request).

scripts/deploy_prod.ps1 runs this from the prod tree after a fast-forward with --deploy and rolls back unless it exits 0 or 1 AND prints the final
summary line "selfcheck: ok|warn|fail" (a crash, which exits 1 with a traceback in Python, has no such line; the command itself catches any
crash and exits 2 with "selfcheck: fail (<kind>)"). In --deploy mode only the checks that depend on the code
(the extensions and their critical hooks, the CLI import, the rules and themes schema on the built-in defaults) may fail; everything that depends on
the machine's data (disk space, the Bluesky session, the ledger, the audit log, the owner's rules and themes files, the gallery index, the taxonomy
cache, the drafts) is a warning at worst: a good deploy is not rolled back for them."""
from __future__ import annotations

import datetime as dt
import importlib
import inspect
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Callable, NamedTuple

from . import settings, social_hooks

OK, WARN, FAIL = "ok", "warn", "fail"
EXIT = {OK: 0, WARN: 1, FAIL: 2}
INDEX_MAX_AGE_DAYS = 3
GB = 1024 ** 3
SYSTEM_DRIVE_FAIL_GB = 15.0                  # the NAS pipeline pauses below this much free space on C: (settings.MIN_FREE_GB)
SYSTEM_DRIVE_WARN_GB = 20.0
DATA_DRIVE_FAIL_GB = 2.0
DATA_DRIVE_WARN_GB = 10.0
ONLINE_TIMEOUT_S = 10
CODE_DEPENDENT = ("rules", "themes", "extensions", "cli")   # --deploy: only a failure of these (they depend on the code being deployed) is a failure; every other one is a warning


class Check(NamedTuple):
    name: str
    level: str
    message: str


class SelfCheckError(Exception):
    pass


def worst(checks: list[Check]) -> str:
    return max((c.level for c in checks), key=lambda lv: EXIT[lv], default=OK)


# ----------------------------------------------------------------------------------------------------------------- the checks

def check_rules(name: str = "rules", *, builtin: bool = False):
    """(Check, rules): the rules as the file and the environment say (builtin: the built-in defaults, which depend on the code alone); the defaults
    stand in for the other checks when they do not load."""
    from .social_rules import RulesError, load_rules

    try:
        rules = load_rules(Path(os.devnull), env={}) if builtin else load_rules()
    except RulesError as e:
        return Check(name, FAIL, f"rules do not load: {e}"), load_rules(Path(os.devnull), env={})
    off = sorted(n for n, s in rules.snapshot().items() if not s["on"])
    return Check(name, OK, f"{'built-in ' if builtin else ''}rules load ({len(rules.snapshot()) - len(off)} on, {len(off)} off)"), rules


def check_cli() -> Check:
    """The command line module imports and has its entry point: a deploy whose CLI cannot start posts nothing."""
    try:
        mod = importlib.import_module(f"{__package__ or 'bio_automation'}.cli")
    except Exception as e:  # noqa: BLE001 -- whatever the import raised, the CLI is broken
        return Check("cli", FAIL, f"the CLI does not import: {type(e).__name__}: {e}"[:300])
    if not callable(getattr(mod, "main", None)):
        return Check("cli", FAIL, "the CLI has no main()")
    return Check("cli", OK, "the CLI imports")


CRITICAL_HOOKS = (("alt_text_privacy", "draft"), ("image_floor", "draft"), ("live_recheck", "pre_send"),
                  ("gate_rescreen_privacy", "pre_send"), ("gate_rescreen_image", "pre_send"))     # the privacy and image gates: a post without them must not ship


def check_extensions() -> Check:
    """Every module in social_hooks.EXTENSIONS must load (a missing one counts) and the critical hooks must be registered: one that is not leaves its gates out of the posting run."""
    mods = social_hooks.load_all()
    bad = dict(social_hooks.failed())
    loaded = {m.__name__.rsplit(".", 1)[-1] for m in mods}
    for name in social_hooks.EXTENSIONS:
        if name not in loaded and name not in bad:
            bad[name] = "not loaded"
    registered = {(h.name, h.stage) for stage in social_hooks.STAGES for h in social_hooks.hooks(stage)}
    missing = [f"{n} ({st})" for n, st in CRITICAL_HOOKS if (n, st) not in registered]
    problems = []
    if bad:
        problems.append(f"{len(bad)} social module(s) did not load: " + "; ".join(f"{n} ({e})" for n, e in sorted(bad.items())))
    if missing:
        problems.append("critical hook(s) not registered: " + ", ".join(missing))
    if problems:
        return Check("extensions", FAIL, "; ".join(problems))
    return Check("extensions", OK, f"{len(mods)} social modules loaded, {len(CRITICAL_HOOKS)} critical hooks registered")


def check_themes(name: str = "themes", *, builtin: bool = False) -> Check:
    from .themes import ThemeError, load_themes

    path = Path(os.devnull) if builtin else None                  # builtin: not a file, so the defaults load (and validate)
    try:
        themes = load_themes(path)
    except ThemeError as e:
        return Check(name, FAIL, f"themes do not load: {e}")
    source = "file" if not builtin and Path(settings.SOCIAL_THEMES_FILE).is_file() else "built-in defaults"
    return Check(name, OK, f"{len(themes)} themes ({source})")


def _light_ledger_check(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    if not raw:
        return OK, "ledger is empty"
    lines = raw.split(b"\n")
    torn_tail = lines[-1] != b""                       # the last record has no newline: a write was cut short, or the next append would glue on
    body = lines[:-1] if not torn_tail else lines
    for n, line in enumerate(body, 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line.decode("utf-8"))
            if not isinstance(rec, dict) or "event" not in rec or "draft_id" not in rec:
                raise ValueError("not a ledger record")
        except ValueError as e:
            where = "last line is torn" if n == len(body) and isinstance(e, json.JSONDecodeError) else f"line {n} is corrupt"
            return FAIL, f"ledger {where} ({type(e).__name__})"
    if torn_tail:
        return FAIL, "ledger last line has no newline (the next record would be glued to it)"
    return OK, f"ledger parses ({sum(1 for x in body if x.strip())} records)"


def _deep_integrity(path: Path):
    """social_state.check_integrity(ledger path), when that module and function exist and accept one path. Returns (problems, note) or None."""
    try:
        mod = importlib.import_module(f"{__package__ or 'bio_automation'}.social_state")
    except ModuleNotFoundError as e:
        if e.name and e.name.endswith("social_state"):
            return None
        raise
    fn = getattr(mod, "check_integrity", None)
    if fn is None:
        return None
    try:
        inspect.signature(fn).bind(path)
    except TypeError:
        return None
    res = fn(path)
    if hasattr(res, "problems"):
        res = res.problems
    elif isinstance(res, dict):
        res = res.get("problems", [])
    if res is None or res is True:
        return [], ""
    if res is False:
        return ["integrity check failed"], ""
    return [str(p) for p in res], ""


def check_ledger() -> Check:
    path = Path(settings.SOCIAL_LEDGER)
    if not path.is_file():
        return Check("ledger", WARN, f"no ledger at {path} (nothing drafted yet)")
    level, message = _light_ledger_check(path)
    if level == FAIL:
        return Check("ledger", FAIL, message)
    deep = _deep_integrity(path)
    if deep is None:
        return Check("ledger", level, f"{message}; deep integrity check skipped (social_state.check_integrity is not available)")
    problems, _ = deep
    if problems:
        return Check("ledger", FAIL, f"{message}; integrity: {'; '.join(problems[:3])}" + (f" (+{len(problems) - 3} more)" if len(problems) > 3 else ""))
    return Check("ledger", OK, f"{message}; integrity check clean")


def check_audit_writable() -> Check:
    path = Path(settings.SOCIAL_AUDIT_LOG)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "ab"):
            pass
    except OSError as e:
        return Check("audit", FAIL, f"audit log {path} is not writable ({e.strerror or e}): a post is not sent when its decision cannot be written")
    return Check("audit", OK, f"audit log {path} is writable")


def check_bsky_session() -> Check:
    path = Path(settings.BSKY_OAUTH_DIR) / "account.json"
    if not path.is_file():
        return Check("bluesky-session", FAIL, f"no Bluesky session: {path} is missing (run `bio social-login`)")
    try:
        acct = json.loads(path.read_text(encoding="utf-8"))
        who = acct["handle"]
        acct["did"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        return Check("bluesky-session", FAIL, f"{path} is unreadable ({type(e).__name__}); run `bio social-login`")
    return Check("bluesky-session", OK, f"session files present for {who}")


def check_gallery_index(now: dt.datetime) -> Check:
    path = Path(settings.GALLERY_OBS_INDEX)
    if not path.is_file():
        return Check("gallery-index", WARN, f"no gallery index at {path}: run `bio social-index`")
    try:
        files = len(json.loads(path.read_text(encoding="utf-8"))["files"])
    except (OSError, ValueError, KeyError, TypeError) as e:
        return Check("gallery-index", FAIL, f"gallery index {path} is unreadable ({type(e).__name__})")
    age = now.timestamp() - path.stat().st_mtime
    days = age / 86400
    if days >= INDEX_MAX_AGE_DAYS:
        return Check("gallery-index", WARN, f"gallery index is {days:.1f} days old (limit {INDEX_MAX_AGE_DAYS}), {files} files")
    return Check("gallery-index", OK, f"gallery index is {days:.1f} days old, {files} files")


def check_taxonomy() -> Check:
    path = Path(settings.OBS_TAXONOMY_FILE)
    if not path.is_file():
        return Check("taxonomy", WARN, f"no taxonomy cache at {path} (the next theme run fetches it from iNat)")
    try:
        n = len(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError) as e:
        return Check("taxonomy", FAIL, f"taxonomy cache {path} is unreadable ({type(e).__name__})")
    return Check("taxonomy", OK, f"taxonomy cache has {n} observations")


def _drafts_dir() -> Path:
    return Path(settings.SOCIAL_DIR) / "drafts"


def _ledger_drafts():
    """(drafts, None), or (None, a skipped Check's message) when the ledger cannot be read: check_ledger already failed, so these say so and go on."""
    from .social import Ledger, SocialError

    try:
        return Ledger().drafts(), None
    except (OSError, ValueError, KeyError, TypeError, SocialError) as e:
        return None, f"skipped: the ledger cannot be read ({type(e).__name__})"


def check_approved_drafts() -> Check:
    from .social import draft_hash

    drafts, skipped = _ledger_drafts()
    if skipped:
        return Check("approved-drafts", WARN, skipped)
    problems, n = [], 0
    for did, v in sorted(drafts.items()):
        if v.get("state") != "approved":
            continue
        n += 1
        try:
            if draft_hash(_drafts_dir() / did) != v.get("approved_hash"):
                problems.append(f"{did} changed since it was approved")
        except (OSError, ValueError, KeyError) as e:
            problems.append(f"{did} cannot be read ({type(e).__name__}: {getattr(e, 'filename', None) or e})")
    if problems:
        return Check("approved-drafts", FAIL, f"{len(problems)} of {n} approved drafts are not sendable: " + "; ".join(problems[:3]))
    return Check("approved-drafts", OK, f"{n} approved drafts match their approval hash and have their images")


def check_pending_quality(rules) -> Check:
    from .social import load_draft

    if not rules.on("lower_quality_photos"):
        return Check("pending-quality", OK, "lower_quality_photos is off: photo size is not checked")
    drafts, skipped = _ledger_drafts()
    if skipped:
        return Check("pending-quality", WARN, skipped)
    floor = rules.value("lower_quality_photos") or 1000
    small, n = [], 0
    for did, v in sorted(drafts.items()):
        if v.get("state") not in ("drafted", "approved"):
            continue
        n += 1
        try:
            images = load_draft(_drafts_dir() / did)["images"]
        except (OSError, ValueError, KeyError):
            continue                                                  # check_approved_drafts reports an unreadable approved draft
        edges = [max(i.get("width") or 0, i.get("height") or 0) for i in images]
        if any(e < floor for e in edges):
            small.append(f"{did} ({min(edges)} px)")
    if small:
        return Check("pending-quality", WARN, f"{len(small)} of {n} pending drafts have a photo under {floor} px and will be withdrawn: " + ", ".join(small[:3]))
    return Check("pending-quality", OK, f"all {n} pending drafts have photos of at least {floor} px")


def check_pause(rules=None) -> Check:
    from . import social_controls

    state = social_controls.read_pause()
    remote = social_controls.sync_marker_path()
    notes = []
    if remote is not None and not remote.parent.is_dir():
        notes.append(f"the synced pause folder {remote.parent} is not reachable, so a remote pause cannot be seen")
    if state is None:
        return Check("pause", WARN if notes else OK, "; ".join(notes) if notes else "info: posting is not paused")
    where = str(settings.SOCIAL_PAUSE_FILE) if state.get("source") == "local" else str(remote)
    if rules is not None and not rules.on("pause_switch"):
        return Check("pause", WARN, f"a pause marker exists ({where}) but the rule pause_switch is OFF, so posting is NOT paused")
    return Check("pause", WARN if notes else OK, f"info: posting is PAUSED ({where} exists)" + "".join("; " + n for n in notes))


def default_drives() -> dict[str, Path]:
    system = Path(os.environ.get("SystemDrive", "C:") + "\\") if os.name == "nt" else Path("/")
    return {"system": system, "data": Path(settings.CACHE_DIR)}


def _free_bytes(path: Path) -> int:
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free


def check_disks(drives: dict[str, Path], free: Callable[[Path], int]) -> list[Check]:
    limits = {"system": (SYSTEM_DRIVE_FAIL_GB, SYSTEM_DRIVE_WARN_GB), "data": (DATA_DRIVE_FAIL_GB, DATA_DRIVE_WARN_GB)}
    out = []
    for name, path in drives.items():
        fail_gb, warn_gb = limits[name]
        gb = free(path) / GB
        label = f"disk-{name}"
        if gb < fail_gb:
            out.append(Check(label, FAIL, f"{path} has {gb:.1f} GB free (below {fail_gb:g})"))
        elif gb < warn_gb:
            out.append(Check(label, WARN, f"{path} has {gb:.1f} GB free (below {warn_gb:g})"))
        else:
            out.append(Check(label, OK, f"{path} has {gb:.1f} GB free"))
    return out


# --------------------------------------------------------------------------------------------------------------- online probes

def default_whoami() -> dict:
    from .social import poster_account

    return poster_account().whoami()


def default_inat_probe() -> int:
    from .inat import INAT_API_BASE, USER_AGENT_DEFAULT

    req = urllib.request.Request(f"{INAT_API_BASE}/observations?per_page=0", headers={"User-Agent": USER_AGENT_DEFAULT})
    with urllib.request.urlopen(req, timeout=ONLINE_TIMEOUT_S) as r:
        return r.status


def check_bsky_online(whoami: Callable[[], dict]) -> Check:
    try:
        who = whoami()
    except ImportError as e:
        return Check("bluesky-online", FAIL, f"the Bluesky client is not installed ({e})")
    except Exception as e:  # noqa: BLE001 -- any failure of the probe is the answer: the session does not work
        return Check("bluesky-online", FAIL, f"Bluesky session probe failed: {type(e).__name__}: {e}"[:300])
    return Check("bluesky-online", OK, f"Bluesky session works for {who.get('handle')} (expires {who.get('expiresAt', 'unknown')})")


def check_inat_online(probe: Callable[[], int]) -> Check:
    try:
        status = probe()
    except Exception as e:  # noqa: BLE001
        return Check("inat-online", WARN, f"iNat is not reachable: {type(e).__name__}: {e}"[:300])
    if status != 200:
        return Check("inat-online", WARN, f"iNat answered HTTP {status}")
    return Check("inat-online", OK, "iNat API reachable (HTTP 200)")


# ------------------------------------------------------------------------------------------------------------------- the run

def run_checks(*, online: bool = False, now: dt.datetime | None = None, drives: dict[str, Path] | None = None,
               free: Callable[[Path], int] | None = None, whoami: Callable[[], dict] | None = None,
               inat_probe: Callable[[], int] | None = None, deploy: bool = False) -> list[Check]:
    """deploy: "rules" and "themes" check the built-in defaults (the code), the owner's files get their own checks (rules-file, themes-file)
    and the CLI import is checked; for_deploy() then lets only the code-dependent checks fail."""
    now = now or dt.datetime.now(settings.LOCAL_TZ)
    rules_check, rules = check_rules("rules", builtin=True) if deploy else check_rules()
    extra = []
    if deploy:
        file_check, file_rules = check_rules("rules-file")
        extra = [file_check, check_themes("themes-file")]
        rules = file_rules
    checks = [rules_check, check_extensions(), *([check_cli()] if deploy else []), check_themes(builtin=deploy), *extra, check_ledger(), check_audit_writable(), check_bsky_session(), check_gallery_index(now), check_taxonomy(),
              check_approved_drafts(), check_pending_quality(rules), check_pause(rules), *check_disks(drives or default_drives(), free or _free_bytes)]
    if online:
        checks += [check_bsky_online(whoami or default_whoami), check_inat_online(inat_probe or default_inat_probe)]
    return checks


def report(checks: list[Check], *, online: bool) -> dict:
    level = worst(checks)
    return {"level": level, "exit": EXIT[level], "online": online,
            "checks": [{"name": c.name, "level": c.level, "message": c.message} for c in checks]}


def for_deploy(checks: list[Check]) -> list[Check]:
    """--deploy: only the code-dependent checks (CODE_DEPENDENT: the built-in rules and themes, the extensions and their critical hooks, the CLI
    import) may fail; every other failure (disk, session, ledger, audit, the owner's rules and themes files, indexes, caches, drafts) depends on
    the machine's data, not on the code being deployed, and becomes a warning (the message says so)."""
    return [Check(c.name, WARN, f"{c.message} [environmental: a warning in --deploy mode]") if c.name not in CODE_DEPENDENT and c.level == FAIL else c
            for c in checks]


def selfcheck(*, online: bool = False, audit=None, deploy: bool = False, **kw) -> dict:
    from .social_rules import Audit

    checks = run_checks(online=online, deploy=deploy, **kw)
    if deploy:
        checks = for_deploy(checks)
    out = report(checks, online=online)
    try:
        (audit or Audit()).log("selfcheck", level=out["level"], online=online, problems=[f"{c.name}: {c.message}" for c in checks if c.level != OK])
    except OSError:
        pass                                                           # an unwritable audit log is already a failed check
    return out


def cmd_social_selfcheck(args) -> int:
    """0 ok, 1 warn, 2 fail. Any crash is a fail: it prints "selfcheck: fail (<exception class>)" and exits 2, never a traceback with exit 1."""
    deploy = bool(getattr(args, "deploy", False))
    try:
        out = selfcheck(online=args.online, deploy=deploy)
        if args.json and not deploy:
            print(json.dumps(out, indent=2, ensure_ascii=False))
            return out["exit"]
        for c in out["checks"]:
            print(f"[{c['level']:<4}] {c['name']}: {c['message']}")
        print(f"selfcheck: {out['level']}")
        return out["exit"]
    except Exception as e:  # noqa: BLE001 -- whatever broke, the answer is "fail", in the form the deploy script reads
        print(f"selfcheck crashed: {type(e).__name__}: {e}"[:500], file=sys.stderr, flush=True)
        print(f"selfcheck: fail ({type(e).__name__})")                           # last, so it is the final line even when stderr is merged in
        return EXIT[FAIL]


def add_parsers(sub) -> None:
    p = sub.add_parser("social-selfcheck", help="Check everything the scheduled Bluesky run depends on; exit 0 ok, 1 warn, 2 fail. Posts nothing.")
    p.add_argument("--online", action="store_true", help="also probe the Bluesky session and iNat (read-only)")
    p.add_argument("--json", action="store_true", help="print the result as JSON")
    p.add_argument("--deploy", action="store_true", help="what scripts/deploy_prod.ps1 runs: only code-dependent checks (extensions, CLI, built-in rules/themes) can fail; "
                   "disk, session, ledger, audit, rules/themes files, indexes and drafts are warnings; the last output line is always `selfcheck: ok|warn|fail`")


COMMANDS = {"social-selfcheck": cmd_social_selfcheck}
ERRORS: tuple = ()
