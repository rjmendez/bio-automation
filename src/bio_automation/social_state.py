"""Ledger integrity and backups for the Bluesky posting workflow (rule ledger_integrity).

The ledger and the audit log are single append-only files on one disk. A corrupt ledger could make the poster forget that a draft went out and
send it again, so:
  * check_integrity() reads the ledger without trusting it: every line valid JSON with at, event and draft_id, only allowed state transitions,
    no observation in two different posted drafts, no unknown event, no torn last line.
  * a pre_send hook (priority 2) blocks every post, scheduled or manual, while the ledger is corrupt.
  * `bio social-integrity [--repair-tail]` prints the report and can move a torn final line aside (after a backup).
    `bio social-integrity --accept-duplicate OBS_ID --reason TEXT` records the owner's acknowledgement of exactly that known duplicate post in
    settings.SOCIAL_INTEGRITY_ACK_FILE (audited); the checker then no longer reports it, while any other duplicate still blocks.
  * `bio social-backup [--keep N]`, a daily maintenance hook, and `bio social-restore` keep and bring back dated copies of the ledger, the
    audit log and the owner's rule, theme and print files, plus drafts_hashes.json (the sha256 of every draft folder, informational).
    Retention: the newest N backups and at most settings.SOCIAL_BACKUP_MAX_MB in all (the newest and the newest clean one always stay).
    Optional off-drive copy: settings.SOCIAL_BACKUP_OFFSITE_DIR (off by default); each new backup is copied there, read back and hashed, audited
    as backup_offsite or backup_offsite_failed; a folder on the backup folder's own drive is refused.
  * `bio social-restore-check` restores the newest backup into a scratch folder and validates it (hashes, ledger integrity, audit parse, drafts).
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import NamedTuple

from . import settings
from .social import NOTES, STATES, SocialError, ledger_lock
from .social_hooks import Verdict

# state -> the states it may be followed by (a note such as "error" is not a state and is ignored)
TRANSITIONS: dict[str, tuple[str, ...]] = {
    "drafted": ("approved", "rejected", "withdrawn", "drafted"),
    "approved": ("approved", "posting", "rejected", "withdrawn", "failed"),     # failed: a draft that sent nothing for STALL_FAIL_DAYS days (social.stalled_drafts)
    "posting": ("posted", "failed", "approved"),                  # approved only as a restored approval (a send that sent nothing)
    "failed": ("approved", "rejected", "withdrawn", "drafted"),
    "withdrawn": ("drafted",),
    "rejected": (),
    "posted": (),
}
KEEP_DEFAULT = 14
STAMP = re.compile(r"\d{8}-\d{6}(-\d+)?")
MANIFEST = "MANIFEST.json"
SHOWN_PROBLEMS = 3


class IntegrityError(SocialError):
    pass


class Problem(NamedTuple):
    line: int                         # 1-based line number in the file (0: about the ledger as a whole)
    kind: str                         # bad_json | not_object | missing_field | bad_time | unknown_event | bad_transition | duplicate_post | torn_tail | missing_newline | bad_encoding
    message: str
    draft_id: str | None = None
    obs: str | None = None            # duplicate_post: the observation
    drafts: tuple = ()                # duplicate_post: the posted drafts it is in

    def text(self) -> str:
        return f"{self.kind}" + (f" at line {self.line}" if self.line else "") + f": {self.message}"


class Report(NamedTuple):
    path: Path
    lines: int                        # non-blank lines read
    problems: list[Problem]

    @property
    def ok(self) -> bool:
        return not self.problems

    @property
    def torn_tail(self) -> bool:
        return any(p.kind == "torn_tail" for p in self.problems)

    def summary(self, shown: int = SHOWN_PROBLEMS) -> str:
        head = "; ".join(p.text() for p in self.problems[:shown])
        more = len(self.problems) - shown
        return head + (f" (and {more} more)" if more > 0 else "")


def _obs_of(rec: dict) -> list:
    ids = rec.get("obs_ids")
    if isinstance(ids, list) and ids:
        return list(ids)
    return [rec["obs_id"]] if rec.get("obs_id") is not None else []


def load_acknowledged(ack_file: Path | str | None = None) -> dict[str, list[str]]:
    """{observation id (str): the sorted posted drafts the owner acknowledged as its known duplicate}; a missing or damaged file acknowledges nothing."""
    try:
        raw = json.loads(Path(ack_file or settings.SOCIAL_INTEGRITY_ACK_FILE).read_text(encoding="utf-8"))
        entries = raw["duplicates"]
        return {str(o): sorted(e["drafts"]) for o, e in entries.items() if isinstance(e, dict) and isinstance(e.get("drafts"), list)}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return {}


def accept_duplicate(report: Report, obs_id: str, reason: str, *, ack_file: Path | str | None = None, now: dt.datetime | None = None) -> dict:
    """Record the owner's acknowledgement of the duplicate_post problem of `obs_id` in `report` (atomic write, audited). The acknowledgement
    names the exact drafts: a further post of the same observation is a different duplicate and blocks again."""
    if not reason.strip():
        raise SocialError("--reason is required: say why the duplicate post is known and accepted")
    found = next((p for p in report.problems if p.kind == "duplicate_post" and p.obs == str(obs_id)), None)
    if found is None:
        raise SocialError(f"the ledger has no duplicate_post problem for observation {obs_id}")
    path = Path(ack_file or settings.SOCIAL_INTEGRITY_ACK_FILE)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        entries = doc["duplicates"] if isinstance(doc.get("duplicates"), dict) else {}
    except (OSError, ValueError, AttributeError):
        entries = {}
    entry = {"drafts": sorted(found.drafts), "reason": reason.strip(), "at": (now or _now()).isoformat(timespec="seconds")}
    entries[str(obs_id)] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"duplicates": entries}, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)
    _audit("integrity_acknowledged", obs_id=str(obs_id), drafts=entry["drafts"], reason=entry["reason"])
    return entry


def check_integrity(ledger_path: Path | str, *, acknowledged: dict[str, list[str]] | None = None) -> Report:
    """Check the ledger file as it is on disk. A missing file is an empty, clean ledger. `acknowledged` (default: the owner's
    settings.SOCIAL_INTEGRITY_ACK_FILE) names duplicate posts that are known and accepted; exactly those are not reported."""
    path = Path(ledger_path)
    problems: list[Problem] = []
    if not path.is_file():
        return Report(path, 0, problems)
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        return Report(path, 0, [Problem(0, "bad_encoding", f"not valid UTF-8 ({e.reason} at byte {e.start})")])
    pieces = text.split("\n")
    ends_with_newline = text.endswith("\n")
    if ends_with_newline:
        pieces.pop()
    state: dict[str, str] = {}                                 # draft id -> its current state
    obs_of_draft: dict[str, list] = {}                         # draft id -> the observations of its latest drafted record
    posted_by_obs: dict[object, list[str]] = {}
    count = 0
    for number, line in enumerate(pieces, start=1):
        if not line.strip():
            continue
        count += 1
        last = number == len(pieces)
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            if last and not ends_with_newline:
                problems.append(Problem(number, "torn_tail", f"the last line stops mid-record ({e.msg}); a write was cut short"))
            else:
                problems.append(Problem(number, "bad_json", f"not valid JSON ({e.msg})"))
            continue
        if not isinstance(rec, dict):
            problems.append(Problem(number, "not_object", "the line is JSON but not an object"))
            continue
        missing = [k for k in ("at", "event", "draft_id") if not isinstance(rec.get(k), str) or not rec.get(k)]
        if missing:
            problems.append(Problem(number, "missing_field", f"missing or empty {', '.join(missing)}"))
            continue
        try:
            dt.datetime.fromisoformat(rec["at"])
        except ValueError:
            problems.append(Problem(number, "bad_time", f"at={rec['at']!r} is not an ISO timestamp", rec["draft_id"]))
        event, draft_id = rec["event"], rec["draft_id"]
        if event not in STATES + NOTES:
            problems.append(Problem(number, "unknown_event", f"event {event!r} (known: {', '.join(STATES + NOTES)})", draft_id))
            continue
        if event in NOTES:
            continue
        before = state.get(draft_id)
        if before is None:
            if event != "drafted":
                problems.append(Problem(number, "bad_transition", f"{draft_id} starts with {event!r}, not drafted", draft_id))
        elif event not in TRANSITIONS[before] or (before == "posting" and event == "approved" and not rec.get("restored")):
            problems.append(Problem(number, "bad_transition", f"{draft_id}: {before} -> {event} is not allowed", draft_id))
        state[draft_id] = event
        if event == "drafted":
            obs_of_draft[draft_id] = _obs_of(rec)
        elif event == "posted":
            for obs in obs_of_draft.get(draft_id, []):
                if draft_id not in posted_by_obs.setdefault(obs, []):
                    posted_by_obs[obs].append(draft_id)
    acked = load_acknowledged() if acknowledged is None else acknowledged
    for obs, drafts in posted_by_obs.items():
        if len(drafts) > 1 and acked.get(str(obs)) != sorted(drafts):
            problems.append(Problem(0, "duplicate_post", f"observation {obs} is in {len(drafts)} posted drafts: {', '.join(drafts)}; if this is known and "
                                    f"accepted: `bio social-integrity --accept-duplicate {obs} --reason TEXT`", drafts[-1], str(obs), tuple(drafts)))
    if raw and not ends_with_newline and not any(p.kind == "torn_tail" for p in problems):
        problems.append(Problem(len(pieces), "missing_newline", "the last record has no newline, the next append would join it to its line"))
    return Report(path, count, problems)


# --------------------------------------------------------------------------------------------------- the pre_send hook

def _stamp(now: dt.datetime) -> str:
    return now.strftime("%Y%m%d-%H%M%S")


def _now() -> dt.datetime:
    return dt.datetime.now(settings.LOCAL_TZ)


def ledger_gate(ctx: dict) -> Verdict | None:
    """pre_send, rule ledger_integrity: a corrupt ledger blocks every post (nothing is withdrawn, the draft is untouched)."""
    report = check_integrity(ctx["ledger"].path)
    if report.ok:
        return None
    reason = f"the ledger {report.path} is corrupt: {report.summary()}; posting is blocked; run `bio social-integrity`"
    try:
        from .social_rules import Audit

        Audit().log("ledger_corrupt", path=str(report.path), count=len(report.problems), problems=[p.text() for p in report.problems[:SHOWN_PROBLEMS]],
                    kinds=sorted({p.kind for p in report.problems}), reason=report.summary(),
                    draft_id=ctx.get("draft_id"), manual=bool(ctx.get("manual")))
    except OSError:
        pass                                                   # the block does not depend on the audit line
    return Verdict("block", reason)


def duplicate_gate(ctx: dict) -> Verdict | None:
    """pre_send, rule ledger_integrity: withdraw a candidate when any of its observations is already in a posted or posting draft with a different
    draft id (two drafts of one observation, e.g. a single and a themed one, must not both go out). A pass for the same draft id."""
    drafts = ctx["ledger"].drafts()
    mine = drafts.get(ctx["draft_id"])
    if mine is None:
        return None
    wanted = {str(o) for o in _obs_of(mine)}
    for did, v in sorted(drafts.items()):
        if did == ctx["draft_id"] or v.get("state") not in ("posted", "posting"):
            continue
        shared = sorted(wanted & {str(o) for o in _obs_of(v)})
        if shared:
            return Verdict("withdraw", f"observation {', '.join(shared)} is already in {v['state']} draft {did}; posting {ctx['draft_id']} would be a duplicate")
    return None


# --------------------------------------------------------------------------------------------------- backups

def backup_sources(ledger_path: Path | str | None = None) -> dict[str, Path]:
    """{name in the backup: the live file}."""
    return {"ledger.jsonl": Path(ledger_path or settings.SOCIAL_LEDGER), "audit.jsonl": Path(settings.SOCIAL_AUDIT_LOG),
            "rules.toml": Path(settings.SOCIAL_RULES_FILE), "themes.toml": Path(settings.SOCIAL_THEMES_FILE),
            "prints.csv": Path(settings.SOCIAL_PRINTS_FILE), "prints_manual.csv": Path(settings.SOCIAL_PRINTS_MANUAL_FILE),
            "obs_taxonomy.json": Path(settings.OBS_TAXONOMY_FILE)}


def backups(root: Path | None = None) -> list[Path]:
    """The backup folders, oldest first."""
    root = Path(root or settings.SOCIAL_BACKUP_DIR)
    if not root.is_dir():
        return []
    return sorted((p for p in root.iterdir() if p.is_dir() and STAMP.fullmatch(p.name)), key=lambda p: p.name)


def read_manifest(folder: Path) -> dict:
    try:
        return json.loads((Path(folder) / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise IntegrityError(f"backup {Path(folder).name} has no readable {MANIFEST}: {e}") from e


def _audit(event: str, **fields) -> None:
    try:
        from .social_rules import Audit

        Audit().log(event, **fields)
    except OSError:
        pass


def _folder_bytes(folder: Path) -> int:
    return sum(p.stat().st_size for p in Path(folder).rglob("*") if p.is_file())


def _retain(root: Path, keep: int, max_bytes: int | None = None) -> list[str]:
    """Delete all but the newest `keep` backups, except that the newest backup whose ledger was clean is never deleted; then, while the
    backups together exceed `max_bytes` (None or 0: no size cap), delete the oldest others (the newest backup and the newest clean one stay).
    Returns the removed names."""
    folders = backups(root)
    survivors = folders[-keep:]
    good = next((f for f in reversed(folders) if _manifest_ledger_ok(f)), None)
    removed = []
    for f in folders:
        if f in survivors or f == good:
            continue
        try:
            shutil.rmtree(f)
        except OSError:
            continue                                           # in use: the next backup retries
        removed.append(f.name)
    if max_bytes:
        left = backups(root)
        sizes = {f: _folder_bytes(f) for f in left}
        for f in left[:-1]:                                    # never the newest
            if sum(sizes.values()) <= max_bytes:
                break
            if f == good:
                continue
            try:
                shutil.rmtree(f)
            except OSError:
                continue
            sizes.pop(f)
            removed.append(f.name)
    return removed


def _manifest_ledger_ok(folder: Path) -> bool:
    try:
        return bool(read_manifest(folder).get("ledger_ok"))
    except IntegrityError:
        return False


DRAFTS_HASHES = "drafts_hashes.json"
INFO_FILES = (DRAFTS_HASHES,)          # in a backup and hash-checked on restore, but not restored onto a live file


def draft_hashes(drafts_dir: Path) -> dict[str, dict]:
    """{draft_id: {"sha256": draft_hash} or {"error": why}} for every folder under drafts_dir that holds a draft.json."""
    from .social import draft_hash

    out: dict[str, dict] = {}
    drafts_dir = Path(drafts_dir)
    if not drafts_dir.is_dir():
        return out
    for d in sorted(drafts_dir.iterdir()):
        if not (d / "draft.json").is_file():
            continue
        try:
            out[d.name] = {"sha256": draft_hash(d)}
        except (OSError, ValueError, KeyError, TypeError) as e:
            out[d.name] = {"error": f"{type(e).__name__}: {e}"[:200]}
    return out


def _device(path: Path):
    p = Path(path).absolute()
    while not p.exists() and p != p.parent:
        p = p.parent
    return os.stat(p).st_dev


def offsite_copy(folder: Path, dest_root: Path | str, *, keep: int | None = KEEP_DEFAULT, drive_of=None) -> Path | None:
    """Copy backup `folder` to <dest_root>/<its name> through a temp folder, read every file back and compare its sha256 with the manifest
    (a copy that lost bytes, as on F:, fails), prune the destination to the newest `keep` backups, audit backup_offsite. A destination on the
    same drive as the backup, or any error, is audited as backup_offsite_failed and returns None; it never raises (the backup itself exists)."""
    folder, dest_root = Path(folder), Path(dest_root)
    drive_of = drive_of or _device
    tmp = dest_root / f".tmp-{folder.name}-{os.getpid()}"
    try:
        if drive_of(folder) == drive_of(dest_root):
            raise SocialError(f"the off-drive folder {dest_root} is on the same drive as the backups {folder.parent}; it protects nothing")
        manifest = read_manifest(folder)
        dest_root.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        for fname in [*manifest["files"], MANIFEST]:
            data = (folder / fname).read_bytes()
            (tmp / fname).write_bytes(data)                    # one write, no appends (multi-write appends to F: lose bytes)
        for fname, info in manifest["files"].items():
            back = hashlib.sha256((tmp / fname).read_bytes()).hexdigest()
            if back != info["sha256"]:
                raise SocialError(f"{fname} read back from {dest_root} does not match its sha256 (expected {info['sha256'][:12]}..., found {back[:12]}...)")
        final = dest_root / folder.name
        if final.exists():
            shutil.rmtree(final)
        os.replace(tmp, final)
        removed = _retain(dest_root, keep) if keep is not None else []
    except (OSError, SocialError) as e:
        shutil.rmtree(tmp, ignore_errors=True)
        _audit("backup_offsite_failed", name=folder.name, dest=str(dest_root), error=str(e)[:300])
        return None
    _audit("backup_offsite", name=folder.name, dest=str(dest_root), files=sorted(manifest["files"]), removed=removed)
    return final


def make_backup(root: Path | None = None, *, sources: dict[str, Path] | None = None, now: dt.datetime | None = None,
                keep: int | None = KEEP_DEFAULT, drafts_dir: Path | None = None, max_mb: int | None = None,
                offsite: Path | str | None = None, drive_of=None) -> Path:
    """Copy the files that exist into <root>/<YYYYMMDD-HHMMSS>/ through a temp folder renamed into place, with a MANIFEST.json (sha256, bytes);
    then keep the newest `keep` (None: keep all) within `max_mb` (default settings.SOCIAL_BACKUP_MAX_MB). The sha256 of every draft folder goes
    in as drafts_hashes.json. Audits backup_made or, on failure, backup_failed (and raises SocialError). `offsite` (default
    settings.SOCIAL_BACKUP_OFFSITE_DIR, empty = off) gets a verified copy afterwards (offsite_copy)."""
    root = Path(root or settings.SOCIAL_BACKUP_DIR)
    drafts_dir = Path(drafts_dir or Path(settings.SOCIAL_DIR) / "drafts")
    max_mb = settings.SOCIAL_BACKUP_MAX_MB if max_mb is None else max_mb
    offsite = settings.SOCIAL_BACKUP_OFFSITE_DIR if offsite is None else offsite
    sources = backup_sources() if sources is None else sources
    now = now or _now()
    if keep is not None and keep < 1:
        raise SocialError(f"--keep must be at least 1, got {keep}")
    tmp = root / f".tmp-{_stamp(now)}-{os.getpid()}"
    try:
        root.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        files = {}
        ledger_ok = True
        for name, src in sources.items():
            if not Path(src).is_file():
                continue
            data = Path(src).read_bytes()                     # one read: the copy and its hash are the same bytes
            (tmp / name).write_bytes(data)
            files[name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "source": str(src)}
            if name == "ledger.jsonl":
                ledger_ok = check_integrity(tmp / name).ok
        if not files:
            raise SocialError("none of the files to back up exist")
        hashes = draft_hashes(drafts_dir)
        if hashes:
            data = json.dumps(hashes, indent=2, sort_keys=True).encode("utf-8")
            (tmp / DRAFTS_HASHES).write_bytes(data)
            files[DRAFTS_HASHES] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "source": str(drafts_dir)}
        (tmp / MANIFEST).write_text(json.dumps({"created": now.isoformat(timespec="seconds"), "ledger_ok": ledger_ok, "files": files},
                                               indent=2, sort_keys=True), encoding="utf-8")
        final, n = root / _stamp(now), 1
        while final.exists():
            n += 1
            final = root / f"{_stamp(now)}-{n}"
        os.replace(tmp, final)
    except (OSError, SocialError) as e:
        shutil.rmtree(tmp, ignore_errors=True)
        _audit("backup_failed", error=str(e)[:300], root=str(root))
        if isinstance(e, SocialError):
            raise
        raise SocialError(f"backup into {root} failed: {e}") from e
    removed = _retain(root, keep, max_mb * 1024 * 1024) if keep is not None else []
    _audit("backup_made", name=final.name, files=sorted(files), bytes=sum(f["bytes"] for f in files.values()), ledger_ok=ledger_ok, removed=removed)
    if str(offsite).strip():
        offsite_copy(final, Path(str(offsite).strip()), keep=keep, drive_of=drive_of)
    return final


def daily_backup(ctx: dict) -> None:
    """maintenance, rule ledger_integrity: one backup a day (none when one already exists for today)."""
    root = Path(settings.SOCIAL_BACKUP_DIR)
    today = ctx["today"].strftime("%Y%m%d")
    if any(f.name.startswith(today) for f in backups(root)):
        return None
    try:
        make_backup(root, sources=backup_sources(ctx["ledger"].path))
    except SocialError:
        pass                                                   # already audited as backup_failed
    return None


def restore(name: str, *, apply: bool = False, root: Path | None = None, targets: dict[str, Path] | None = None,
            now: dt.datetime | None = None) -> list[tuple[str, str]]:
    """Restore the files of backup `name`. Dry run (default) returns [(file, "same" | "would change" | "would create")] and touches nothing.
    apply: verifies every sha256 first (a mismatch refuses the whole restore), makes a pre-restore backup, then replaces each file atomically."""
    root = Path(root or settings.SOCIAL_BACKUP_DIR)
    if not name or name != Path(name).name or name in (".", ".."):
        raise SocialError(f"{name!r} is not a backup name (see `bio social-restore --list`)")
    folder = root / name
    if not folder.is_dir():
        raise SocialError(f"no backup {name!r} in {root} (see `bio social-restore --list`)")
    manifest = read_manifest(folder)
    targets = backup_sources() if targets is None else targets
    plan, payload = [], {}
    for fname, info in manifest["files"].items():
        if fname not in targets and fname not in INFO_FILES:
            raise SocialError(f"backup {name} holds {fname}, which this version does not know how to restore")
        src = folder / fname
        try:
            data = src.read_bytes()
        except OSError as e:
            raise IntegrityError(f"backup {name}: cannot read {fname}: {e}") from e
        digest = hashlib.sha256(data).hexdigest()
        if digest != info["sha256"]:
            raise IntegrityError(f"backup {name}: {fname} does not match its recorded sha256 (expected {info['sha256'][:12]}..., found {digest[:12]}...); nothing was restored")
        if fname in INFO_FILES:
            continue                                           # hash-checked above, nothing to put back
        payload[fname] = data
        target = Path(targets[fname])
        plan.append((fname, "would create" if not target.is_file() else "same" if hashlib.sha256(target.read_bytes()).hexdigest() == digest else "would change"))
    if not apply:
        return plan
    pre = make_backup(root, sources=targets, now=now, keep=None)   # the pre-restore backup; never evicts anything
    done = []
    ledger_target = targets.get("ledger.jsonl")
    with ledger_lock(ledger_target) if ledger_target and "ledger.jsonl" in payload else contextlib.nullcontext():
        for fname, data in payload.items():
            target = Path(targets[fname])
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".restore-tmp")
            try:
                tmp.write_bytes(data)
                os.replace(tmp, target)
            except PermissionError as e:
                tmp.unlink(missing_ok=True)
                raise SocialError(f"cannot replace {target}: another program has it open or it is read-only ({e.strerror or e}); "
                                  f"restored so far: {', '.join(done) or 'nothing'}; close the program and run the restore again") from e
            done.append(fname)
    _audit("restored", name=name, files=done, pre_restore_backup=pre.name)       # after the replace: a restored audit log would otherwise lose this line
    return [(f, "restored" if f in done else s) for f, s in plan]


# --------------------------------------------------------------------------------------------------- verified restore

def restore_check(name: str | None = None, *, root: Path | None = None, scratch_root: Path | None = None, now: dt.datetime | None = None,
                  stale_days: float | None = None) -> dict:
    """Restore backup `name` (default the newest) into a scratch folder and validate what came back: every file's sha256 against the manifest,
    the ledger (check_integrity), the audit log (every line a JSON object), and the drafts: for each draft that is approved or posting in the
    restored ledger, the sha256 recorded in drafts_hashes.json must equal the hash the approval was bound to. The newest backup must also be
    younger than `stale_days`. Touches no live file. Audits restore_check (ok) or restore_check_failed. Returns
    {"ok", "name", "problems": [...], "notes": [...], "files": {...}}."""
    from .social import Ledger

    root = Path(root or settings.SOCIAL_BACKUP_DIR)
    now = now or _now()
    stale_days = settings.SOCIAL_BACKUP_STALE_DAYS if stale_days is None else stale_days
    problems: list[str] = []
    notes: list[str] = []
    files: dict[str, str] = {}
    folders = backups(root)
    if name is None and not folders:
        problems.append(f"no backup in {root}")
    elif name is None:
        name = folders[-1].name
    result = {"ok": False, "name": name, "problems": problems, "notes": notes, "files": files}
    if name is not None and not problems:
        scratch_root = Path(scratch_root or Path(settings.CACHE_DIR) / "tmp")
        scratch_root.mkdir(parents=True, exist_ok=True)
        scratch = Path(tempfile.mkdtemp(prefix="restore-check-", dir=scratch_root))
        try:
            _check_into(scratch, root, name, now, stale_days, problems, notes, files, Ledger)
        except (SocialError, OSError) as e:
            problems.append(str(e))
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
    result["ok"] = not problems
    _audit("restore_check" if result["ok"] else "restore_check_failed", name=name, files=sorted(files), problems=problems[:5], notes=notes[:5])
    return result


def _check_into(scratch: Path, root: Path, name: str, now, stale_days, problems, notes, files, Ledger) -> None:
    if not name or name != Path(name).name or name in (".", ".."):
        raise SocialError(f"{name!r} is not a backup name (see `bio social-restore --list`)")
    folder = root / name
    if not folder.is_dir():
        raise SocialError(f"no backup {name!r} in {root}")
    manifest = read_manifest(folder)
    try:
        created = dt.datetime.fromisoformat(manifest["created"])
        age = (now - created).total_seconds() / 86400
        if age > stale_days:
            problems.append(f"backup {name} is {age:.1f} days old, more than {stale_days:g}")
    except (KeyError, ValueError, TypeError):
        problems.append(f"backup {name}: manifest has no readable 'created' time")
    for fname, info in manifest["files"].items():
        try:
            data = (folder / fname).read_bytes()
        except OSError as e:
            problems.append(f"{fname}: cannot read ({e})")
            continue
        digest = hashlib.sha256(data).hexdigest()
        if digest != info["sha256"]:
            problems.append(f"{fname}: sha256 {digest[:12]}... differs from the manifest's {info['sha256'][:12]}...")
            continue
        (scratch / fname).write_bytes(data)
        files[fname] = digest
    if "ledger.jsonl" not in files:
        problems.append("the backup has no ledger.jsonl that restored")
    else:
        report = check_integrity(scratch / "ledger.jsonl")
        if not report.ok:
            problems.append("restored ledger: " + "; ".join(p.text() for p in report.problems[:SHOWN_PROBLEMS]))
    if "audit.jsonl" in files:
        bad = 0
        for line in (scratch / "audit.jsonl").read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                if not isinstance(json.loads(line), dict):
                    bad += 1
            except ValueError:
                bad += 1
        if bad:
            problems.append(f"restored audit log: {bad} line(s) are not JSON objects")
    else:
        notes.append("the backup has no audit.jsonl")
    if DRAFTS_HASHES not in files:
        notes.append(f"the backup has no {DRAFTS_HASHES} (no drafts existed, or an older backup): draft hashes not checked")
    elif "ledger.jsonl" in files:
        try:
            hashes = json.loads((scratch / DRAFTS_HASHES).read_text(encoding="utf-8"))
            drafts = Ledger(scratch / "ledger.jsonl").drafts()
        except (ValueError, SocialError, OSError) as e:
            problems.append(f"drafts check: {e}")
            return
        for did, d in sorted(drafts.items()):
            if d.get("state") not in ("approved", "posting"):
                continue
            got = (hashes.get(did) or {}).get("sha256")
            if got is None:
                problems.append(f"draft {did} is {d['state']} but the backup has no usable hash of its folder ({(hashes.get(did) or {}).get('error', 'folder missing')})")
            elif got != d.get("approved_hash"):
                problems.append(f"draft {did} is {d['state']}: its folder hashes to {got[:12]}..., the approval was bound to {str(d.get('approved_hash'))[:12]}...")


# --------------------------------------------------------------------------------------------------- torn tail

def repair_tail(ledger_path: Path | str, *, now: dt.datetime | None = None) -> Path | None:
    """Move a torn final line into <ledger>.torn-<time> (after a backup); a last record without a newline just gets its newline.
    No other line is touched. Returns the torn file, or None when there was nothing to move. Refuses when other problems exist."""
    path = Path(ledger_path)
    now = now or _now()
    report = check_integrity(path)
    tail_kinds = {"torn_tail", "missing_newline"}
    others = [p for p in report.problems if p.kind not in tail_kinds]
    if others:
        raise IntegrityError(f"{path} has problems other than a torn last line, which are not repaired automatically: {report.summary()}")
    if not report.problems:
        return None
    with ledger_lock(path):
        make_backup(sources=backup_sources(path), now=now, keep=KEEP_DEFAULT)
        raw = path.read_bytes()
        cut = raw.rfind(b"\n") + 1
        if report.torn_tail:
            torn = path.with_name(f"{path.name}.torn-{_stamp(now)}")
            torn.write_bytes(raw[cut:])
            fixed = raw[:cut]
        else:
            torn = None
            fixed = raw + b"\n"
        tmp = path.with_name(path.name + ".repair-tmp")
        try:
            tmp.write_bytes(fixed)
            os.replace(tmp, path)
        except PermissionError as e:
            tmp.unlink(missing_ok=True)
            raise SocialError(f"cannot replace {path}: another program has it open ({e.strerror or e})") from e
    return torn


# --------------------------------------------------------------------------------------------------- commands

def cmd_social_integrity(args: argparse.Namespace) -> int:
    path = Path(settings.SOCIAL_LEDGER)
    report = check_integrity(path)
    if args.accept_duplicate is not None:
        if args.repair_tail:
            raise SocialError("--accept-duplicate and --repair-tail are separate steps; run them one at a time")
        entry = accept_duplicate(report, args.accept_duplicate, args.reason or "")
        print(f"accepted the duplicate post of observation {args.accept_duplicate} ({', '.join(entry['drafts'])}): {entry['reason']}")
        report = check_integrity(path)
        print(f"ledger {path}: {len(report.problems)} problems left")
        return 0 if report.ok else 1
    print(f"ledger {path}: {report.lines} records, {len(report.problems)} problems")
    for p in report.problems:
        print(f"  {p.text()}")
    if args.repair_tail:
        torn = repair_tail(path)
        if torn is not None:
            print(f"moved the torn last line to {torn} (a backup was made first)")
        elif report.ok:
            print("nothing to repair")
        else:
            print("added the missing newline (a backup was made first)")
        report = check_integrity(path)
        print(f"after repair: {len(report.problems)} problems")
    return 0 if report.ok else 1


def cmd_social_backup(args: argparse.Namespace) -> int:
    folder = make_backup(keep=args.keep)
    manifest = read_manifest(folder)
    print(f"backup {folder}")
    for name, info in manifest["files"].items():
        print(f"  {name}  {info['bytes']} bytes  sha256 {info['sha256'][:12]}")
    return 0


def cmd_social_restore(args: argparse.Namespace) -> int:
    if args.list or not args.source:
        for f in reversed(backups()):
            try:
                m = read_manifest(f)
                print(f"{f.name}  {len(m['files'])} files  ledger {'ok' if m.get('ledger_ok') else 'CORRUPT'}  {', '.join(m['files'])}")
            except IntegrityError as e:
                print(f"{f.name}  unreadable: {e}")
        return 0
    if args.apply and not args.yes:
        raise SocialError("--apply replaces the live files; add --yes to confirm")
    result = restore(args.source, apply=args.apply)
    for fname, what in result:
        print(f"  {fname}: {what}")
    if not args.apply:
        print("dry run: nothing changed (add --apply --yes to restore)")
    return 0


def cmd_social_restore_check(args: argparse.Namespace) -> int:
    res = restore_check(args.source, scratch_root=Path(args.scratch) if args.scratch else None)
    print(f"restore check of backup {res['name']}: {'ok' if res['ok'] else 'FAILED'} ({len(res['files'])} files restored to a scratch folder and removed)")
    for p in res["problems"]:
        print(f"  problem: {p}")
    for n in res["notes"]:
        print(f"  note: {n}")
    return 0 if res["ok"] else 1


COMMANDS = {"social-integrity": cmd_social_integrity, "social-backup": cmd_social_backup, "social-restore": cmd_social_restore,
            "social-restore-check": cmd_social_restore_check}
ERRORS = (IntegrityError,)


def add_parsers(sub) -> None:
    p = sub.add_parser("social-integrity", help="Check the posting ledger (valid lines, allowed state changes, no observation posted twice).")
    p.add_argument("--repair-tail", action="store_true", help="move a torn last line to <ledger>.torn-<time> (after a backup)")
    p.add_argument("--accept-duplicate", metavar="OBS_ID", default=None, help="acknowledge the known duplicate post of this observation (needs --reason)")
    p.add_argument("--reason", default=None, help="why the duplicate post is known and accepted (with --accept-duplicate)")
    p = sub.add_parser("social-backup", help="Copy the ledger, audit log, rules, themes and prints into the backup folder (dated; keeps the newest N).")
    p.add_argument("--keep", type=int, default=KEEP_DEFAULT, help=f"backups to keep (default {KEEP_DEFAULT})")
    p = sub.add_parser("social-restore", help="List backups, or show (default) / apply a restore from one.")
    p.add_argument("--list", action="store_true", help="list the backups")
    p.add_argument("--from", dest="source", default=None, help="backup name (folder in the backup dir)")
    p.add_argument("--apply", action="store_true", help="really restore (a pre-restore backup is made first)")
    p.add_argument("--yes", action="store_true", help="confirm --apply")
    p = sub.add_parser("social-restore-check", help="Restore the newest (or --from) backup into a scratch folder and validate it; touches no live file; exit 0 ok, 1 problems.")
    p.add_argument("--from", dest="source", default=None, help="backup name (default the newest)")
    p.add_argument("--scratch", default=None, help="folder to restore into (default <cache dir>/tmp; a temp subfolder, deleted afterwards)")


def register_hooks() -> None:
    from . import social_hooks

    social_hooks.register("ledger_integrity", "pre_send", ledger_gate, rule="ledger_integrity", priority=2)
    social_hooks.register("duplicate_post", "pre_send", duplicate_gate, rule="ledger_integrity", priority=3)
    social_hooks.register("daily_backup", "maintenance", daily_backup, rule="ledger_integrity", priority=90)
