"""Daily Bluesky posts of the owner's own photos (folded in from github.com/rjmendez/inat_skeet).

Source = the gallery (settings.GALLERY_DIR, <year>/<year-month>/), not iNat's 2048 px copies. Every day draws from the CURRENT month of
every year ("October" = 2024-10, 2025-10, ...). One post = one iNat observation: up to 4 of its gallery files, its live-fetched taxon,
date, public place and link, hashtags from its iconic taxon, and per image alt text written by a LOCAL vision model (Ollama, GPU only)
that sees the picture together with the label embedded in the file (title, caption, keywords) and the observation's data.

Nothing is posted that the owner has not approved:
  build_obs_index  gallery file -> observation id, from the audit/candidate csvs, each csv row checked against the title embedded in
                   the file (a row whose title differs is not evidence; two observations left = ambiguous, not guessed)
  draft_month      tops the month's unposted drafts up to keep_pending: SOCIAL_DIR/drafts/<id>/ (draft.json + prepared images)
  approve/reject   the owner's decision, bound to a hash of the draft's files (an edit after approval needs a new approval)
  post_next        posts the oldest approved draft of the current month, at most posts_per_day per local day; dry-run unless live
Every state change is one line in an append-only ledger (settings.SOCIAL_LEDGER). A post whose outcome is unknown ("posting" with no
result: the send raised or the process died mid-call) blocks all posting until the owner resolves it (resolve()), so nothing is ever
sent twice; only a failed login, before anything is sent, is recorded as "failed".
"""
from __future__ import annotations

import base64
import contextlib
import csv
import datetime as dt
import hashlib
import io
import json
import os
import random
import shutil
import re
import subprocess
import tempfile
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Iterator

from PIL import Image, ImageOps

from . import settings, social_hooks
from .social_rules import Audit, Rules, load_rules

MAX_IMAGES = 4                     # app.bsky.embed.images: images maxLength 4
TEXT_MAX_GRAPHEMES = 300           # app.bsky.feed.post text: maxGraphemes 300, maxLength 3000 (UTF-8 bytes)
TEXT_MAX_BYTES = 3000
MAX_CORRECTIONS = 2                # follow-up requests when an answer misses the owner's length band or uses mood words
MIN_ALT_CHARS = 15                # shorter than the owner's shortest real alt texts (p10 49): not a description
_TRAILING_LINK = re.compile(r"[\s,;:-]*(?:\b(?:photographed|taken|captured|shot|seen|observed|found)\b)?[\s,]*(?:\b(?:on|in|at|near|around|by|from)\b)?[\s,;:-]*$",
                            re.IGNORECASE)
EMBED_BATCH = 64                  # texts per Ollama /api/embed request
ALT_MAX_CHARS = 2000              # the Bluesky app's alt-text limit (the lexicon sets none)
INAT_OBS_URL = "https://www.inaturalist.org/observations/{}"

COMMON_TAGS = ("macrophotography", "nature", "inaturalist")
# inat_skeet's per-group hashtags, as data: an iconic taxon not listed here gets the common tags and its own name only
ICONIC_TAGS = {
    "Insecta": ("insects", "inverts", "bugs"),
    "Animalia": ("animals",),
    "Aves": ("birds",),
    "Amphibia": ("amphibians", "herps"),
    "Reptilia": ("reptiles", "herps"),
    "Mammalia": ("mammals",),
    "Actinopterygii": ("fish",),
    "Mollusca": ("molluscs",),
    "Arachnida": ("arachnids",),
    "Plantae": ("plants",),
    "Fungi": ("fungi", "mycology", "mushrooms", "fungifriends"),
    "Protozoa": ("protozoa",),
}
GRADE_TEXT = {"research": "Research Grade", "needs_id": "Needs ID"}
STATES = ("drafted", "approved", "rejected", "posting", "posted", "failed", "withdrawn")    # withdrawn: taken back, its observations are free again
NOTES = ("error",)    # recorded, not a state change


class SocialError(Exception):
    """A social-post step cannot go on; the message says what to do."""


class VisionError(SocialError):
    """The local vision model is missing, failed, or did not run fully on the GPU."""


class SkipObservation(SocialError):
    """This observation cannot be posted (casual, deleted, no taxon); the next candidate is tried."""


class HookRejected(SkipObservation):
    """A "draft" hook (a gate) rejected this observation's photo or alt text: it counts against the run's RejectBudget."""


MAX_DRAFT_REJECTS = 25               # draft-hook rejects one social-draft run may spend before it stops (each costs a GPU describe, an iNat fetch, a prepare)


class RejectBudget:
    """The draft-hook rejects a run may still spend; shared by the ordinary and the themed drafting of one social-draft run."""

    def __init__(self, limit: int = MAX_DRAFT_REJECTS):
        self.limit, self.used = limit, 0

    def charge(self) -> None:
        self.used += 1

    @property
    def exhausted(self) -> bool:
        return self.used >= self.limit


class UnusableAnswer(SkipObservation):
    """The vision model's answer is not alt text (too short, or JSON it could not be read from): the observation is skipped this run."""


# --------------------------------------------------------------------------------------------------- gallery file -> observation index

def _exiftool() -> str:
    from .inat_stock import _exiftool_binary

    return _exiftool_binary()


def read_titles(paths: list[Path]) -> dict[Path, str]:
    """{path: embedded XMP-dc title ('' when the file has none)} for many files in one exiftool run (argfile, UTF-8). A file exiftool skipped
    (locked, timed out, empty, corrupt: a record carrying an Error) is absent from the result: that is a miss, not an empty title."""
    if not paths:
        return {}
    with tempfile.NamedTemporaryFile("w", suffix=".args", delete=False, encoding="utf-8", newline="\n") as af:
        af.write("\n".join(str(p) for p in paths) + "\n")
    try:
        r = subprocess.run([_exiftool(), "-j", "-charset", "utf8", "-charset", "filename=utf8", "-XMP-dc:Title", "-Error", "-@", af.name],
                           capture_output=True)
    finally:
        os.unlink(af.name)
    if not r.stdout.strip():
        raise SocialError(f"exiftool read no titles: {r.stderr.decode('utf-8', 'replace').strip()[:200]}")
    by_name = {}
    for d in json.loads(r.stdout.decode("utf-8")):
        if d.get("Error"):
            continue                                     # empty, locked or corrupt: exiftool prints the error only on stderr and a record with no Title (a miss, never '')
        t = d.get("Title")
        by_name[os.path.normcase(os.path.normpath(d["SourceFile"]))] = "" if t is None else str(t)
    got = {p: by_name[os.path.normcase(os.path.normpath(str(p)))] for p in paths if os.path.normcase(os.path.normpath(str(p))) in by_name}
    return got                                           # a file exiftool did not report is left OUT (a miss, retried next run), never ''



def _render_name(row: dict) -> str:
    from .stock_select import render_name

    return render_name(row)


def _load_title_cache(path: Path) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        entries = data["titles"]
        return entries if isinstance(entries, dict) else {}
    except (OSError, ValueError, KeyError, TypeError):
        return {}                                                # a missing or damaged cache only costs a full read


def _cached_titles(gallery_dir: Path, files: list[Path], wanted: list[Path], titles, cache_path: Path) -> dict[Path, str]:
    entries = _load_title_cache(cache_path)
    stats = {}
    for p in wanted:
        st = p.stat()
        stats[p] = (st.st_size, st.st_mtime_ns)
    out: dict[Path, str] = {}
    stale: list[Path] = []
    for p in wanted:
        e = entries.get(p.relative_to(gallery_dir).as_posix())
        if isinstance(e, dict) and (e.get("size"), e.get("mtime_ns")) == stats[p] and isinstance(e.get("title"), str):
            out[p] = e["title"]
        else:
            stale.append(p)
    read: dict[Path, str] = titles(stale) if stale else {}
    out.update(read)
    present = {p.relative_to(gallery_dir).as_posix() for p in files}
    fresh = {rel: e for rel, e in entries.items() if rel in present}                      # a deleted file is dropped
    for p in stale:
        rel = p.relative_to(gallery_dir).as_posix()
        if p in read:                                                  # only a successful read is cached; a miss is retried next run
            fresh[rel] = {"size": stats[p][0], "mtime_ns": stats[p][1], "title": read[p]}
        else:
            fresh.pop(rel, None)
    if fresh != entries or not Path(cache_path).is_file():
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
        tmp.write_text(json.dumps({"version": 1, "titles": fresh}, sort_keys=True), encoding="utf-8")
        os.replace(tmp, cache_path)
    return out


def build_obs_index(gallery_dir: Path, audit_csvs: list[Path], candidate_csvs: list[Path], *,
                    titles: Callable[[list[Path]], dict[Path, str]] = read_titles, title_cache: Path | None = None) -> dict:
    """{"files": {"<year>/<year-month>/<name>": obs_id}, "unresolved": {relpath: reason}} for every .jpg in the gallery.

    Evidence: an audit csv row (file, obs_id: the owner's Lightroom exports, one row per file) and a candidate csv row (a published Stock
    candidate: render_name(row) is its gallery name, obs_id, title). A candidate row counts only when its title equals the title embedded in
    the gallery file -- older matching passes left rows that point the same name at another observation. Exactly one observation left =
    resolved; none or several = unresolved with the reason.

    title_cache (a JSON file): the embedded titles already read, keyed by relative path, size and mtime_ns, so a run reads only the files that are
    new or changed (exiftool over the whole gallery took minutes); entries of files that are gone are dropped. The result is the same as without it."""
    gallery_dir = Path(gallery_dir)
    files = sorted(p for p in gallery_dir.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg") and p.is_file())
    by_name: dict[str, list[Path]] = {}
    for p in files:
        by_name.setdefault(p.name, []).append(p)
    evidence: dict[Path, list[tuple[int, str | None]]] = {}

    def add(name: str, obs: str, title: str | None) -> None:
        if not obs or name not in by_name:
            return
        for p in by_name[name]:
            evidence.setdefault(p, []).append((int(obs), title))

    for c in audit_csvs:
        with open(c, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                add(r["file"], r.get("obs_id", ""), None)
    for c in candidate_csvs:
        with open(c, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if r.get("stock_candidate") == "True" and r.get("obs_id") and r.get("raw") and r.get("day") and r.get("id"):
                    add(_render_name(r), r["obs_id"], r.get("title", ""))
    wanted = [p for p in files if any(t is not None for _, t in evidence.get(p, []))]
    embedded = _cached_titles(gallery_dir, files, wanted, titles, title_cache) if title_cache else titles(wanted)
    out = {"files": {}, "unresolved": {}}
    for p in files:
        rel = p.relative_to(gallery_dir).as_posix()
        ev = evidence.get(p, [])
        kept = {obs for obs, t in ev if t is None or t == embedded.get(p)}
        if len(kept) == 1:
            out["files"][rel] = kept.pop()
        elif kept:
            out["unresolved"][rel] = "ambiguous: observations " + ", ".join(str(o) for o in sorted(kept))
        elif ev:
            out["unresolved"][rel] = "every csv row's title differs from the one embedded in the file"
        else:
            out["unresolved"][rel] = "no csv row names this file"
    return out


def write_index(index: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(index, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


PHOTO_TOKEN = "inat-photo:"                                      # a candidate photo that is not on disk yet: inat-photo:<iNat photo id>
OPEN_DATA_URL = "https://inaturalist-open-data.s3.amazonaws.com/photos/{id}/{size}.jpg"
PHOTO_SIZES = ("original", "large")                              # best first
PHOTO_PAUSE_S = 1.0                                              # between two downloads from iNat's open data


def inat_photo_candidates(snapshot: Path | None = None, *, min_edge: int, exclude=frozenset()) -> dict[int, dict]:
    """The best iNat photo of each observation the gallery does not have: {obs id: {"photo_id", "edge", "observed_on"}} for the observations of
    the owner's iNat snapshot (settings.INAT_SNAPSHOT) not in `exclude`, with the biggest photo by its original dimensions, when that is at
    least `min_edge` px on the long side and licensed (iNat's open data holds only the licensed ones). {} without the snapshot."""
    snapshot = Path(snapshot or settings.INAT_SNAPSHOT)
    if not snapshot.is_file():
        return {}
    out: dict[int, dict] = {}
    for o in json.loads(snapshot.read_text(encoding="utf-8")).get("observations") or []:
        oid = int(o["id"])
        if oid in exclude:
            continue
        best = None
        for p in o.get("photos") or []:
            d = p.get("dims") or {}
            w, h = d.get("width") or 0, d.get("height") or 0
            if p.get("license") and (best is None or w * h > best[0]):
                best = (w * h, max(w, h), int(p["id"]))
        if best and best[1] >= min_edge:
            out[oid] = {"photo_id": best[2], "edge": best[1], "observed_on": o.get("observed_on")}
    return out


def _http_photo(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "bio-automation-photo-fetch/0.1 (owner photos)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def fetch_best_photo(photo_id: int, dest_dir: Path, *, min_edge: int, get: Callable[[str], bytes] | None = None,
                     sleep: Callable[[float], None] = time.sleep, pause_s: float = PHOTO_PAUSE_S) -> Path:
    """<dest_dir>/<photo id>.jpg: the biggest copy of the iNat photo in iNat's open data (original, else large) that is at least `min_edge`
    px on its long side, downloaded once and kept. SkipObservation when there is none (not in open data, too small, or the network failed)."""
    dest = Path(dest_dir) / f"{photo_id}.jpg"
    if dest.is_file():
        with contextlib.suppress(OSError):
            with Image.open(dest) as im:
                if max(im.size) >= min_edge:
                    return dest
    get = get or _http_photo
    for size in PHOTO_SIZES:
        url = OPEN_DATA_URL.format(id=photo_id, size=size)
        try:
            data = get(url)
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):                              # not in open data, or not in that size
                continue
            raise SkipObservation(f"iNat photo {photo_id}: HTTP {e.code} from {url}") from e
        except (urllib.error.URLError, OSError) as e:
            raise SkipObservation(f"iNat photo {photo_id}: {type(e).__name__}: {e}") from e
        finally:
            sleep(pause_s)
        try:
            with Image.open(io.BytesIO(data)) as im:
                edge = max(im.size)
        except (OSError, ValueError):
            continue
        if edge >= min_edge:
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".part")
            tmp.write_bytes(data)
            os.replace(tmp, dest)
            return dest
    raise SkipObservation(f"iNat photo {photo_id}: no copy of at least {min_edge} px in iNat's open data")


def resolve_photos(rels: list[str], fetch_photo: Callable[[int], Path] | None) -> list[str]:
    """`rels` with every inat-photo:<id> token replaced by the downloaded photo's path (fetch_photo raises SkipObservation when it cannot)."""
    if fetch_photo is None and any(r.startswith(PHOTO_TOKEN) for r in rels):
        raise SocialError("an iNat photo has to be fetched but no fetch_photo was given")
    return [str(fetch_photo(int(r[len(PHOTO_TOKEN):]))) if r.startswith(PHOTO_TOKEN) else r for r in rels]


def default_photo_fetch(rules: Rules) -> Callable[[int], Path]:
    min_edge = rules.value("lower_quality_photos") or 1000
    return lambda pid: fetch_best_photo(pid, settings.INAT_BEST_PHOTO_DIR, min_edge=min_edge)


def withdraw_small(ledger: "Ledger", drafts_dir: Path, *, min_edge: int, audit: Audit | None = None) -> list[str]:
    """Withdraw every pending (drafted or approved) draft with an image of fewer than `min_edge` px on its long side: it is taken back (state
    "withdrawn", reason in the ledger and the audit log) and its observations are free for a new draft. Returns the draft ids."""
    audit = audit or Audit()
    out = []
    for did, v in ledger.drafts().items():
        if v["state"] not in ("drafted", "approved"):
            continue
        try:
            draft = load_draft(Path(drafts_dir) / did)
        except (OSError, ValueError):
            continue
        small = {i["file"]: max(i.get("width") or 0, i.get("height") or 0) for i in draft["images"]
                 if max(i.get("width") or 0, i.get("height") or 0) < min_edge}
        if small:
            reason = f"photo(s) under {min_edge} px: " + ", ".join(f"{f} is {e}" for f, e in small.items())
            with ledger_lock_wait(ledger.path):                  # a post run may have moved the draft on since the listing above
                now_v = ledger.drafts().get(did)
                if now_v is None or now_v["state"] not in ("drafted", "approved"):
                    audit.log("withdraw_skipped", draft_id=did, state=now_v["state"] if now_v else None,
                              reason="the draft is no longer pending; it was not withdrawn")
                    continue
                ledger.append("withdrawn", did, obs_id=now_v["obs_id"], month=now_v["month"], reason=reason)
            audit.log("withdrawn", draft_id=did, obs_ids=_obs_of(v), reason=reason)
            out.append(did)
    return out


def load_index(path: Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise SocialError(f"no gallery index at {path}: run `bio social-index` first")
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------------------- choosing observations

def month_observations(index: dict, month: int) -> dict[int, list[str]]:
    """{obs_id: [relpath, ...]} for gallery files in a <year>-<MM> folder of `month` (1-12), any year; paths sorted."""
    suffix = f"-{month:02d}"
    out: dict[int, list[str]] = {}
    for rel, obs in index["files"].items():
        parts = rel.split("/")
        if len(parts) >= 2 and parts[-2].endswith(suffix):
            out.setdefault(int(obs), []).append(rel)
    return {o: sorted(v) for o, v in out.items()}


def order_candidates(cands: dict[int, list[str]], exclude: set[int], seed: str, *, rotate_years: bool = True,
                     years: dict[int, str] | None = None) -> list[int]:
    """Observations not in `exclude`, years taking turns (newest year first), shuffled within a year by `seed` -- the same seed gives the
    same order, so a re-run of one day's drafting proposes the same observations. Without `rotate_years`: one shuffled list. An
    observation's year is its gallery folder's, or `years[obs]` (for photos that are not in the gallery's year folders)."""
    rng = random.Random(seed)
    years = years or {}
    if not rotate_years:
        out = sorted(o for o in cands if o not in exclude)
        rng.shuffle(out)
        return out
    by_year: dict[str, list[int]] = {}
    for obs, rels in cands.items():
        if obs not in exclude:
            by_year.setdefault(years.get(obs) or min(r.split("/")[0] for r in rels), []).append(obs)
    queues = []
    for year in sorted(by_year, reverse=True):
        q = sorted(by_year[year])
        rng.shuffle(q)
        queues.append(q)
    out = []
    while any(queues):
        for q in queues:
            if q:
                out.append(q.pop(0))
    return out


# --------------------------------------------------------------------------------------------------- ledger

_CORRUPT_LOGGED: set[tuple] = set()               # (ledger, run, line, kind) already audited as ledger_corrupt
LEDGER_LOCK_WAIT_S = 60.0                         # a writer other than the post run waits this long for a running post to release the ledger lock
LEDGER_LOCK_POLL_S = 0.5
STALL_AFTER = 2                                   # consecutive NothingSent failures of one draft after which it is skipped for the rest of that day
STALL_FAIL_DAYS = 3                               # a draft that has been stalled this long is marked failed (the owner re-approves or rejects it)
MAX_WITHDRAWALS_PER_OBS = 3                       # pre_send gates may take back drafts of one observation this often; then it is no candidate any more
                                                  # (a draft that fails the gates every time would otherwise be redrafted and withdrawn forever, burning stock and slots)


class Ledger:
    """Append-only JSON lines: {"at", "event", "draft_id", ...}. A draft's state is its last event."""

    def __init__(self, path: Path | None = None, audit: Audit | None = None):
        self.path = Path(path or settings.SOCIAL_LEDGER)
        self.audit = audit                              # where a corrupt line is recorded (default: a new Audit() at that moment)

    def records(self) -> list[dict]:
        if not self.path.is_file():
            return []
        out = []
        text = self.path.read_text(encoding="utf-8")
        lines = text.splitlines()
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                kind = "torn_tail" if number == len(lines) and not text.endswith(chr(10)) else "bad_json"
                self._audit_corrupt(number, kind)
                fix = "`bio social-integrity --repair-tail`" if kind == "torn_tail" else "`bio social-integrity`"
                raise SocialError(f"the ledger {self.path} is corrupt at line {number} ({kind}: {e.msg}); posting is blocked until it is repaired; "
                                  f"nothing was changed; run {fix}") from e
        return out

    def _audit_corrupt(self, line: int, kind: str) -> None:
        """Audit ledger_corrupt (line number and kind only, no ledger content) once per run and ledger; the alerter reads it from the audit log."""
        key = (str(self.path), self.audit.run if self.audit is not None else "process", line, kind)
        if key in _CORRUPT_LOGGED:
            return
        _CORRUPT_LOGGED.add(key)
        reason = f"{kind} at line {line}"
        with contextlib.suppress(OSError):                 # the block does not depend on the audit line
            (self.audit or Audit()).log("ledger_corrupt", path=str(self.path), line=line, kind=kind, count=1, problems=[reason], reason=reason)

    def append_locked(self, event: str, draft_id: str, **fields) -> dict:
        """append() under the ledger's poster lock (waiting up to LEDGER_LOCK_WAIT_S for a running post), for writers other than the post run."""
        with ledger_lock_wait(self.path):
            return self.append(event, draft_id, **fields)

    def withdrawal_counts(self) -> dict[int, tuple[int, str]]:
        """{observation id: (how many times a pre_send hook took a draft of it back, the last reason)}, from the ledger's withdrawn records
        that carry a `hook` (the withdrawal a gate made at send time); the observations are those of the draft's latest drafted record."""
        obs_of: dict[str, list[int]] = {}
        out: dict[int, tuple[int, str]] = {}
        for r in self.records():
            if r["event"] == "drafted":
                obs_of[r["draft_id"]] = _obs_of(r)
            elif r["event"] == "withdrawn" and r.get("hook"):
                for o in obs_of.get(r["draft_id"], []):
                    out[o] = (out.get(o, (0, ""))[0] + 1, str(r.get("reason") or ""))
        return out

    def nothing_sent_streaks(self) -> dict[str, dict]:
        """{draft_id: {"count", "first", "last"}}: the "error" notes (a send that sent nothing) of each draft since its last drafted, owner
        approval, failed, posted, rejected or withdrawn record. A restored approval (the send's own undo) does not end the streak."""
        out: dict[str, dict] = {}
        for r in self.records():
            did, ev = r["draft_id"], r["event"]
            if ev == "error":
                s = out.setdefault(did, {"count": 0, "first": r["at"], "last": r["at"]})
                s["count"] += 1
                s["last"] = r["at"]
            elif ev in ("drafted", "failed", "posted", "rejected", "withdrawn") or (ev == "approved" and not r.get("restored")):
                out.pop(did, None)
        return out

    def append(self, event: str, draft_id: str, *, now: dt.datetime | None = None, **fields) -> dict:
        if event not in STATES + NOTES:
            raise SocialError(f"unknown ledger event {event!r}")
        rec = {"at": (now or dt.datetime.now(settings.LOCAL_TZ)).isoformat(timespec="seconds"), "event": event, "draft_id": draft_id, **fields}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lead = self._lead_newline()
        with open(self.path, "a", encoding="utf-8", newline="\n") as f:
            f.write(lead + json.dumps(rec, sort_keys=True) + "\n")
        return rec

    def _lead_newline(self) -> str:
        """"" when the file is empty, absent or ends with a newline. When the last record has no newline: "\n" if that record is complete (the
        new one then starts its own line), SocialError if it is torn (appending would bury a damaged line in the middle of the ledger)."""
        try:
            size = self.path.stat().st_size
        except OSError:
            return ""
        if size == 0:
            return ""
        with open(self.path, "rb") as f:
            f.seek(max(0, size - 1_000_000))
            tail = f.read()
        if tail.endswith(b"\n"):
            return ""
        last = tail.rsplit(b"\n", 1)[-1]
        try:
            json.loads(last.decode("utf-8"))
        except ValueError as e:
            raise SocialError(f"the ledger {self.path} ends with a torn record (no newline); nothing was appended; run `bio social-integrity --repair-tail`") from e
        return "\n"

    def drafts(self) -> dict[str, dict]:
        """{draft_id: {"state", "obs_id", "month", "drafted_at", "last": <last state record>, "seq": its ledger position, "approved_hash",
        "approved_seq": position of the owner's approval, "last_error": the last error note}}."""
        out: dict[str, dict] = {}
        for seq, r in enumerate(self.records()):
            d = out.setdefault(r["draft_id"], {"obs_id": r.get("obs_id"), "month": r.get("month"), "drafted_at": r["at"], "approved_hash": None,
                                               "obs_ids": r.get("obs_ids"), "for_date": r.get("for_date"), "theme": r.get("theme")})
            if r["event"] in NOTES:
                d["last_error"] = r
                continue
            if r["event"] == "drafted":            # a (re)draft under this id (after a withdrawal): its observations, day and theme are the latest record's
                d.update(obs_id=r.get("obs_id"), month=r.get("month"), drafted_at=r["at"], approved_hash=None, obs_ids=r.get("obs_ids"),
                         for_date=r.get("for_date"), theme=r.get("theme"))
                d.pop("approved_seq", None)
            d["state"], d["last"], d["seq"] = r["event"], r, seq       # seq: ledger position (timestamps tie within a second)
            if r["event"] == "approved":
                d["approved_hash"] = r.get("hash")
                if not r.get("restored"):              # an approval given back after a send that sent nothing keeps its place in line
                    d["approved_seq"] = seq
        return out


def draft_hash(draft_dir: Path) -> str:
    """sha256 over draft.json and every image the draft names, so approval covers exactly what will be sent."""
    draft_dir = Path(draft_dir)
    h = hashlib.sha256()
    raw = (draft_dir / "draft.json").read_bytes()
    h.update(raw)
    for img in json.loads(raw)["images"]:
        h.update((draft_dir / img["file"]).read_bytes())
    return h.hexdigest()


def load_draft(draft_dir: Path) -> dict:
    return json.loads((Path(draft_dir) / "draft.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------------------- images

def prepare_image(src: Path, dst: Path, *, max_bytes: int | None = None, max_edge: int | None = None) -> dict:
    """A JPEG copy of `src` for Bluesky: upright, sRGB pixels, long edge <= max_edge, at most max_bytes, NO metadata (EXIF/XMP/GPS are not
    copied; the ICC profile is kept so colours stay right). Lowers the quality 90 -> 60, then shrinks the size, until it fits.
    Returns {"width", "height", "bytes", "quality"}."""
    max_bytes = max_bytes or settings.BSKY_IMAGE_MAX_BYTES
    max_edge = max_edge or settings.BSKY_IMAGE_MAX_EDGE
    with Image.open(src) as im:
        icc = im.info.get("icc_profile")
        im = ImageOps.exif_transpose(im).convert("RGB")
    edge = max_edge
    while edge >= 256:
        work = im.copy()
        work.thumbnail((edge, edge), Image.LANCZOS)
        for q in (90, 85, 80, 75, 70, 65, 60):
            buf = io.BytesIO()
            work.save(buf, "JPEG", quality=q, optimize=True, **({"icc_profile": icc} if icc else {}))
            if buf.tell() <= max_bytes:
                Path(dst).write_bytes(buf.getvalue())
                return {"width": work.width, "height": work.height, "bytes": buf.tell(), "quality": q}
        edge = int(edge * 0.85)
    raise SocialError(f"{src}: no JPEG of at least 256 px fits {max_bytes} bytes")


# --------------------------------------------------------------------------------------------------- local vision model

def _ollama(path: str, body: dict | None = None, *, timeout: float = 600) -> dict:
    req = urllib.request.Request(settings.OLLAMA_URL + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise VisionError(f"Ollama {path}: HTTP {e.code} {e.read()[:200]!r}") from e
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        raise VisionError(f"Ollama at {settings.OLLAMA_URL} unreachable: {e}") from e


def pool_model(role: str, env_name: str, *, api: Callable = _ollama, pool_file: Path | None = None) -> str:
    """The model for a role: env `env_name`, else the best-ranked (lowest rank) model of the pool file whose roles include `role` and
    that Ollama has installed. Never a literal tag; nothing found raises."""
    named = os.environ.get(env_name)
    if named:
        return named
    pool_file = Path(pool_file or settings.MODEL_POOL_FILE)
    if not pool_file.is_file():
        raise VisionError(f"no {role} model: set {env_name} or give the pool file ({pool_file}) a '{role}' role")
    pool = tomllib.loads(pool_file.read_text(encoding="utf-8")).get("models", {}).get("pool", [])
    installed = {m["name"] for m in api("/api/tags").get("models", [])}
    ranked = sorted((e.get("rank", 99), e["name"]) for e in pool if role in e.get("roles", []) and e.get("name") in installed)
    if not ranked:
        raise VisionError(f"no installed model has the '{role}' role in {pool_file}; set {env_name}")
    return ranked[0][1]


def vision_model(*, api: Callable = _ollama, pool_file: Path | None = None) -> str:
    return pool_model("vision", settings.VISION_MODEL_ENV, api=api, pool_file=pool_file)


def _require_gpu(model: str, api: Callable) -> None:
    loaded = {m["name"]: m for m in api("/api/ps").get("models", [])}
    m = loaded.get(model)
    if m is None or m.get("size_vram", 0) < m.get("size", 0):
        raise VisionError(f"{model} is not fully on the GPU (loaded: {m and (m.get('size_vram'), m.get('size'))}); refusing CPU inference")


def embedder(*, api: Callable = _ollama, pool_file: Path | None = None) -> tuple[str, Callable[[list[str]], list[list[float]]]]:
    """(model, embed(texts) -> vectors) with the pool's "embed" model on Ollama; GPU only, like the vision model."""
    model = pool_model("embed", settings.EMBED_MODEL_ENV, api=api, pool_file=pool_file)

    def embed(texts: list[str]) -> list[list[float]]:
        out = []
        for i in range(0, len(texts), EMBED_BATCH):   # one request with all 275 alt texts failed every time (2026-10-22); <= 128 worked
            out += api("/api/embed", {"model": model, "input": list(texts[i:i + EMBED_BATCH]), "keep_alive": "10m"}).get("embeddings") or []
        _require_gpu(model, api)
        if len(out) != len(texts):
            raise VisionError(f"{model} returned {len(out)} embeddings for {len(texts)} texts")
        return out

    return model, embed


def alt_text_prompt(context: dict) -> str:
    """Everything known about the photo, then the instruction. The identification comes from the observation, not from the model."""
    lines = ["Known about this photo (from the photographer's records):"]
    for key, label in (("common_name", "Common name"), ("scientific_name", "Scientific name"), ("rank", "Identified to rank"),
                       ("grade", "iNaturalist grade"), ("observed_on", "Date"), ("place", "Place"), ("title", "Stock title"),
                       ("caption", "Stock caption")):
        if context.get(key):
            lines.append(f"- {label}: {context[key]}")
    if context.get("keywords"):
        lines.append("- Stock keywords: " + ", ".join(context["keywords"]))
    lines += ["", "Write alt text for this photograph for someone who cannot see it. Rules:",
              "- Start with the subject itself, named as above (e.g. 'A spotted lanternfly perched on...'). Never start with 'The image' or 'This photo'.",
              "- Describe only what is visible: the organism or part shown, pose, colours, what it is on, background, light.",
              "- Use the facts above to name things, but add no other facts (no habitat, behaviour or species traits that are not visible).",
              "- Do not repeat the date or the place: the post already shows them.",
              "- Answer with the alt text only, as plain text: no JSON, no quotes, no labels."]
    name = context.get("common_name") or context.get("scientific_name")
    if name:
        lines.append(f"- Call the organism '{name}' (the photographer's identification), not a vaguer word like 'growth' or 'plant'.")
    style = context.get("style")
    if style:
        lines += ["", "Write it the way the photographer writes alt text (learned from their own posts):"]
        lines += [f"- {r}" for r in style["rules"]]
        lines += ["", "The photographer's own alt texts for similar subjects (match this voice and length, not their content):"]
        lines += [f"- {e}" for e in style["examples"]]
    else:
        lines.append("- One to three sentences, under 350 characters.")
    return "\n".join(lines)


_OPENER = re.compile(r"^(?:the|this)\s+(?:image|photo|photograph|picture)\s+(?:shows|captures|depicts|features|is of)\s+", re.IGNORECASE)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def clean_alt(text: str, context: dict) -> str:
    """The model's answer, tidied the same way every time: whitespace collapsed, a leading "The image shows" removed, and sentences that
    restate the date or the place (the post text carries both) dropped. Nothing else is rewritten; an answer that would end up empty is
    kept whole. The vision model sometimes answers with a JSON object ({"scene": "...", "confidence": 0.99}); then its longest string
    value is the answer."""
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            obj = None
        strings = [v for v in obj.values() if isinstance(v, str) and v.strip()] if isinstance(obj, dict) else []
        if strings:
            text = max(strings, key=len)
    text = " ".join(text.split())
    sentences = [_OPENER.sub("", x) for x in _SENTENCE.split(text)] if text else []
    sentences = [x[:1].upper() + x[1:] for x in sentences if x]
    facts = place_and_date_forms(context)
    kept = []
    for x in sentences:
        x = _SPACE_COMMA.sub(",", x)
        low = x.lower()
        hits = [low.find(f) for f in facts if f in low]
        if not hits:
            kept.append(x)
            continue
        head = _TRAILING_LINK.sub("", x[:min(hits)])       # "..., photographed on October 9, 2025, in ..." -> "..."
        if len(head) >= MIN_ALT_CHARS:
            kept.append(head + ".")
    return " ".join(kept or sentences)


_SPACE_COMMA = re.compile(r"\s+,")


def place_and_date_forms(context: dict) -> list[str]:
    """Lower-case forms of the date and place a sentence must not repeat: the date, the place with "X , Y" spacing normalised (the
    Stock captions carry "SGL220 , PA, USA", the model writes "SGL220, PA, USA"), and the place's first part when it is specific
    (4+ characters)."""
    out = []
    if context.get("observed_on"):
        out.append(context["observed_on"].lower())
    place = _SPACE_COMMA.sub(",", (context.get("place") or "").strip())
    if place:
        out.append(place.lower())
        first = place.split(",")[0].strip()
        if len(first) >= 4 and first.lower() != place.lower():
            out.append(first.lower())
    return out


def describe(image: Path, context: dict, *, model: str, api: Callable = _ollama) -> str:
    """Alt text for `image` from the local vision model. Refuses an empty answer, and refuses when Ollama holds the model partly in system
    memory (CPU inference is not allowed)."""
    with Image.open(image) as im:
        im = im.convert("RGB")
        im.thumbnail((settings.VISION_IMAGE_EDGE, settings.VISION_IMAGE_EDGE), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=90)
    messages = [{"role": "user", "content": alt_text_prompt(context), "images": [base64.b64encode(buf.getvalue()).decode()]}]

    def ask() -> str:
        r = api("/api/chat", {"model": model, "stream": False, "think": False, "keep_alive": "10m", "options": {"temperature": 0},
                              "messages": messages})
        _require_gpu(model, api)
        return r.get("message", {}).get("content") or ""

    raw = ask()
    text = clean_alt(raw, context)
    attempts = context.get("attempts")          # a list the caller passes in to see every answer
    if attempts is not None:
        attempts.append(text)
    if not text:
        raise VisionError(f"{model} returned no alt text for {image}")
    if not usable(text):
        raise UnusableAnswer(f"{model} gave no usable alt text for {image.name}: {text[:60]!r}")
    band = (context.get("style") or {}).get("band")
    if not band:
        return text[:ALT_MAX_CHARS]
    best = current = text
    for _ in range(MAX_CORRECTIONS):
        problems = style_problems(current, band)
        if not problems:
            break
        # a correction in the same conversation (the model still sees the photo), naming exactly what is off
        messages += [{"role": "assistant", "content": raw},
                     {"role": "user", "content": " ".join(problems) + " Write it again. Answer with the alt text only."}]
        raw = ask()
        current = clean_alt(raw, context)
        if attempts is not None:
            attempts.append(current)
        if not usable(current):
            break                                    # an unusable follow-up ends the corrections; the best answer so far stands
        if style_cost(current, band) < style_cost(best, band):
            best = current
    return best[:ALT_MAX_CHARS]


def style_problems(text: str, band: list[int]) -> list[str]:
    """What is off about an answer against the owner's style: length outside the band, mood words."""
    from .social_style import _MOOD_WORD

    lo, hi = band
    out = []
    if len(text) < lo:
        out.append(f"That is {len(text)} characters; the photographer's alt texts are usually {lo}-{hi}. Make it longer: add what else is "
                   "visible (colours, textures, the parts of the organism you can see, what it is on, the background), still plain, nothing "
                   "that is not visible.")
    elif len(text) > hi:
        out.append(f"That is {len(text)} characters; the photographer's alt texts are usually {lo}-{hi}. Make it shorter: keep the subject "
                   "and the most visible details.")
    words = sorted({m.group(0).lower() for m in _MOOD_WORD.finditer(text)})
    if words:
        out.append(f"Drop {', '.join(repr(w) for w in words)}: the photographer does not use words like that.")
    return out


def style_cost(text: str, band: list[int]) -> int:
    """Lower is closer to the owner's style: characters outside the band, plus a large cost per mood word."""
    from .social_style import _MOOD_WORD

    return band_distance(len(text), band) + 100 * len(_MOOD_WORD.findall(text))


def usable(text: str) -> bool:
    """Alt text, not a fragment or a JSON object the model failed to fill."""
    return len(text) >= MIN_ALT_CHARS and not text.lstrip().startswith(("{", "["))


def band_distance(n: int, band: list[int]) -> int:
    """How far a length is outside [lo, hi]; 0 inside."""
    lo, hi = band
    return lo - n if n < lo else n - hi if n > hi else 0


# --------------------------------------------------------------------------------------------------- post text

def format_date(observed_on: str | None) -> str | None:
    if not observed_on:
        return None
    d = dt.date.fromisoformat(observed_on[:10])
    return f"{d.strftime('%B')} {d.day}, {d.year}"


def public_place(obs: dict) -> str | None:
    """iNat's public place guess; nothing for a private-geoprivacy observation."""
    if obs.get("geoprivacy") == "private":
        return None
    return (obs.get("place_guess") or "").strip() or None


def display_name(taxon: dict) -> str:
    sci, common = taxon.get("name"), taxon.get("preferred_common_name") or taxon.get("common_name")
    common = common[:1].upper() + common[1:] if common else common      # iNat lowercases many plant names ("blue vervain")
    return f"{common} ({sci})" if common and sci else (sci or common or "")


def hashtags(iconic: str | None) -> list[str]:
    tags = list(COMMON_TAGS)
    if iconic:
        tags.append(iconic)
        tags += [t for t in ICONIC_TAGS.get(iconic, ()) if t.lower() not in {x.lower() for x in tags}]
    return tags


def text_segments(obs: dict, prints_url: str | None = None) -> list[list[str]]:
    """The post as [kind, text(, target)] segments: ["text", s], ["link", s, url], ["tag", "#t", "t"]. prints_url adds a "Prints:" line
    after the iNat link. While the text is over Bluesky's limits, tags are dropped from the end; if it is still over without any tag, the
    prints line goes and the tags are tried again; a post over them even then raises."""
    taxon = obs.get("taxon") or {}
    url = INAT_OBS_URL.format(obs["observation_id"])
    head = [display_name(taxon), GRADE_TEXT.get(obs.get("quality_grade") or "", ""),
            " · ".join(x for x in (format_date(obs.get("observed_on")), public_place(obs)) if x)]
    all_tags = hashtags(taxon.get("iconic_taxon_name"))
    for prints in ([[["text", "\nPrints: "], ["link", prints_url, prints_url]]] if prints_url else []) + [[]]:
        tags = list(all_tags)
        while True:
            out = ([["text", "\n".join(x for x in head if x) + "\n"], ["link", url, url]] + prints
                   + [["text", "\n"]] + [s for t in tags for s in (["tag", "#" + t, t], ["text", " "])][:-1])
            text = segments_text(out)
            if len(text) <= TEXT_MAX_GRAPHEMES and len(text.encode("utf-8")) <= TEXT_MAX_BYTES:   # code points >= graphemes: a safe count
                return out
            if not tags:
                break
            tags.pop()
    raise SocialError(f"observation {obs['observation_id']}: post text is {len(text)} characters even without hashtags")


def load_prints(path: Path | None = None) -> dict[int, str]:
    """{obs_id: print url} from the owner's csv (header obs_id,url); no file -> {}. A malformed row raises, naming its line."""
    path = Path(path or settings.SOCIAL_PRINTS_FILE)
    if not path.is_file():
        return {}
    out: dict[int, str] = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not {"obs_id", "url"} <= set(reader.fieldnames or ()):
            raise SocialError(f"{path}: needs the header obs_id,url")
        for row in reader:
            obs, url = (row["obs_id"] or "").strip(), (row["url"] or "").strip()
            where = f"{path} line {reader.line_num}"
            if not obs.isdigit() or not url.startswith("https://") or " " in url:
                raise SocialError(f"{where}: expected an observation id and an https:// url, got {obs!r}, {url!r}")
            if int(obs) in out and out[int(obs)] != url:
                raise SocialError(f"{where}: observation {obs} already has {out[int(obs)]}")
            out[int(obs)] = url
    return out


def segments_text(segs: list[list[str]]) -> str:
    return "".join(s[1] for s in segs)


# --------------------------------------------------------------------------------------------------- drafting

def _fetch(obs_id: int) -> dict:
    from .inat import fetch_inat_observation

    return fetch_inat_observation(obs_id)


def _read_label(path: Path) -> dict:
    from .inat_stock import read_label

    return read_label(path)


def make_draft(obs_id: int, rels: list[str], *, month: str, gallery_dir: Path, drafts_dir: Path, model: str,
               fetch: Callable[[int], dict] = _fetch, read_label: Callable[[Path], dict] = _read_label,
               describe_fn: Callable[..., str] = describe, style: Callable[[str], dict] | None = None,
               prints_url: str | None = None, now: dt.datetime | None = None, skip_casual: bool = True,
               rules: Rules | None = None, audit: Audit | None = None) -> dict:
    """Write drafts_dir/<month>-obs<obs_id>/ (draft.json + 1.jpg..4.jpg) and return the draft. The observation is fetched live: casual,
    deleted and taxon-less observations raise SkipObservation. style(subject) -> {"rules", "examples", ...}: the owner's alt-text style
    for this photo (social_style.style_for), given to the vision model and recorded per image. The "draft" hooks (social_hooks) see each
    image with its alt text; a reject raises HookRejected (a SkipObservation) and leaves no folder. Every gate must have loaded: library callers
    cannot bypass them by skipping the CLI."""
    from . import social_gates
    from .inat import InatNotFoundError

    _require_gates()
    rules, audit = rules or load_rules(), audit or Audit()
    try:
        obs = fetch(obs_id)
    except InatNotFoundError as e:
        raise SkipObservation(f"observation {obs_id} no longer exists on iNat") from e
    grade, taxon = obs.get("quality_grade"), obs.get("taxon") or {}
    if grade == "casual" and skip_casual:
        raise SkipObservation(f"observation {obs_id} is casual")
    if not taxon.get("name"):
        raise SkipObservation(f"observation {obs_id} has no taxon")
    draft_id = f"{month}-obs{obs_id}"
    out = Path(drafts_dir) / draft_id
    out.mkdir(parents=True, exist_ok=True)
    images = []
    for i, rel in enumerate(rels[:MAX_IMAGES], 1):
        src = Path(gallery_dir) / rel
        name = f"{i}.jpg"
        info = prepare_image(src, out / name)
        label = read_label(src)
        shown = label
        if rules.on("alt_text_privacy") and social_gates.location_hidden(obs):
            shown = social_gates.label_without_place(label, obs)         # a hidden place must not reach the vision model through the Stock label
        context = {"common_name": taxon.get("preferred_common_name") or taxon.get("common_name"), "scientific_name": taxon.get("name"),
                   "rank": taxon.get("rank"), "grade": GRADE_TEXT.get(grade or "", grade), "observed_on": format_date(obs.get("observed_on")),
                   "place": None if shown is not label else public_place(obs), "title": shown.get("title"), "caption": shown.get("caption"),
                   "keywords": shown.get("keywords")}
        if style is not None:
            subject = " ".join(x for x in (context["common_name"], context["scientific_name"], shown.get("title"),
                                           ", ".join((shown.get("keywords") or [])[:10])) if x)
            context["style"] = style(subject)
        try:
            context["attempts"] = []
            alt = describe_fn(out / name, context, model=model)
        except SkipObservation:
            shutil.rmtree(out)           # no half-written draft folder is left behind
            raise
        hooked = social_hooks.run("draft", {"obs": obs, "obs_id": obs_id, "image": out / name, "alt": alt, "label": label, "month": month,
                                            "rules": rules}, rules=rules, audit=audit)
        stop = social_hooks.first_non_pass(hooked)
        if stop is not None:
            shutil.rmtree(out)
            raise HookRejected(f"observation {obs_id}: {stop[1].reason or stop[0].name}")
        images.append({"file": name, "source": rel, "alt": alt, **info, "alt_attempts": context["attempts"],
                       **({"style_examples": context["style"]["examples"]} if style is not None else {})})
    draft = {"draft_id": draft_id, "month": month, "obs_id": obs_id, "url": INAT_OBS_URL.format(obs_id), "text": text_segments(obs, prints_url),
             "images": images, "vision_model": model, "observation": obs,
             "created_at": (now or dt.datetime.now(settings.LOCAL_TZ)).isoformat(timespec="seconds")}
    (out / "draft.json").write_text(json.dumps(draft, indent=1, ensure_ascii=False), encoding="utf-8")
    return draft


def _require_gates() -> None:
    """Load every social extension (idempotent) and refuse to go on when one failed: its gates would be missing."""
    try:
        social_hooks.require_loaded()
    except social_hooks.ExtensionsFailed as e:
        raise SocialError(str(e)) from e


def default_style() -> Callable[[str], dict]:
    """style(subject) from the saved profile (settings.SOCIAL_STYLE_PROFILE) and the pool's embed model."""
    from .social_style import load_profile, style_for

    profile = load_profile()
    _, embed = embedder()
    return lambda subject: {**style_for(profile, subject, embed=embed), "profile_built_at": profile["built_at"]}


def _obs_of(v: dict) -> list[int]:
    """The observation ids a ledger draft stands for: all of a themed draft's, the one of an ordinary draft."""
    return [int(o) for o in (v.get("obs_ids") or [v.get("obs_id")]) if o is not None]


def _taken(v: dict, month: str, today: dt.date, rules: Rules) -> bool:
    """Is the draft's observation off the table for new drafts: posted, rejected or awaiting an outcome ever; in a themed draft that is still
    to be posted (its day is not over); or in an ordinary draft that is still pending (with month_posting on: drafted this month, an earlier
    month's unposted draft having expired)."""
    if v["state"] in ("posted", "rejected", "posting"):
        return True
    if v.get("for_date"):                                              # a themed draft holds its observations only until its day is over
        return v["for_date"] >= today.isoformat() and v["state"] in ("drafted", "approved")
    if v["state"] in ("drafted", "approved", "failed"):
        return v["month"] == month or not rules.on("month_posting")
    return False


def _exhausted_by_withdrawals(ledger: Ledger, audit: Audit, *, wanted=None) -> set[int]:
    """Observations whose drafts the pre_send gates took back MAX_WITHDRAWALS_PER_OBS times: no candidates any more. Each one that is in `wanted`
    (the candidates of this run; None: all) is audited as draft_skipped "withdrawn N times: <last reason>"."""
    out = set()
    for o, (n, last) in sorted(ledger.withdrawal_counts().items()):
        if n >= MAX_WITHDRAWALS_PER_OBS:
            out.add(o)
            if wanted is None or o in wanted or str(o) in wanted:
                audit.log("draft_skipped", obs_id=o, reason=f"withdrawn {n} times: {last}")
    return out


def draft_month(index: dict, ledger: Ledger, *, today: dt.date, gallery_dir: Path, drafts_dir: Path, keep_pending: int | None = None,
                model: str | None = None, fetch: Callable[[int], dict] = _fetch, read_label: Callable[[Path], dict] = _read_label,
                describe_fn: Callable[..., str] = describe, pick_model: Callable[[], str] = vision_model,
                style_source: Callable[[], Callable[[str], dict]] | None = None, prints: dict[int, str] | None = None,
                rules: Rules | None = None, audit: Audit | None = None, local: dict[int, dict] | None = None,
                fetch_photo: Callable[[int], Path] | None = None, rejects: RejectBudget | None = None) -> dict:
    """Top up the unposted ordinary drafts (drafted or approved) to keep_pending from the observations that are not taken: posted, rejected
    or awaiting an outcome (posting) ever, or already in a pending draft or a themed draft still to be posted.

    The rules (social_rules) shape it, each optional: lower_quality_photos (`local`, inat_photo_candidates(): observations the gallery does not
    have are candidates too, with their best iNat photo fetched by `fetch_photo` when the draft is made), month_candidates (only observations photographed in
    `today`'s month of any year),
    month_posting (only this month's drafts count as pending, an earlier month's expired), prints_first (observations with a print
    first), year_rotation (years take turns), skip_casual. The owner's alt-text style is required: style_source() (default: the saved
    profile + the pool's embed model) is loaded once, before any drafting; a missing profile raises and drafts nothing. prints
    ({obs_id: url}, default load_prints()) adds a print link to the observations it lists. Every decision goes to the audit log.
    Draft-hook rejects are capped (`rejects`, default MAX_DRAFT_REJECTS per call): the run stops there, audits draft_reject_cap and reports
    "reject_cap". Returns {"month", "pending", "drafted", "skipped"}."""
    _require_gates()
    rules, audit = rules or load_rules(), audit or Audit()
    rejects = rejects or RejectBudget()
    keep_pending = settings.SOCIAL_KEEP_PENDING if keep_pending is None else keep_pending
    month = f"{today.year}-{today.month:02d}"
    known = ledger.drafts()
    this_month_only = rules.on("month_posting")
    pending = [d for d, v in known.items() if v["state"] in ("drafted", "approved") and not v.get("for_date")
               and (v["month"] == month or not this_month_only)]
    report = {"month": month, "pending": len(pending), "drafted": [], "skipped": {}}
    need = keep_pending - len(pending)
    if need <= 0:
        audit.log("draft_run", month=month, pending=len(pending), keep_pending=keep_pending, outcome="queue full", rules=rules.snapshot())
        return report
    seen = {int(o) for v in known.values() if _taken(v, month, today, rules) for o in _obs_of(v)}
    seen |= _exhausted_by_withdrawals(ledger, audit, wanted=set(index["files"].values()))
    prints = load_prints() if prints is None else prints
    if rules.on("month_candidates"):
        cands = month_observations(index, today.month)
    else:
        cands = {}
        for rel, obs in index["files"].items():
            cands.setdefault(int(obs), []).append(rel)
    cands = {o: sorted(r) for o, r in cands.items()}
    years: dict[int, str] = {}
    from_local = 0
    if rules.on("lower_quality_photos") and local:
        for o, v in local.items():
            if o in cands or (rules.on("month_candidates") and (v.get("observed_on") or "")[5:7] != f"{today.month:02d}"):
                continue                                                                       # the gallery's Stock-quality photos are the best there is
            cands[o] = [PHOTO_TOKEN + str(v["photo_id"])]
            years[o] = (v.get("observed_on") or "0000")[:4]
            from_local += 1
    model = model or pick_model()
    style = (style_source or default_style)()
    fetch_photo = fetch_photo or default_photo_fetch(rules)
    order = order_candidates(cands, seen, seed=today.isoformat(), rotate_years=rules.on("year_rotation"), years=years)
    if rules.on("prints_first"):
        order.sort(key=lambda o: o not in prints)                 # observations with a print first (stable: the year rotation is kept inside each group)
    social_hooks.run("order", {"candidates": order, "kind": "single", "taxonomy": None, "ledger": ledger, "today": today, "rules": rules},
                     rules=rules, audit=audit)                    # the hooks reorder the list in place
    audit.log("draft_run", month=month, pending=len(pending), keep_pending=keep_pending, need=need, candidates=len(cands), local_only=from_local,
              taken=len(seen), eligible=len(order), with_print=sum(1 for o in order if o in prints), rules=rules.snapshot())
    for obs_id in order:
        if len(report["drafted"]) >= need:
            break
        if rejects.exhausted:
            report["reject_cap"] = rejects.limit
            audit.log("draft_reject_cap", scope="single", limit=rejects.limit, drafted=len(report["drafted"]))
            break
        try:
            draft = make_draft(obs_id, resolve_photos(cands[obs_id], fetch_photo), month=month, gallery_dir=gallery_dir, drafts_dir=drafts_dir,
                               model=model, fetch=fetch, read_label=read_label, describe_fn=describe_fn, style=style,
                               prints_url=prints.get(obs_id), skip_casual=rules.on("skip_casual"), rules=rules, audit=audit)
        except SkipObservation as e:
            report["skipped"][obs_id] = str(e)
            audit.log("draft_skipped", obs_id=obs_id, reason=str(e))
            if isinstance(e, HookRejected):
                rejects.charge()
                if rejects.exhausted:
                    report["reject_cap"] = rejects.limit
                    audit.log("draft_reject_cap", scope="single", limit=rejects.limit, drafted=len(report["drafted"]))
                    break
            continue
        ledger.append_locked("drafted", draft["draft_id"], obs_id=obs_id, month=month, images=len(draft["images"]))
        audit.log("drafted", draft_id=draft["draft_id"], obs_id=obs_id, print=obs_id in prints, images=len(draft["images"]))
        report["drafted"].append(draft["draft_id"])
    return report


# --------------------------------------------------------------------------------------------------- themed posts

def group_text_segments(theme, items: list[dict]) -> list[list[str]]:
    """The text of a themed post as segments: "<emoji> <Theme name>", one line per photo (common name, scientific name in brackets), one
    Prints link when an observation has one, then the hashtags (the theme's first). items: [{"common", "name", "print"}]. While the text is
    over Bluesky's limits the richer parts go first: scientific names, then the print link, then hashtags from the end, then the names."""
    head = (f"{theme.emoji} " if theme.emoji else "") + theme.name
    tags: list[str] = []
    for t in (*theme.tags, *COMMON_TAGS):
        if t.lower() not in {x.lower() for x in tags}:
            tags.append(t)
    prints = next((i["print"] for i in items if i.get("print")), None)

    def name(i: dict, both: bool) -> str:
        common = (i.get("common") or "").strip()
        common = common[:1].upper() + common[1:]
        if common and both and i.get("name") and common.lower() != i["name"].lower():
            return f"{common} ({i['name']})"
        return common or i.get("name") or "an observation"

    def build(mode: str, with_print: bool, n_tags: int) -> list[list[str]]:
        names = {"both": [name(i, True) for i in items], "short": [name(i, False) for i in items],
                 "count": [f"{len(items)} finds"]}[mode]
        segs: list[list[str]] = [["text", head + "\n" + "\n".join(names) + "\n"]]
        if with_print and prints:
            segs += [["text", "Prints: "], ["link", prints, prints], ["text", "\n"]]
        use = tags[:n_tags]
        return segs + [s for t in use for s in (["tag", "#" + t, t], ["text", " "])][:-1]

    for mode, with_print, n_tags in (("both", True, 99), ("both", False, 99), ("short", True, 99), ("short", False, 99),
                                     ("short", False, 3), ("short", False, 1), ("count", False, 1)):
        segs = build(mode, with_print, n_tags)
        text = segments_text(segs)
        if len(text) <= TEXT_MAX_GRAPHEMES and len(text.encode("utf-8")) <= TEXT_MAX_BYTES:
            return segs
    raise SocialError(f"{theme.name}: the post text is over {TEXT_MAX_GRAPHEMES} characters even in its shortest form")


def make_group_draft(obs_ids: list[int], rels_of: dict[int, list[str]], theme, day: dt.date, *, gallery_dir: Path, drafts_dir: Path,
                     model: str, fetch: Callable[[int], dict] = _fetch, read_label: Callable[[Path], dict] = _read_label,
                     describe_fn: Callable[..., str] = describe, style: Callable[[str], dict] | None = None,
                     prints: dict[int, str] | None = None, now: dt.datetime | None = None,
                     skip_casual: bool = True, fetch_photo: Callable[[int], Path] | None = None, rules: Rules | None = None,
                     audit: Audit | None = None, rejects: RejectBudget | None = None) -> tuple[dict, dict[int, str]]:
    """Write drafts_dir/<day>-<theme key>/ (draft.json + 1.jpg..) for the theme on `day`: the first photo of each observation of `obs_ids`
    (in that order, each fetched live and described like an ordinary draft) until theme.max_photos are made; an observation that cannot be
    posted (casual, gone, no taxon) is skipped for the next one. Returns (draft, {skipped obs id: reason}); SkipObservation when fewer than
    theme.min_photos could be made. The draft belongs to the month of `day`."""
    prints = prints or {}
    month = f"{day.year}-{day.month:02d}"
    scratch = Path(drafts_dir) / "_scratch"
    picked: list[tuple[int, dict]] = []
    skipped: dict[int, str] = {}
    try:
        for o in obs_ids:
            if len(picked) >= theme.max_photos:
                break
            try:
                picked.append((o, make_draft(o, resolve_photos(rels_of[o][:1], fetch_photo), month=month, gallery_dir=gallery_dir, drafts_dir=scratch, model=model, fetch=fetch,
                                             read_label=read_label, describe_fn=describe_fn, style=style, prints_url=prints.get(o), now=now,
                                             skip_casual=skip_casual, rules=rules, audit=audit)))
            except SkipObservation as e:
                skipped[o] = str(e)
                if isinstance(e, HookRejected) and rejects is not None:
                    rejects.charge()
                    if rejects.exhausted:
                        break                                                 # the run's reject budget is spent: stop burning describes
        if len(picked) < theme.min_photos:
            raise SkipObservation(f"{theme.name} on {day}: {len(picked)} postable observation(s) of the {theme.min_photos} needed")
        draft_id = f"{day.isoformat()}-{theme.key}"
        out = Path(drafts_dir) / draft_id
        out.mkdir(parents=True, exist_ok=True)
        images, items = [], []
        for n, (o, d) in enumerate(picked, 1):
            shutil.copyfile(scratch / d["draft_id"] / d["images"][0]["file"], out / f"{n}.jpg")
            images.append({**d["images"][0], "file": f"{n}.jpg", "obs_id": o})
            tx = d["observation"].get("taxon") or {}
            items.append({"common": tx.get("preferred_common_name") or tx.get("common_name"), "name": tx.get("name"), "print": prints.get(o)})
        draft = {"draft_id": draft_id, "month": month, "for_date": day.isoformat(), "theme": theme.key, "obs_id": picked[0][0],
                 "obs_ids": [o for o, _ in picked], "url": picked[0][1]["url"], "text": group_text_segments(theme, items), "images": images,
                 "vision_model": model, "observations": {str(o): d["observation"] for o, d in picked},
                 "created_at": (now or dt.datetime.now(settings.LOCAL_TZ)).isoformat(timespec="seconds")}
        (out / "draft.json").write_text(json.dumps(draft, indent=1, ensure_ascii=False), encoding="utf-8")
        return draft, skipped
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def theme_candidates(index: dict, taxonomy: dict[int, dict], prints: dict[int, str], local: dict[int, dict] | None = None,
                     ) -> tuple[dict[int, dict], dict[int, list[str]]]:
    """({obs id: {"entry", "print"}}, {obs id: [photo paths]}) for the indexed observations the taxonomy knows: gallery relpaths, and with
    `local` (inat_photo_candidates(), the lower_quality_photos rule) an inat-photo:<id> token for each observation the gallery does not have."""
    rels_of: dict[int, list[str]] = {}
    for rel, obs in index["files"].items():
        rels_of.setdefault(int(obs), []).append(rel)
    rels_of = {o: sorted(r) for o, r in rels_of.items()}
    for o, v in (local or {}).items():
        rels_of.setdefault(o, [PHOTO_TOKEN + str(v["photo_id"])])
    return {o: {"entry": taxonomy[o], "print": prints.get(o)} for o in rels_of if o in taxonomy}, rels_of


def draft_themes(index: dict, ledger: Ledger, taxonomy: dict[int, dict], themes: list, *, today: dt.date, days_ahead: int,
                 gallery_dir: Path, drafts_dir: Path, model: str | None = None, fetch: Callable[[int], dict] = _fetch,
                 read_label: Callable[[Path], dict] = _read_label, describe_fn: Callable[..., str] = describe,
                 pick_model: Callable[[], str] = vision_model, style_source: Callable[[], Callable[[str], dict]] | None = None,
                 prints: dict[int, str] | None = None, rules: Rules | None = None, audit: Audit | None = None,
                 local: dict[int, dict] | None = None, fetch_photo: Callable[[int], Path] | None = None,
                 rejects: RejectBudget | None = None) -> dict:
    """Prepare the themed post of each of the next `days_ahead` days (today included) that has an enabled theme and no themed draft yet:
    up to 4 observations of the theme, one photo each (rules: theme_species_first = different species first, species-level IDs before
    coarse ones; prints_first = observations with a print first), none already taken. Nothing happens, and no vision model is loaded,
    when the `themes` rule is off or there is nothing to draft. Every decision goes to the audit log. Draft-hook rejects are capped like
    draft_month's (`rejects`, shared with it in one social-draft run): at the cap it stops, audits draft_reject_cap and reports "reject_cap".
    Returns {"drafted", "skipped", "short"}."""
    from . import themes as themes_mod

    _require_gates()
    rules, audit = rules or load_rules(), audit or Audit()
    rejects = rejects or RejectBudget()
    report: dict = {"drafted": [], "skipped": {}, "short": {}}
    if not rules.on("themes"):
        audit.log("theme_run", outcome="rule themes is off", rules=rules.snapshot())
        return report
    prints = load_prints() if prints is None else prints
    known = ledger.drafts()
    have = {(v["for_date"], v["theme"]) for v in known.values() if v.get("for_date") and v["state"] != "withdrawn"}
    taken = {o for v in known.values() if _taken(v, f"{today.year}-{today.month:02d}", today, rules) for o in _obs_of(v)}
    candidates, rels_of = theme_candidates(index, taxonomy, prints if rules.on("prints_first") else {}, local if rules.on("lower_quality_photos") else None)
    taken |= _exhausted_by_withdrawals(ledger, audit, wanted=set(candidates))
    audit.log("theme_run", today=today.isoformat(), days_ahead=days_ahead, themes=[t.key for t in themes if t.enabled], candidates=len(candidates),
              taken=len(taken), rules=rules.snapshot())
    model = style = None
    for k in range(days_ahead):
        if rejects.exhausted:
            report["reject_cap"] = rejects.limit
            audit.log("draft_reject_cap", scope="theme", limit=rejects.limit, drafted=len(report["drafted"]))
            break
        day = today + dt.timedelta(days=k)
        theme = themes_mod.theme_for(day, themes)
        if theme is None or (day.isoformat(), theme.key) in have:
            continue
        ids = list(themes_mod.pick_group(theme, candidates, day=day, taken=taken, species_first=rules.on("theme_species_first")))
        if ids:
            before = list(ids)
            social_hooks.run("order", {"candidates": ids, "kind": f"theme:{theme.key}", "taxonomy": taxonomy, "ledger": ledger, "today": today,
                                       "rules": rules}, rules=rules, audit=audit)
            ids[:] = [o for o in ids if o in before]              # reordered or thinned by the hooks, never extended
        if not ids:
            report["short"][day.isoformat()] = f"{theme.name}: fewer than {theme.min_photos} observations left"
            audit.log("theme_short", day=day.isoformat(), theme=theme.key, reason=report["short"][day.isoformat()])
            continue
        if model is None:
            model = pick_model()
            style = (style_source or default_style)()
        try:
            draft, skipped = make_group_draft(ids, rels_of, theme, day, gallery_dir=gallery_dir, drafts_dir=drafts_dir, model=model, fetch=fetch,
                                              read_label=read_label, describe_fn=describe_fn, style=style, prints=prints,
                                              skip_casual=rules.on("skip_casual"), fetch_photo=fetch_photo or default_photo_fetch(rules),
                                              rules=rules, audit=audit, rejects=rejects)
        except SkipObservation as e:
            report["short"][day.isoformat()] = str(e)
            audit.log("theme_short", day=day.isoformat(), theme=theme.key, reason=str(e))
            continue
        report["skipped"].update({str(o): r for o, r in skipped.items()})
        for o, reason in skipped.items():
            audit.log("draft_skipped", obs_id=o, theme=theme.key, reason=reason)
        ledger.append_locked("drafted", draft["draft_id"], obs_id=draft["obs_id"], obs_ids=draft["obs_ids"], month=draft["month"], kind="theme",
                             theme=theme.key, for_date=draft["for_date"], images=len(draft["images"]))
        audit.log("theme_drafted", draft_id=draft["draft_id"], theme=theme.key, for_date=draft["for_date"], obs_ids=draft["obs_ids"],
                  with_print=sum(1 for o in draft["obs_ids"] if o in prints))
        taken |= set(draft["obs_ids"])
        report["drafted"].append(draft["draft_id"])
    if rejects.exhausted and "reject_cap" not in report:
        report["reject_cap"] = rejects.limit
        audit.log("draft_reject_cap", scope="theme", limit=rejects.limit, drafted=len(report["drafted"]))
    return report


# --------------------------------------------------------------------------------------------------- owner decisions

REVIEW_CLAIM = "\nAlt text: local AI model, reviewed.\n"      # the line older drafts carry; a post nobody reviewed must not claim a review


def strip_review_claim(segments: list[list[str]]) -> list[list[str]]:
    """`segments` without the legacy "Alt text: local AI model, reviewed." line (the newline around it stays one newline)."""
    return [[s[0], s[1].replace(REVIEW_CLAIM, "\n"), *s[2:]] if s[0] == "text" else s for s in segments]


def _drop_review_claim(ddir: Path) -> bool:
    """Remove the legacy review-claim line from the draft's text on disk; True when there was one."""
    draft = load_draft(ddir)
    clean = strip_review_claim(draft["text"])
    if clean == draft["text"]:
        return False
    draft["text"] = clean
    (Path(ddir) / "draft.json").write_text(json.dumps(draft, indent=1, ensure_ascii=False), encoding="utf-8")
    return True


def approve(ledger: Ledger, drafts_dir: Path, draft_id: str, *, rules: Rules | None = None, audit: Audit | None = None,
            by: str = "owner") -> dict:
    """Approve a drafted (or failed) draft, binding the approval to the hash of its files as they are now. With the strip_review_claim rule
    on (the default) the legacy review-claim line is removed from the draft first, so the hash covers the text that will be sent. `by`
    ("owner" | "auto") goes to the audit log."""
    rules, audit = rules or load_rules(), audit or Audit()
    d = ledger.drafts().get(draft_id)
    if d is None:
        raise SocialError(f"no draft {draft_id!r} in {ledger.path}")
    if d["state"] not in ("drafted", "failed", "approved"):
        raise SocialError(f"draft {draft_id} is {d['state']}; only a drafted or failed draft can be approved")
    stripped = rules.on("strip_review_claim") and _drop_review_claim(Path(drafts_dir) / draft_id)
    _audit_or_stop(audit, "approved", then="the draft was not approved", draft_id=draft_id, by=by, review_claim_removed=bool(stripped))
    return ledger.append("approved", draft_id, obs_id=d["obs_id"], month=d["month"], hash=draft_hash(Path(drafts_dir) / draft_id))


def reject(ledger: Ledger, draft_id: str, reason: str = "") -> dict:
    d = ledger.drafts().get(draft_id)
    if d is None:
        raise SocialError(f"no draft {draft_id!r} in {ledger.path}")
    if d["state"] not in ("drafted", "approved", "failed"):
        raise SocialError(f"draft {draft_id} is {d['state']}; it can no longer be rejected")
    return ledger.append("rejected", draft_id, obs_id=d["obs_id"], month=d["month"], reason=reason)


def resolve(ledger: Ledger, draft_id: str, *, posted_uri: str | None, now: dt.datetime | None = None) -> dict:
    """Settle a post whose outcome was never recorded: the owner checked the profile. posted_uri = the post's at:// URI if it went out,
    None if it did not (the draft becomes failed and can be approved again)."""
    d = ledger.drafts().get(draft_id)
    if d is None or d["state"] != "posting":
        raise SocialError(f"draft {draft_id} is not awaiting an outcome (state {d and d['state']})")
    if posted_uri is not None and not re.fullmatch(r"at://\S+", posted_uri):
        raise SocialError(f"posted_uri must be the post's at:// URI, got {posted_uri!r}")
    if posted_uri:
        return ledger.append("posted", draft_id, obs_id=d["obs_id"], month=d["month"], uri=posted_uri, resolved=True, now=now)
    return ledger.append("failed", draft_id, obs_id=d["obs_id"], month=d["month"], error="not posted (resolved by the owner)", now=now)


# --------------------------------------------------------------------------------------------------- posting

class NothingSent(SocialError):
    """The send stopped before anything was posted (no session, refresh refused, an image upload failed): safe to retry."""


def poster_profile():
    """The bsky-oauth "poster" profile from settings (env overrides apply)."""
    from bsky_oauth import Profile

    return Profile(name="poster", client_id=settings.BSKY_CLIENT_ID, redirect_uri=settings.BSKY_REDIRECT_URI,
                   scope=settings.BSKY_OAUTH_SCOPE, state_dir=Path(settings.BSKY_OAUTH_DIR), client_name=settings.BSKY_CLIENT_NAME)


def poster_account():
    from bsky_oauth import Account

    return Account(poster_profile())


def bsky_post(payload: dict, *, account=None) -> tuple[str, str]:
    """Post through bsky-oauth: payload {text, langs, images: [{path, alt, width, height}]} -> (uri, cid).
    bsky_oauth.UnknownOutcome (the createRecord call failed) propagates; every other bsky-oauth error stopped before that call, so it
    becomes NothingSent."""
    from bsky_oauth import BskyError, Image, UnknownOutcome

    acct = account or poster_account()
    try:
        ref = acct.post(payload["text"], [Image(Path(i["path"]), i["alt"], i["width"], i["height"]) for i in payload["images"]],
                        langs=tuple(payload["langs"]))
    except UnknownOutcome:
        raise
    except BskyError as e:
        raise NothingSent(str(e)) from e
    except OSError as e:                                       # a local file that could not be read: that happens before any network call
        raise _local_read_failure(e, payload) from e
    return ref.uri, ref.cid


def _local_read_failure(e: OSError, payload: dict) -> OSError:
    """NothingSent when `e` names an image of the post or a file of the OAuth session; any other OSError (the network) is returned as it is,
    because the post may exist."""
    if e.filename:
        name = Path(str(e.filename))
        for i in payload["images"]:
            if name == Path(i["path"]):
                return NothingSent(f"cannot read the image {i['path']}: {e.strerror or e}")
        if name.is_relative_to(Path(settings.BSKY_OAUTH_DIR)):
            return NothingSent(f"cannot read the Bluesky session file {name}: {e.strerror or e}")
    return e


@contextlib.contextmanager
def ledger_lock(ledger_path: Path | str) -> Iterator[None]:
    """Hold the ledger's poster lock (an OS byte-range lock on <ledger>.lock, released by the OS if the process dies). A second holder
    raises SocialError at once instead of waiting: two runs that both read the ledger before either appends "posting" would post one draft twice."""
    path = Path(str(ledger_path) + ".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "a+b")
    try:
        try:
            if os.name == "nt":
                import msvcrt

                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            raise SocialError(f"another social-post run is posting ({path}); nothing was sent by this one") from e
        try:
            yield
        finally:
            if os.name == "nt":
                import msvcrt

                f.seek(0)
                with contextlib.suppress(OSError):
                    msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
    finally:
        f.close()


@contextlib.contextmanager
def ledger_lock_wait(ledger_path: Path | str, *, wait: float | None = None) -> Iterator[None]:
    """ledger_lock, but a busy lock is retried every LEDGER_LOCK_POLL_S for `wait` seconds (default LEDGER_LOCK_WAIT_S) before SocialError:
    prep and drafting append to the ledger while a post run may hold the lock, and must not interleave with it."""
    deadline = time.monotonic() + (LEDGER_LOCK_WAIT_S if wait is None else wait)
    with contextlib.ExitStack() as stack:
        while True:
            try:
                stack.enter_context(ledger_lock(ledger_path))
                break
            except SocialError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(LEDGER_LOCK_POLL_S)
        yield


def post_next(ledger: Ledger, drafts_dir: Path, *, now: dt.datetime, live: bool = False, posts_per_day: int | None = None,
              send: Callable[[dict], tuple[str, str]] = bsky_post, manual: bool = False, draft_id: str | None = None,
              auto_approve: bool | None = None, rules: Rules | None = None, audit: Audit | None = None) -> dict:
    """Post one draft. Dry-run (the default) returns what would be sent and records nothing in the ledger.

    Scheduled runs (the default): themed draft of today first (rule theme_day_only), else the oldest approved draft; when none is
    approved and `auto_approve` (default: the auto_approve rule) is on, the oldest drafted draft is approved first and posted. With the
    month_posting rule on, only drafts of now's month are eligible. They do nothing once daily_cap (the rule's number, or
    `posts_per_day`) scheduled posts went out on now's local day.
    Manual (`manual`, or `draft_id`, which posts that very draft): not counted in the daily cap and not stopped by it while the
    manual_exempt_from_cap rule is on; the posted record carries manual=true. Either way a post whose outcome is unknown blocks everything
    until social-resolve.

    Every decision is written to the audit log (rules in force, the reasons, the choice) BEFORE anything is sent: a decision that cannot be
    recorded is not acted on. Live: "posting" is recorded in the ledger before the send. A send that stopped before anything went out
    (NothingSent) records an "error" note and the draft stays approved for the next run; any other failure leaves it "posting" (outcome
    unknown -> social-resolve)."""
    _require_gates()
    rules, audit = rules or load_rules(), audit or Audit()
    if ledger.audit is None:
        ledger.audit = audit                                          # a corrupt ledger line is audited in this run's log
    kw = dict(now=now, posts_per_day=posts_per_day, send=send, manual=manual or draft_id is not None, draft_id=draft_id,
              auto_approve=rules.on("auto_approve") if auto_approve is None else auto_approve, rules=rules, audit=audit)
    if not live:
        return _post_next(ledger, drafts_dir, live=False, **kw)
    after: list[Callable[[], None]] = []
    with ledger_lock(ledger.path):                                    # the checks, the "posting" record and the send are one step
        result = _post_next(ledger, drafts_dir, live=True, after=after, **kw)
    for run_after in after:                                           # the post_send hooks (the verify waits up to a minute for the AppView) run with the lock released:
        run_after()                                                   # the post is recorded, and the owner's commands and the draft run must not wait for the read-back
    return result


MAX_PRE_SEND_WITHDRAWALS = 3         # drafts the pre_send hooks may take back in one run before it gives up


def _withdrawn_report(taken_back: list[str], reasons: dict[str, str]) -> dict:
    """{"withdrawn": [ids], "withdrawn_count", "withdrawn_reasons": {id: why}} of the drafts the pre_send hooks took back this run, else {}:
    a withdrawal spends a posting slot, so every result and audit record says how many and why."""
    if not taken_back:
        return {}
    return {"withdrawn": list(taken_back), "withdrawn_count": len(taken_back), "withdrawn_reasons": {d: reasons[d] for d in taken_back}}


def stalled_drafts(ledger: Ledger, now: dt.datetime, *, live: bool, audit: Audit) -> set[str]:
    """The drafts to skip today: STALL_AFTER or more consecutive NothingSent failures, the last one on now's local day (a deterministic failure
    must not block the queue forever). Live: each is audited as draft_stalled (an alert names it), and one stalled for STALL_FAIL_DAYS days is
    marked failed (so choose() stops considering it until the owner approves or rejects it)."""
    out: set[str] = set()
    states = ledger.drafts()
    for did, s in sorted(ledger.nothing_sent_streaks().items()):
        if s["count"] < STALL_AFTER or states.get(did, {}).get("state") not in ("approved", "drafted"):
            continue
        first, last = dt.datetime.fromisoformat(s["first"]), dt.datetime.fromisoformat(s["last"])
        if now - first >= dt.timedelta(days=STALL_FAIL_DAYS):
            if live:
                reason = f"{s['count']} sends in a row sent nothing since {s['first']}; marked failed after {STALL_FAIL_DAYS} days"
                ledger.append("failed", did, obs_id=states[did]["obs_id"], month=states[did]["month"], error=reason, now=now)
                _audit_or_stop(audit, "draft_stalled", draft_id=did, count=s["count"], action="failed", reason=reason)
            continue
        if last.astimezone(now.tzinfo).date() == now.date():
            out.add(did)
            if live:
                _audit_or_stop(audit, "draft_stalled", draft_id=did, count=s["count"], action="skipped",
                               reason=f"{s['count']} sends in a row sent nothing (last {s['last']}); skipped for the rest of the day")
    return out


def _audit_or_stop(audit: Audit, event: str, *, then: str = "nothing was sent", **fields) -> None:
    """Write the audit record or raise SocialError saying that `then` (what did not happen) is the consequence."""
    try:
        audit.log(event, **fields)
    except OSError as e:
        raise SocialError(f"the audit log {audit.path} cannot be written ({e}); {then}") from e


def _post_next(ledger: Ledger, drafts_dir: Path, *, now: dt.datetime, live: bool, posts_per_day: int | None,
               send: Callable[[dict], tuple[str, str]], manual: bool, draft_id: str | None, auto_approve: bool,
               rules: Rules, audit: Audit, after: list[Callable[[], None]] | None = None) -> dict:
    cap = posts_per_day if posts_per_day is not None else rules.value("daily_cap")
    month = f"{now.year}-{now.month:02d}"
    stalled = stalled_drafts(ledger, now, live=live, audit=audit)
    drafts = ledger.drafts()
    why: list[str] = []
    stuck = [d for d, v in drafts.items() if v["state"] == "posting"]
    if stuck:
        _audit_or_stop(audit, "post_blocked", live=live, reason="outcome unknown", drafts=stuck, rules=rules.snapshot())
        raise SocialError(f"outcome unknown for {', '.join(stuck)}: check the profile, then `bio social-resolve` each one")
    exempt = manual and rules.on("manual_exempt_from_cap")
    if rules.on("daily_cap") and not exempt:
        today = now.date().isoformat()
        posted_today = [r for r in ledger.records() if r["event"] == "posted" and r["at"][:10] == today
                        and not (r.get("manual") and rules.on("manual_exempt_from_cap"))]
        if len(posted_today) >= cap:
            _audit_or_stop(audit, "post_skipped", live=live, status="done for today", cap=cap, posted_today=[r["draft_id"] for r in posted_today],
                           manual=manual, rules=rules.snapshot())
            return {"status": "done for today", "posted_today": [r["draft_id"] for r in posted_today]}
        why.append(f"daily_cap: {len(posted_today)} of {cap} scheduled posts used today")
    else:
        why.append("daily_cap: not applied (" + ("manual post, manual_exempt_from_cap on" if exempt else "rule off") + ")")
    month_ok = (lambda v: v["month"] == month) if rules.on("month_posting") else (lambda v: True)
    why.append("month_posting: " + ("only drafts of " + month if rules.on("month_posting") else "off, drafts of any month are eligible"))
    explicit = draft_id is not None
    owner_pick = draft_id

    def choose(skip: set[str]):
        """(draft_id, needs_approval, why) of the draft to post, skipping those the pre_send hooks took back this run; or (None, result, None)."""
        why_here = list(why)
        if owner_pick is not None:
            d = drafts.get(owner_pick)
            if d is None:
                raise SocialError(f"no draft {owner_pick!r} in {ledger.path}")
            if d["state"] not in ("drafted", "failed", "approved"):
                raise SocialError(f"draft {owner_pick} is {d['state']}; only a drafted, failed or approved draft can be posted")
            why_here.append(f"chosen by the owner: {owner_pick}")
            return owner_pick, d["state"] != "approved", why_here
        today_iso = now.date().isoformat()
        day_only = rules.on("theme_day_only")
        # theme_day_only: a themed draft is for its own day only (today's goes first, a past day's never posts, a coming day's waits);
        # off: themed drafts are ordinary drafts
        themed = [d for d, v in drafts.items() if d not in skip and day_only and v.get("for_date") == today_iso
                  and (v["state"] == "approved" or (v["state"] == "drafted" and auto_approve))]
        ordinary = lambda v: not (day_only and v.get("for_date"))        # noqa: E731
        ready = [(v["approved_seq"], d) for d, v in drafts.items() if d not in skip and v["state"] == "approved" and month_ok(v) and ordinary(v)]
        if themed:
            why_here.append(f"theme_day_only: today's themed draft {themed[0]} goes first")
            return themed[0], drafts[themed[0]]["state"] != "approved", why_here
        if ready:
            why_here.append(f"oldest approved draft: {min(ready)[1]}")
            return min(ready)[1], False, why_here
        fresh = [d for d, v in drafts.items() if d not in skip and v["state"] == "drafted" and month_ok(v) and ordinary(v)]
        if not (auto_approve and fresh):
            _audit_or_stop(audit, "post_skipped", live=live, status="nothing approved", auto_approve=auto_approve, drafted=len(fresh),
                           rules=rules.snapshot())
            return None, {"status": "nothing approved", "month": month}, None
        why_here.append(f"auto_approve: nothing approved, the oldest drafted draft {fresh[0]} is approved by this run")
        return fresh[0], True, why_here

    taken_back: list[str] = []
    reasons: dict[str, str] = {}
    for _attempt in range(MAX_PRE_SEND_WITHDRAWALS):
        draft_id, needs_approval, why_now = choose(set(taken_back) | (stalled if not explicit else set()))
        if draft_id is None:
            return {**needs_approval, **_withdrawn_report(taken_back, reasons)}
        ddir = Path(drafts_dir) / draft_id
        pre_ctx = {"draft_id": draft_id, "draft": load_draft(ddir), "draft_dir": ddir, "ledger": ledger, "now": now, "live": live,
                   "manual": manual, "fetch": _fetch, "rules": rules, "audit": audit, "on_send": []}
        hooked = social_hooks.run("pre_send", pre_ctx, rules=rules, audit=audit)
        stop = social_hooks.first_non_pass(hooked)
        if stop is None:
            why = why_now
            break
        hook, verdict = stop
        reason = verdict.reason or f"hook {hook.name}"
        if verdict.action == "block":
            _audit_or_stop(audit, "post_skipped", live=live, status="blocked", draft_id=draft_id, hook=hook.name, reason=reason, rules=rules.snapshot(),
                           **_withdrawn_report(taken_back, reasons))
            return {"status": "blocked", "reason": reason, "draft_id": draft_id, **_withdrawn_report(taken_back, reasons)}
        reasons[draft_id] = reason
        # a dry-run only says what a live run would do: "would_withdraw", never "withdrawn" (and nothing in the ledger)
        _audit_or_stop(audit, "withdrawn" if live else "would_withdraw", live=live, draft_id=draft_id, obs_ids=_obs_of(drafts[draft_id]),
                       hook=hook.name, reason=reason)
        if live:
            ledger.append("withdrawn", draft_id, obs_id=drafts[draft_id]["obs_id"], month=drafts[draft_id]["month"], reason=reason, hook=hook.name, now=now)
        taken_back.append(draft_id)
        if explicit:
            _audit_or_stop(audit, "post_skipped", live=live, status="withdrawn", draft_id=draft_id, reason=reason, rules=rules.snapshot(),
                           **_withdrawn_report(taken_back, reasons))
            return {"status": "withdrawn", "draft_id": draft_id, "reason": reason, **_withdrawn_report(taken_back, reasons)}
    else:
        _audit_or_stop(audit, "post_skipped", live=live, status="withdrawn", drafts=taken_back, rules=rules.snapshot(),
                       **_withdrawn_report(taken_back, reasons))
        return {"status": "withdrawn", **_withdrawn_report(taken_back, reasons),
                "reason": f"{len(taken_back)} drafts were withdrawn before sending (the cap per run is {MAX_PRE_SEND_WITHDRAWALS}); nothing was posted"}
    ddir = Path(drafts_dir) / draft_id
    if needs_approval and live:
        approve(ledger, drafts_dir, draft_id, rules=rules, audit=audit, by="owner" if explicit else "auto")
        drafts = ledger.drafts()
    elif not needs_approval and draft_hash(ddir) != drafts[draft_id]["approved_hash"]:
        raise SocialError(f"draft {draft_id} changed after it was approved: review it and approve it again")
    dmonth = drafts[draft_id]["month"]
    draft = load_draft(ddir)
    text = strip_review_claim(draft["text"]) if needs_approval and rules.on("strip_review_claim") else draft["text"]   # a dry-run shows what approval leaves
    request = {"text": segments_text(text), "langs": ["en"],
               "images": [{"path": str(ddir / i["file"]), "alt": i["alt"], "width": i["width"], "height": i["height"]} for i in draft["images"]]}
    payload = {"draft_id": draft_id, "text": request["text"], "alts": [i["alt"] for i in draft["images"]],
               "image_bytes": [(ddir / i["file"]).stat().st_size for i in draft["images"]], "manual": manual,
               "approved_by_this_run": needs_approval}
    _audit_or_stop(audit, "post_decision", live=live, draft_id=draft_id, manual=manual, themed=bool(drafts[draft_id].get("for_date")),
                   obs_ids=_obs_of(drafts[draft_id]), images=len(request["images"]), approved_by_this_run=needs_approval, reasons=why,
                   rules=rules.snapshot(), **_withdrawn_report(taken_back, reasons))
    if not live:
        return {"status": "dry-run", **payload, **_withdrawn_report(taken_back, reasons)}
    ledger.append("posting", draft_id, obs_id=draft["obs_id"], month=dmonth, now=now)
    for started in pre_ctx["on_send"]:                                # effects of pre_send hooks that must wait until a send really starts
        started()
    try:
        uri, cid = send(request)
    except NothingSent as e:
        ledger.append("approved", draft_id, obs_id=draft["obs_id"], month=dmonth, hash=drafts[draft_id]["approved_hash"], restored=True, now=now)
        ledger.append("error", draft_id, obs_id=draft["obs_id"], month=dmonth, error=str(e)[:500], now=now)
        with contextlib.suppress(OSError):
            audit.log("post_failed", draft_id=draft_id, outcome="nothing sent", error=str(e)[:300])
        raise SocialError(f"nothing was posted ({e}); {draft_id} stays approved for the next run") from e
    except Exception as e:  # noqa: BLE001 -- the post may or may not exist now: the draft stays "posting" and blocks until resolved
        with contextlib.suppress(OSError):
            audit.log("post_failed", draft_id=draft_id, outcome="unknown", error=f"{type(e).__name__}: {e}"[:300])
        raise SocialError(f"posting {draft_id} failed with an unknown outcome ({type(e).__name__}: {e}); check the profile, then "
                          f"`bio social-resolve {draft_id} --posted-uri <at:// uri>` or `--not-posted`") from e
    extra = {"manual": True} if manual else {}
    ledger.append("posted", draft_id, obs_id=draft["obs_id"], month=dmonth, uri=uri, cid=cid, now=now, **extra)
    result = {"status": "posted", "uri": uri, **payload, **_withdrawn_report(taken_back, reasons)}
    try:
        audit.log("posted", draft_id=draft_id, uri=uri, manual=manual)
    except OSError:
        result["audit"] = "the posted record could not be written to the audit log"    # the post exists and the ledger has it: say so, do not raise

    def post_send() -> None:
        social_hooks.run("post_send", {"draft_id": draft_id, "draft": draft, "uri": uri, "cid": cid, "request": request, "ledger": ledger, "now": now,
                                       "manual": manual, "rules": rules}, rules=rules, audit=audit)

    if after is None:
        post_send()
    else:
        after.append(post_send)                                       # post_next runs it once the ledger lock is released
    return result
