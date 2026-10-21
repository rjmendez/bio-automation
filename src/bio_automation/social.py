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
import csv
import datetime as dt
import hashlib
import io
import json
import os
import random
import re
import subprocess
import tempfile
import tomllib
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from PIL import Image, ImageOps

from . import settings

MAX_IMAGES = 4                     # app.bsky.embed.images: images maxLength 4
TEXT_MAX_GRAPHEMES = 300           # app.bsky.feed.post text: maxGraphemes 300, maxLength 3000 (UTF-8 bytes)
TEXT_MAX_BYTES = 3000
ALT_MAX_CHARS = 2000               # the Bluesky app's alt-text limit (the lexicon sets none)
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
STATES = ("drafted", "approved", "rejected", "posting", "posted", "failed")


class SocialError(Exception):
    """A social-post step cannot go on; the message says what to do."""


class VisionError(SocialError):
    """The local vision model is missing, failed, or did not run fully on the GPU."""


class SkipObservation(SocialError):
    """This observation cannot be posted (casual, deleted, no taxon); the next candidate is tried."""


# --------------------------------------------------------------------------------------------------- gallery file -> observation index

def _exiftool() -> str:
    from .inat_stock import _exiftool_binary

    return _exiftool_binary()


def read_titles(paths: list[Path]) -> dict[Path, str]:
    """{path: embedded XMP-dc title ('' when none)} for many files in one exiftool run (argfile, UTF-8)."""
    if not paths:
        return {}
    with tempfile.NamedTemporaryFile("w", suffix=".args", delete=False, encoding="utf-8", newline="\n") as af:
        af.write("\n".join(str(p) for p in paths) + "\n")
    try:
        r = subprocess.run([_exiftool(), "-j", "-charset", "utf8", "-charset", "filename=utf8", "-XMP-dc:Title", "-@", af.name],
                           capture_output=True)
    finally:
        os.unlink(af.name)
    if not r.stdout.strip():
        raise SocialError(f"exiftool read no titles: {r.stderr.decode('utf-8', 'replace').strip()[:200]}")
    by_name = {}
    for d in json.loads(r.stdout.decode("utf-8")):
        t = d.get("Title")
        by_name[os.path.normcase(os.path.normpath(d["SourceFile"]))] = "" if t is None else str(t)
    return {p: by_name.get(os.path.normcase(os.path.normpath(str(p))), "") for p in paths}


def _render_name(row: dict) -> str:
    from .stock_select import render_name

    return render_name(row)


def build_obs_index(gallery_dir: Path, audit_csvs: list[Path], candidate_csvs: list[Path], *,
                    titles: Callable[[list[Path]], dict[Path, str]] = read_titles) -> dict:
    """{"files": {"<year>/<year-month>/<name>": obs_id}, "unresolved": {relpath: reason}} for every .jpg in the gallery.

    Evidence: an audit csv row (file, obs_id: the owner's Lightroom exports, one row per file) and a candidate csv row (a published Stock
    candidate: render_name(row) is its gallery name, obs_id, title). A candidate row counts only when its title equals the title embedded in
    the gallery file -- older matching passes left rows that point the same name at another observation. Exactly one observation left =
    resolved; none or several = unresolved with the reason."""
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
    embedded = titles([p for p in files if any(t is not None for _, t in evidence.get(p, []))])
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


def order_candidates(cands: dict[int, list[str]], exclude: set[int], seed: str) -> list[int]:
    """Observations not in `exclude`, years taking turns (newest year first), shuffled within a year by `seed` -- the same seed gives the
    same order, so a re-run of one day's drafting proposes the same observations."""
    rng = random.Random(seed)
    by_year: dict[str, list[int]] = {}
    for obs, rels in cands.items():
        if obs not in exclude:
            by_year.setdefault(min(r.split("/")[0] for r in rels), []).append(obs)
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

class Ledger:
    """Append-only JSON lines: {"at", "event", "draft_id", ...}. A draft's state is its last event."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path or settings.SOCIAL_LEDGER)

    def records(self) -> list[dict]:
        if not self.path.is_file():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def append(self, event: str, draft_id: str, *, now: dt.datetime | None = None, **fields) -> dict:
        if event not in STATES:
            raise SocialError(f"unknown ledger event {event!r}")
        rec = {"at": (now or dt.datetime.now(settings.LOCAL_TZ)).isoformat(timespec="seconds"), "event": event, "draft_id": draft_id, **fields}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
        return rec

    def drafts(self) -> dict[str, dict]:
        """{draft_id: {"state", "obs_id", "month", "drafted_at", "last": <last record>, "seq": its ledger position, "approved_hash"}}."""
        out: dict[str, dict] = {}
        for seq, r in enumerate(self.records()):
            d = out.setdefault(r["draft_id"], {"obs_id": r.get("obs_id"), "month": r.get("month"), "drafted_at": r["at"], "approved_hash": None})
            d["state"], d["last"], d["seq"] = r["event"], r, seq       # seq: ledger position (timestamps tie within a second)
            if r["event"] == "approved":
                d["approved_hash"] = r.get("hash")
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


def vision_model(*, api: Callable = _ollama, pool_file: Path | None = None) -> str:
    """The vision model to use: env BIO_VISION_MODEL, else the best-ranked (lowest rank) model of the pool file whose roles include
    "vision" and that Ollama has installed. Never a literal tag; nothing found raises."""
    named = os.environ.get(settings.VISION_MODEL_ENV)
    if named:
        return named
    pool_file = Path(pool_file or settings.MODEL_POOL_FILE)
    if not pool_file.is_file():
        raise VisionError(f"no vision model: set {settings.VISION_MODEL_ENV} or give the pool file ({pool_file}) a 'vision' role")
    pool = tomllib.loads(pool_file.read_text(encoding="utf-8")).get("models", {}).get("pool", [])
    installed = {m["name"] for m in api("/api/tags").get("models", [])}
    ranked = sorted((e.get("rank", 99), e["name"]) for e in pool if "vision" in e.get("roles", []) and e.get("name") in installed)
    if not ranked:
        raise VisionError(f"no installed model has the 'vision' role in {pool_file}; set {settings.VISION_MODEL_ENV}")
    return ranked[0][1]


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
              "- One to three sentences, under 350 characters."]
    return "\n".join(lines)


_OPENER = re.compile(r"^(?:the|this)\s+(?:image|photo|photograph|picture)\s+(?:shows|captures|depicts|features|is of)\s+", re.IGNORECASE)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def clean_alt(text: str, context: dict) -> str:
    """The model's answer, tidied the same way every time: whitespace collapsed, a leading "The image shows" removed, and sentences that
    restate the date or the place (the post text carries both) dropped. Nothing else is rewritten; an answer that would end up empty is
    kept whole."""
    text = " ".join(text.split())
    sentences = [_OPENER.sub("", x) for x in _SENTENCE.split(text)] if text else []
    sentences = [x[:1].upper() + x[1:] for x in sentences if x]
    facts = [f for f in (context.get("observed_on"), context.get("place")) if f]
    kept = [x for x in sentences if not any(f in x for f in facts)]
    return " ".join(kept or sentences)


def describe(image: Path, context: dict, *, model: str, api: Callable = _ollama) -> str:
    """Alt text for `image` from the local vision model. Refuses an empty answer, and refuses when Ollama holds the model partly in system
    memory (CPU inference is not allowed)."""
    with Image.open(image) as im:
        im = im.convert("RGB")
        im.thumbnail((settings.VISION_IMAGE_EDGE, settings.VISION_IMAGE_EDGE), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=90)
    r = api("/api/chat", {"model": model, "stream": False, "think": False, "keep_alive": "10m", "options": {"temperature": 0},
                          "messages": [{"role": "user", "content": alt_text_prompt(context),
                                        "images": [base64.b64encode(buf.getvalue()).decode()]}]})
    loaded = {m["name"]: m for m in api("/api/ps").get("models", [])}
    m = loaded.get(model)
    if m is None or m.get("size_vram", 0) < m.get("size", 0):
        raise VisionError(f"{model} is not fully on the GPU (loaded: {m and (m.get('size_vram'), m.get('size'))}); refusing CPU inference")
    text = clean_alt(r.get("message", {}).get("content") or "", context)
    if not text:
        raise VisionError(f"{model} returned no alt text for {image}")
    return text[:ALT_MAX_CHARS]


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


def text_segments(obs: dict) -> list[list[str]]:
    """The post as [kind, text(, target)] segments: ["text", s], ["link", s, url], ["tag", "#t", "t"]. Tags are dropped from the end
    while the text is over Bluesky's limits; a post still over them without any tag raises."""
    taxon = obs.get("taxon") or {}
    url = INAT_OBS_URL.format(obs["observation_id"])
    head = [display_name(taxon), GRADE_TEXT.get(obs.get("quality_grade") or "", ""),
            " · ".join(x for x in (format_date(obs.get("observed_on")), public_place(obs)) if x)]
    segs = [["text", "\n".join(x for x in head if x) + "\n"], ["link", url, url], ["text", "\nAlt text: local AI model, reviewed.\n"]]
    tags = hashtags(taxon.get("iconic_taxon_name"))
    while True:
        out = segs + [s for t in tags for s in (["tag", "#" + t, t], ["text", " "])][:-1]
        text = segments_text(out)
        if len(text) <= TEXT_MAX_GRAPHEMES and len(text.encode("utf-8")) <= TEXT_MAX_BYTES:   # code points >= graphemes: a safe count
            return out
        if not tags:
            raise SocialError(f"observation {obs['observation_id']}: post text is {len(text)} characters even without hashtags")
        tags.pop()


def segments_text(segs: list[list[str]]) -> str:
    return "".join(s[1] for s in segs)


def text_builder(segs: list[list[str]]):
    from atproto import client_utils

    tb = client_utils.TextBuilder()
    for s in segs:
        if s[0] == "text":
            tb.text(s[1])
        elif s[0] == "link":
            tb.link(s[1], s[2])
        elif s[0] == "tag":
            tb.tag(s[1], s[2])
        else:
            raise SocialError(f"unknown text segment {s[0]!r}")
    return tb


# --------------------------------------------------------------------------------------------------- drafting

def _fetch(obs_id: int) -> dict:
    from .inat import fetch_inat_observation

    return fetch_inat_observation(obs_id)


def _read_label(path: Path) -> dict:
    from .inat_stock import read_label

    return read_label(path)


def make_draft(obs_id: int, rels: list[str], *, month: str, gallery_dir: Path, drafts_dir: Path, model: str,
               fetch: Callable[[int], dict] = _fetch, read_label: Callable[[Path], dict] = _read_label,
               describe_fn: Callable[..., str] = describe, now: dt.datetime | None = None) -> dict:
    """Write drafts_dir/<month>-obs<obs_id>/ (draft.json + 1.jpg..4.jpg) and return the draft. The observation is fetched live: casual,
    deleted and taxon-less observations raise SkipObservation."""
    from .inat import InatNotFoundError

    try:
        obs = fetch(obs_id)
    except InatNotFoundError as e:
        raise SkipObservation(f"observation {obs_id} no longer exists on iNat") from e
    grade, taxon = obs.get("quality_grade"), obs.get("taxon") or {}
    if grade == "casual":
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
        context = {"common_name": taxon.get("preferred_common_name") or taxon.get("common_name"), "scientific_name": taxon.get("name"),
                   "rank": taxon.get("rank"), "grade": GRADE_TEXT.get(grade or "", grade), "observed_on": format_date(obs.get("observed_on")),
                   "place": public_place(obs), "title": label.get("title"), "caption": label.get("caption"), "keywords": label.get("keywords")}
        alt = describe_fn(out / name, context, model=model)
        images.append({"file": name, "source": rel, "alt": alt, **info})
    draft = {"draft_id": draft_id, "month": month, "obs_id": obs_id, "url": INAT_OBS_URL.format(obs_id), "text": text_segments(obs),
             "images": images, "vision_model": model, "observation": obs,
             "created_at": (now or dt.datetime.now(settings.LOCAL_TZ)).isoformat(timespec="seconds")}
    (out / "draft.json").write_text(json.dumps(draft, indent=1, ensure_ascii=False), encoding="utf-8")
    return draft


def draft_month(index: dict, ledger: Ledger, *, today: dt.date, gallery_dir: Path, drafts_dir: Path, keep_pending: int | None = None,
                model: str | None = None, fetch: Callable[[int], dict] = _fetch, read_label: Callable[[Path], dict] = _read_label,
                describe_fn: Callable[..., str] = describe, pick_model: Callable[[], str] = vision_model) -> dict:
    """Top up `today`'s month's unposted drafts (drafted or approved) to keep_pending, from the observations with gallery files in that
    month of any year that the ledger has never seen (posted, rejected or pending alike). Returns {"month", "pending", "drafted", "skipped"}."""
    keep_pending = settings.SOCIAL_KEEP_PENDING if keep_pending is None else keep_pending
    month = f"{today.year}-{today.month:02d}"
    known = ledger.drafts()
    pending = [d for d, v in known.items() if v["month"] == month and v["state"] in ("drafted", "approved")]
    report = {"month": month, "pending": len(pending), "drafted": [], "skipped": {}}
    need = keep_pending - len(pending)
    if need <= 0:
        return report
    seen = {int(v["obs_id"]) for v in known.values() if v.get("obs_id") is not None}
    cands = month_observations(index, today.month)
    model = model or pick_model()
    for obs_id in order_candidates(cands, seen, seed=today.isoformat()):
        if len(report["drafted"]) >= need:
            break
        try:
            draft = make_draft(obs_id, cands[obs_id], month=month, gallery_dir=gallery_dir, drafts_dir=drafts_dir, model=model,
                               fetch=fetch, read_label=read_label, describe_fn=describe_fn)
        except SkipObservation as e:
            report["skipped"][obs_id] = str(e)
            continue
        ledger.append("drafted", draft["draft_id"], obs_id=obs_id, month=month, images=len(draft["images"]))
        report["drafted"].append(draft["draft_id"])
    return report


# --------------------------------------------------------------------------------------------------- owner decisions

def approve(ledger: Ledger, drafts_dir: Path, draft_id: str) -> dict:
    """Approve a drafted (or failed) draft, binding the approval to the hash of its files as they are now."""
    d = ledger.drafts().get(draft_id)
    if d is None:
        raise SocialError(f"no draft {draft_id!r} in {ledger.path}")
    if d["state"] not in ("drafted", "failed", "approved"):
        raise SocialError(f"draft {draft_id} is {d['state']}; only a drafted or failed draft can be approved")
    return ledger.append("approved", draft_id, obs_id=d["obs_id"], month=d["month"], hash=draft_hash(Path(drafts_dir) / draft_id))


def reject(ledger: Ledger, draft_id: str, reason: str = "") -> dict:
    d = ledger.drafts().get(draft_id)
    if d is None:
        raise SocialError(f"no draft {draft_id!r} in {ledger.path}")
    if d["state"] not in ("drafted", "approved", "failed"):
        raise SocialError(f"draft {draft_id} is {d['state']}; it can no longer be rejected")
    return ledger.append("rejected", draft_id, obs_id=d["obs_id"], month=d["month"], reason=reason)


def resolve(ledger: Ledger, draft_id: str, *, posted_uri: str | None) -> dict:
    """Settle a post whose outcome was never recorded: the owner checked the profile. posted_uri = the post's at:// URI if it went out,
    None if it did not (the draft becomes failed and can be approved again)."""
    d = ledger.drafts().get(draft_id)
    if d is None or d["state"] != "posting":
        raise SocialError(f"draft {draft_id} is not awaiting an outcome (state {d and d['state']})")
    if posted_uri:
        return ledger.append("posted", draft_id, obs_id=d["obs_id"], month=d["month"], uri=posted_uri, resolved=True)
    return ledger.append("failed", draft_id, obs_id=d["obs_id"], month=d["month"], error="not posted (resolved by the owner)")


# --------------------------------------------------------------------------------------------------- posting

def bluesky_client():
    """A logged-in atproto Client from env BSKY_HANDLE + BSKY_APP_PASSWORD."""
    handle, password = os.environ.get(settings.BSKY_HANDLE_ENV), os.environ.get(settings.BSKY_APP_PASSWORD_ENV)
    if not handle or not password:
        raise SocialError(f"set {settings.BSKY_HANDLE_ENV} and {settings.BSKY_APP_PASSWORD_ENV} (a Bluesky app password) to post")
    from atproto import Client

    client = Client()
    client.login(handle, password)
    return client


def post_next(ledger: Ledger, drafts_dir: Path, *, now: dt.datetime, live: bool = False, posts_per_day: int | None = None,
              client_factory: Callable[[], object] = bluesky_client) -> dict:
    """Post the oldest approved draft of now's month. Dry-run (the default) returns what would be sent and records nothing.
    Refuses while any post's outcome is unknown; does nothing once posts_per_day posts went out on now's local day."""
    posts_per_day = settings.SOCIAL_POSTS_PER_DAY if posts_per_day is None else posts_per_day
    month = f"{now.year}-{now.month:02d}"
    drafts = ledger.drafts()
    stuck = [d for d, v in drafts.items() if v["state"] == "posting"]
    if stuck:
        raise SocialError(f"outcome unknown for {', '.join(stuck)}: check the profile, then `bio social-resolve` each one")
    today = now.date().isoformat()
    posted_today = [r for r in ledger.records() if r["event"] == "posted" and r["at"][:10] == today]
    if len(posted_today) >= posts_per_day:
        return {"status": "done for today", "posted_today": [r["draft_id"] for r in posted_today]}
    ready = [(v["seq"], d) for d, v in drafts.items() if v["state"] == "approved" and v["month"] == month]
    if not ready:
        return {"status": "nothing approved", "month": month}
    draft_id = min(ready)[1]
    ddir = Path(drafts_dir) / draft_id
    if draft_hash(ddir) != drafts[draft_id]["approved_hash"]:
        raise SocialError(f"draft {draft_id} changed after it was approved: review it and approve it again")
    draft = load_draft(ddir)
    images = [(ddir / i["file"]).read_bytes() for i in draft["images"]]
    payload = {"draft_id": draft_id, "text": segments_text(draft["text"]), "alts": [i["alt"] for i in draft["images"]],
               "image_bytes": [len(b) for b in images]}
    if not live:
        return {"status": "dry-run", **payload}
    from atproto import models

    tb = text_builder(draft["text"])
    ratios = [models.AppBskyEmbedDefs.AspectRatio(width=i["width"], height=i["height"]) for i in draft["images"]]
    try:
        client = client_factory()
    except Exception as e:  # noqa: BLE001 -- nothing was sent: recorded as failed with its message, then re-raised unchanged
        ledger.append("failed", draft_id, obs_id=draft["obs_id"], month=month, error=f"login: {type(e).__name__}: {e}"[:500])
        raise
    ledger.append("posting", draft_id, obs_id=draft["obs_id"], month=month)
    try:
        resp = client.send_images(text=tb, images=images, image_alts=payload["alts"], langs=["en"], image_aspect_ratios=ratios)
    except Exception as e:  # noqa: BLE001 -- the post may or may not exist now: the draft stays "posting" and blocks until resolved
        raise SocialError(f"posting {draft_id} failed with an unknown outcome ({type(e).__name__}: {e}); check the profile, then "
                          f"`bio social-resolve {draft_id} --posted-uri <at:// uri>` or `--not-posted`") from e
    ledger.append("posted", draft_id, obs_id=draft["obs_id"], month=month, uri=resp.uri, cid=resp.cid)
    return {"status": "posted", "uri": resp.uri, **payload}
