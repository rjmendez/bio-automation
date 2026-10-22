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
import tomllib
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Iterator

from PIL import Image, ImageOps

from . import settings

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
STATES = ("drafted", "approved", "rejected", "posting", "posted", "failed")
NOTES = ("error",)    # recorded, not a state change


class SocialError(Exception):
    """A social-post step cannot go on; the message says what to do."""


class VisionError(SocialError):
    """The local vision model is missing, failed, or did not run fully on the GPU."""


class SkipObservation(SocialError):
    """This observation cannot be posted (casual, deleted, no taxon); the next candidate is tried."""


class UnusableAnswer(SkipObservation):
    """The vision model's answer is not alt text (too short, or JSON it could not be read from): the observation is skipped this run."""


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
        if event not in STATES + NOTES:
            raise SocialError(f"unknown ledger event {event!r}")
        rec = {"at": (now or dt.datetime.now(settings.LOCAL_TZ)).isoformat(timespec="seconds"), "event": event, "draft_id": draft_id, **fields}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
        return rec

    def drafts(self) -> dict[str, dict]:
        """{draft_id: {"state", "obs_id", "month", "drafted_at", "last": <last state record>, "seq": its ledger position, "approved_hash",
        "approved_seq": position of the owner's approval, "last_error": the last error note}}."""
        out: dict[str, dict] = {}
        for seq, r in enumerate(self.records()):
            d = out.setdefault(r["draft_id"], {"obs_id": r.get("obs_id"), "month": r.get("month"), "drafted_at": r["at"], "approved_hash": None})
            if r["event"] in NOTES:
                d["last_error"] = r
                continue
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
                   + [["text", "\nAlt text: local AI model, reviewed.\n"]] + [s for t in tags for s in (["tag", "#" + t, t], ["text", " "])][:-1])
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
               prints_url: str | None = None, now: dt.datetime | None = None) -> dict:
    """Write drafts_dir/<month>-obs<obs_id>/ (draft.json + 1.jpg..4.jpg) and return the draft. The observation is fetched live: casual,
    deleted and taxon-less observations raise SkipObservation. style(subject) -> {"rules", "examples", ...}: the owner's alt-text style
    for this photo (social_style.style_for), given to the vision model and recorded per image."""
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
        if style is not None:
            subject = " ".join(x for x in (context["common_name"], context["scientific_name"], label.get("title"),
                                           ", ".join((label.get("keywords") or [])[:10])) if x)
            context["style"] = style(subject)
        try:
            context["attempts"] = []
            alt = describe_fn(out / name, context, model=model)
        except SkipObservation:
            shutil.rmtree(out)           # no half-written draft folder is left behind
            raise
        images.append({"file": name, "source": rel, "alt": alt, **info, "alt_attempts": context["attempts"],
                       **({"style_examples": context["style"]["examples"]} if style is not None else {})})
    draft = {"draft_id": draft_id, "month": month, "obs_id": obs_id, "url": INAT_OBS_URL.format(obs_id), "text": text_segments(obs, prints_url),
             "images": images, "vision_model": model, "observation": obs,
             "created_at": (now or dt.datetime.now(settings.LOCAL_TZ)).isoformat(timespec="seconds")}
    (out / "draft.json").write_text(json.dumps(draft, indent=1, ensure_ascii=False), encoding="utf-8")
    return draft


def default_style() -> Callable[[str], dict]:
    """style(subject) from the saved profile (settings.SOCIAL_STYLE_PROFILE) and the pool's embed model."""
    from .social_style import load_profile, style_for

    profile = load_profile()
    _, embed = embedder()
    return lambda subject: {**style_for(profile, subject, embed=embed), "profile_built_at": profile["built_at"]}


def draft_month(index: dict, ledger: Ledger, *, today: dt.date, gallery_dir: Path, drafts_dir: Path, keep_pending: int | None = None,
                model: str | None = None, fetch: Callable[[int], dict] = _fetch, read_label: Callable[[Path], dict] = _read_label,
                describe_fn: Callable[..., str] = describe, pick_model: Callable[[], str] = vision_model,
                style_source: Callable[[], Callable[[str], dict]] | None = None, prints: dict[int, str] | None = None) -> dict:
    """Top up `today`'s month's unposted drafts (drafted or approved) to keep_pending, from the observations with gallery files in that
    month of any year that are not taken: posted, rejected or awaiting an outcome (posting) ever, or drafted in this month already. A draft
    of an earlier month that was never posted has expired (only the current month posts), so its observation comes back next year.
    Observations listed in `prints` are candidates whatever month they were photographed in (owner, 2026-10-22: ignore the month rule for
    them) and are drafted first; the draft belongs to today's month, so it posts this month.
    The owner's alt-text style is required: style_source() (default: the saved profile + the pool's embed model) is loaded once, before
    any drafting; a missing profile raises and drafts nothing. prints ({obs_id: url}, default load_prints()) adds a print link to the
    observations it lists. Returns {"month", "pending", "drafted", "skipped"}."""
    keep_pending = settings.SOCIAL_KEEP_PENDING if keep_pending is None else keep_pending
    month = f"{today.year}-{today.month:02d}"
    known = ledger.drafts()
    pending = [d for d, v in known.items() if v["month"] == month and v["state"] in ("drafted", "approved")]
    report = {"month": month, "pending": len(pending), "drafted": [], "skipped": {}}
    need = keep_pending - len(pending)
    if need <= 0:
        return report
    seen = {int(v["obs_id"]) for v in known.values()
            if v.get("obs_id") is not None and (v["state"] in ("posted", "rejected", "posting") or v["month"] == month)}
    prints = load_prints() if prints is None else prints
    cands = month_observations(index, today.month)
    for rel, obs in index["files"].items():                       # an observation with a print is a candidate whatever month it was shot in
        if int(obs) in prints and int(obs) not in cands:
            cands.setdefault(int(obs), []).append(rel)
    cands = {o: sorted(r) for o, r in cands.items()}
    model = model or pick_model()
    style = (style_source or default_style)()
    order = order_candidates(cands, seen, seed=today.isoformat())
    order.sort(key=lambda o: o not in prints)                     # observations with a print first (stable: the year rotation is kept inside each group)
    for obs_id in order:
        if len(report["drafted"]) >= need:
            break
        try:
            draft = make_draft(obs_id, cands[obs_id], month=month, gallery_dir=gallery_dir, drafts_dir=drafts_dir, model=model,
                               fetch=fetch, read_label=read_label, describe_fn=describe_fn, style=style, prints_url=prints.get(obs_id))
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


def post_next(ledger: Ledger, drafts_dir: Path, *, now: dt.datetime, live: bool = False, posts_per_day: int | None = None,
              send: Callable[[dict], tuple[str, str]] = bsky_post) -> dict:
    """Post the oldest approved draft of now's month. Dry-run (the default) returns what would be sent and records nothing.
    Refuses while any post's outcome is unknown; does nothing once posts_per_day posts went out on now's local day.

    Live: "posting" is recorded before the send. A send that stopped before anything went out (NothingSent) records an "error"
    note and the draft stays approved for the next run; any other failure leaves it "posting" (outcome unknown -> social-resolve)."""
    if not live:
        return _post_next(ledger, drafts_dir, now=now, live=False, posts_per_day=posts_per_day, send=send)
    with ledger_lock(ledger.path):                                    # the checks, the "posting" record and the send are one step
        return _post_next(ledger, drafts_dir, now=now, live=True, posts_per_day=posts_per_day, send=send)


def _post_next(ledger: Ledger, drafts_dir: Path, *, now: dt.datetime, live: bool, posts_per_day: int | None,
               send: Callable[[dict], tuple[str, str]]) -> dict:
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
    ready = [(v["approved_seq"], d) for d, v in drafts.items() if v["state"] == "approved" and v["month"] == month]
    if not ready:
        return {"status": "nothing approved", "month": month}
    draft_id = min(ready)[1]
    ddir = Path(drafts_dir) / draft_id
    if draft_hash(ddir) != drafts[draft_id]["approved_hash"]:
        raise SocialError(f"draft {draft_id} changed after it was approved: review it and approve it again")
    draft = load_draft(ddir)
    request = {"text": segments_text(draft["text"]), "langs": ["en"],
               "images": [{"path": str(ddir / i["file"]), "alt": i["alt"], "width": i["width"], "height": i["height"]} for i in draft["images"]]}
    payload = {"draft_id": draft_id, "text": request["text"], "alts": [i["alt"] for i in draft["images"]],
               "image_bytes": [(ddir / i["file"]).stat().st_size for i in draft["images"]]}
    if not live:
        return {"status": "dry-run", **payload}
    ledger.append("posting", draft_id, obs_id=draft["obs_id"], month=month, now=now)
    try:
        uri, cid = send(request)
    except NothingSent as e:
        ledger.append("approved", draft_id, obs_id=draft["obs_id"], month=month, hash=drafts[draft_id]["approved_hash"], restored=True, now=now)
        ledger.append("error", draft_id, obs_id=draft["obs_id"], month=month, error=str(e)[:500], now=now)
        raise SocialError(f"nothing was posted ({e}); {draft_id} stays approved for the next run") from e
    except Exception as e:  # noqa: BLE001 -- the post may or may not exist now: the draft stays "posting" and blocks until resolved
        raise SocialError(f"posting {draft_id} failed with an unknown outcome ({type(e).__name__}: {e}); check the profile, then "
                          f"`bio social-resolve {draft_id} --posted-uri <at:// uri>` or `--not-posted`") from e
    ledger.append("posted", draft_id, obs_id=draft["obs_id"], month=month, uri=uri, cid=cid, now=now)
    return {"status": "posted", "uri": uri, **payload}
