"""Content gates for the Bluesky posting flow: automatic checks that stand in for the owner's review of each post.

  alt_text_privacy  (draft, priority 10)    an alt text never carries coordinates, a street address, or (for an observation that hides its
                                            location) a token of its place name
  image_floor       (draft, priority 30)    a black, grossly under/over-exposed or far too blurry photo is not drafted
  live_recheck      (pre_send, priority 20) right before the send every observation of the draft is read from iNat again; a deleted, casual,
                                            taxon-less or re-identified one, or one whose location became hidden, withdraws the draft

Each gate is a hook (social_hooks) that runs only while its rule is on; register_hooks() is called by social_hooks.load_all()."""
from __future__ import annotations

import re
import socket
import unicodedata
import urllib.error
from pathlib import Path

from . import social, social_hooks
from .social_hooks import Verdict
from .social_rules import Audit

# ------------------------------------------------------------------------------------------------------ alt_text_privacy

# Coordinates, in the forms a photographer or a GPS app writes them. Unicode minus signs are normalised to "-" first (see _normalise).
#   decimal pair   "40.4123, -75.9876", "40.41234 -75.98765", "+40.4123;-75.9876", comma decimals "40,4123 -75,9876", also at the end of a sentence
#   labelled       "lat 40.4", "Longitude: -75.9", "lng=-75.9"
#   hemisphere     "40.4123 N 75.9876 W", "N 40.41 W 75.98", "40.4123° N"
#   DMS            40°24'44"N 75°59'15"W (degrees and minutes with their marks)
# A number must have 3+ decimals (comma decimals: 4+) so a measurement like "2.5, 3.1 cm", "f/2.8", "10.5 mm", "4000 px" or "2025" never matches.
_NUM = r"(?:[+\-]?\d{1,3}\.\d{3,}|[+\-]?\d{1,3},\d{4,})"
_ANGLE = r"\d{1,3}(?:[.,]\d+)?"
COORDINATES = re.compile(rf"(?<![\w.,]){_NUM}\s*(?:[,;/]\s*|\s)\s*{_NUM}(?!\d)")
LABELLED_COORDINATE = re.compile(rf"\b(?:lat(?:itude)?|lon(?:gitude)?|lng)\b\.?\s*[:=]?\s*[+\-]?{_ANGLE}", re.IGNORECASE)
HEMISPHERE_PAIR = re.compile(rf"(?<![\d.,]){_ANGLE}\s*°?\s*[NS]\b[\s,;/]*{_ANGLE}\s*°?\s*[EW]\b"
                             rf"|(?<![\w.,])[NS]\s*{_ANGLE}\s*°?[\s,;/]*[EW]\s*{_ANGLE}")
HEMISPHERE_SINGLE = re.compile(r"(?<![\d.,])\d{1,3}\.\d{3,}\s*°?\s*[NSEW]\b")
DMS = re.compile(r"\d{1,3}\s*°\s*\d{1,2}(?:[.,]\d+)?\s*['′’]")
COORDINATE_FORMS = (COORDINATES, LABELLED_COORDINATE, HEMISPHERE_PAIR, HEMISPHERE_SINGLE, DMS)
_MINUS_SIGNS = dict.fromkeys(map(ord, "−‒–—‐‑﹣－"), "-")
# A house number, one to three capitalised words, then a street word: "1234 Old Mill Rd", "7 Oak Street". The street word must be capitalised
# so "2 long drive" or "3 ways" is not an address.
STREET_WORDS = "St|Street|Rd|Road|Ave|Avenue|Dr|Drive|Ln|Lane|Blvd|Way|Ct|Court"
ADDRESS = re.compile(rf"\b\d{{1,6}}\s+(?:[A-Z][\w'-]*\s+){{1,3}}(?:{STREET_WORDS})\b")
# Place tokens are only taken from a place name when 4+ letters long and not one of these generic words (which say nothing about where).
PLACE_TOKEN_MIN = 4
PLACE_STOPLIST = frozenset({"county", "state", "usa", "united", "states", "township", "park", "city", "town", "borough", "village",
                            "north", "south", "east", "west", "national", "forest", "near", "area", "region", "province"})
# A place token that is also an everyday word only counts as a leak when the alt text writes it capitalised ("Black Rock", not "a black rock").
COMMON_WORDS = frozenset({"black", "white", "red", "green", "blue", "yellow", "brown", "gray", "grey", "rock", "rocks", "stone", "hill", "hills",
                          "ridge", "creek", "river", "lake", "pond", "spring", "springs", "woods", "wood", "field", "fields", "farm", "mill",
                          "mountain", "mountains", "valley", "trail", "bridge", "road", "swamp", "marsh", "meadow", "island", "little", "long",
                          "high", "great", "old", "upper", "lower", "middle", "falls", "glen", "pine", "oak", "maple", "cedar", "beaver", "bucks",
                          "bear", "deer", "eagle", "hawk", "fox", "wolf", "bald", "sandy", "rocky", "cold", "clear", "round", "flat", "grove"})
SEASON_WORDS = frozenset({"spring", "summer", "fall", "autumn", "winter"})
HIDING = ("private", "obscured-private", "obscured")  # geoprivacy / taxon_geoprivacy values under which the true (or even a general) place is not for the text
TAXON_HIDING = HIDING                                 # kept as a name: a taxon-obscured observation's coordinates must not leak either
LEAK_COORDINATES, LEAK_ADDRESS, LEAK_PLACE = "coordinates", "street address", "hidden place name"


def _normalise(text: str) -> str:
    return unicodedata.normalize("NFKC", text).translate(_MINUS_SIGNS)


def _fold(word: str) -> str:
    """Case- and accent-folded form of a word ("Río" == "RIO" == "rio")."""
    return "".join(c for c in unicodedata.normalize("NFKD", word.casefold()) if not unicodedata.combining(c))


def has_coordinates(text: str) -> bool:
    text = _normalise(text)
    return any(rx.search(text) for rx in COORDINATE_FORMS)


def hides_place(obs: dict) -> bool:
    """geoprivacy or taxon_geoprivacy says the place is private or obscured."""
    return (obs.get("geoprivacy") or "") in HIDING or (obs.get("taxon_geoprivacy") or "") in HIDING


def location_hidden(obs: dict) -> bool:
    """True when the observation hides its location: no public place, or private/obscured geoprivacy, or a taxon geoprivacy that hides it."""
    return social.public_place(obs) is None or hides_place(obs)


def place_tokens(place_guess: str | None) -> list[str]:
    """The place name's words (split on commas and spaces) of PLACE_TOKEN_MIN+ characters that are not generic. A token is letters, or letters
    with digits ("SGL220"); pure numbers are not tokens."""
    words = re.split(r"[,\s]+", place_guess or "")
    out = []
    for w in words:
        w = w.strip(".()'\"")
        if (len(w) >= PLACE_TOKEN_MIN and w.isalnum() and any(c.isalpha() for c in w) and w.casefold() not in PLACE_STOPLIST
                and _fold(w) not in [_fold(t) for t in out]):
            out.append(w)
    return out


_CAPTION_PLACE = re.compile(r"\bphotographed\s+(?:at|in|near)\s+(.+?)(?:\s+on\s+(?:\d|[A-Z][a-z]+\s+\d)|\.?\s*$)")
_TITLE_SPLIT = re.compile(r"\s+[—–-]\s+")


def label_places(label: dict | None) -> list[str]:
    """The place phrases of a Stock label: the caption's "photographed at <place>" and the title's context after the dash (a season is not one);
    keywords are not read (a keyword cannot be told from a taxon term), the redundant ones are caught through these phrases."""
    label = label or {}
    out = []
    m = _CAPTION_PLACE.search(_normalise(label.get("caption") or ""))
    if m:
        out.append(m.group(1))
    parts = _TITLE_SPLIT.split(label.get("title") or "")
    if len(parts) > 1 and parts[-1].strip().casefold() not in SEASON_WORDS:
        out.append(parts[-1])
    return out


def forbidden_tokens(obs: dict, label: dict | None = None) -> list[str]:
    """The place words an alt text must not carry for an observation whose location is hidden: the iNat place guess, the place text the
    draft shows, and the place in the Stock label (title context, caption)."""
    out: list[str] = []
    for phrase in [obs.get("place_guess"), social.public_place(obs), *label_places(label)]:
        for t in place_tokens(phrase):
            if _fold(t) not in [_fold(o) for o in out]:
                out.append(t)
    return out


def label_without_place(label: dict, obs: dict) -> dict:
    """The Stock label with everything that names the place removed, for the vision model when the location is hidden: the title keeps its
    name only, the caption goes, keywords that spell a place word go."""
    tokens = {_fold(t) for t in forbidden_tokens(obs, label)}
    parts = _TITLE_SPLIT.split(label.get("title") or "")
    title = parts[0] if len(parts) > 1 and parts[-1].strip().casefold() not in SEASON_WORDS else label.get("title")
    keywords = [k for k in (label.get("keywords") or []) if not ({_fold(w) for w in re.findall(r"[^\W_]+", k)} & tokens)]
    return {**label, "title": title, "caption": None, "keywords": keywords}


def alt_leaks(alt: str, obs: dict, label: dict | None = None) -> list[str]:
    """The KINDS of leak in the alt text ([] = clean): "coordinates", "street address", "hidden place name". Never the leaked text itself
    (reasons land in the audit log, social.log, the digest and the backups)."""
    found = []
    if has_coordinates(alt):
        found.append(LEAK_COORDINATES)
    if ADDRESS.search(_normalise(alt)):
        found.append(LEAK_ADDRESS)
    if location_hidden(obs):
        tokens = {_fold(t): t for t in forbidden_tokens(obs, label)}
        for w in re.findall(r"[^\W_]+", alt):
            key = _fold(w)
            if key in tokens and (w[0].isupper() or key not in COMMON_WORDS):
                found.append(LEAK_PLACE)
                break
    return found


def alt_text_privacy(ctx: dict) -> Verdict | None:
    leaks = alt_leaks(ctx.get("alt") or "", ctx["obs"], ctx.get("label"))
    if leaks:
        return Verdict("reject", "alt text leaks " + "; ".join(leaks))
    return None


# ------------------------------------------------------------------------------------------------------------ image_floor

ANALYSIS_EDGE = 1024                 # photos are measured at this long edge, so the floors do not depend on the upload size
BLACK_LUMA = 2.0                     # median luma (0-255) below this: a black or nearly black frame
UNDER_LUMA = 14.0                    # median luma below this: grossly under-exposed (the Stock floor's p10 is a typical shot's lower tail)
OVER_LUMA = 247.0                    # median luma above this: blown out
BLUR_SHARP = 8.0                     # tile sharpness (stock_select.tile_sharpness) below this: far too blurry. Calibrated on 153 real shots (min 17.4, p5 121) vs the same shots Gaussian-blurred at radius 6 on a 2048 px frame (2.2-4.1)
MIN_LONG_EDGE = 256                  # prepare_image never goes below 256 px; a smaller picture is not a real photo


def measure_image(path: Path) -> dict:
    """{"width", "height", "luma", "sharp"} of an image (sharp is None when the picture is too small for the tile grid)."""
    import numpy as np
    from PIL import Image

    from .stock_select import TILES, _small, median_luma, tile_sharpness

    with Image.open(path) as im:
        width, height = im.size
        small = np.asarray(_small(im.convert("RGB"), ANALYSIS_EDGE))
    sharp = tile_sharpness(small).sharp if min(small.shape[:2]) >= TILES * 3 else None
    return {"width": width, "height": height, "luma": median_luma(small), "sharp": sharp}


def image_problem(m: dict) -> str | None:
    """The first floor a measurement falls below, as a reason; None for a postable photo."""
    if m["luma"] < BLACK_LUMA:
        return f"image is black (median luma {m['luma']:.1f} < {BLACK_LUMA})"
    if m["luma"] < UNDER_LUMA:
        return f"image is grossly under-exposed (median luma {m['luma']:.1f} < {UNDER_LUMA})"
    if m["luma"] > OVER_LUMA:
        return f"image is grossly over-exposed (median luma {m['luma']:.1f} > {OVER_LUMA})"
    if max(m["width"], m["height"]) < MIN_LONG_EDGE:
        return f"image is too small ({m['width']}x{m['height']}, long edge < {MIN_LONG_EDGE})"
    if m["sharp"] is not None and m["sharp"] < BLUR_SHARP:
        return f"image is far too blurry (sharpness {m['sharp']:.2f} < {BLUR_SHARP})"
    return None


def image_floor(ctx: dict) -> Verdict | None:
    problem = image_problem(measure_image(Path(ctx["image"])))
    return Verdict("reject", problem) if problem else None


# ----------------------------------------------------------------------------------------------------------- live_recheck

def draft_observations(draft: dict) -> dict[int, dict]:
    """{obs id: the observation as the draft was written with it}: one for a single, one per photo for a themed draft."""
    if draft.get("observations"):
        return {int(k): v for k, v in draft["observations"].items()}
    return {int(draft["obs_id"]): draft.get("observation") or {}}


def _taxon_name(obs: dict) -> str:
    return ((obs.get("taxon") or {}).get("name") or "").strip()


def _levels(obs: dict) -> tuple:
    return (obs.get("geoprivacy") or None, obs.get("taxon_geoprivacy") or None)


def recheck_problem(obs_id: int, then: dict, now: dict, text: str) -> str | None:
    """Why the draft written with `then` is no longer fine to send now that iNat says `now` (None = still fine). Reasons never repeat the
    place (it may be the very thing iNat now hides)."""
    if now.get("quality_grade") == "casual":
        return f"observation {obs_id} is now casual"
    if not _taxon_name(now):
        return f"observation {obs_id} now has no taxon"
    if _taxon_name(then) and _taxon_name(now) != _taxon_name(then):
        return f"observation {obs_id} was re-identified: {_taxon_name(then)} -> {_taxon_name(now)}"
    shown = social.public_place(then)
    if shown and shown in text:
        # hidden now (private; obscured or taxon-obscured when that is new since the draft: an unchanged obscured place was already the
        # public, generalised one when the draft was written, but a draft written before taxon_geoprivacy was recorded cannot prove it)
        if social.public_place(now) is None or (hides_place(now) and _levels(now) != _levels(then)):
            return f"observation {obs_id} now hides its location but the post still shows its place"
        if social.public_place(now) != shown:
            return f"observation {obs_id} has a different place on iNat than the one the post shows"
    return None


def _transient_errors() -> tuple:
    from .inat import InatTransientError

    return (InatTransientError, urllib.error.URLError, TimeoutError, ConnectionError, socket.timeout, socket.gaierror, socket.herror)


UNAVAILABLE_BLOCK_AFTER = 3          # unavailable rechecks (observations iNat could not be asked about) in ONE run after which live_recheck blocks the send
                                     # instead of failing open: with iNat unreachable for that many observations the privacy of a multi-photo draft cannot be
                                     # shown. A single-observation draft still fails open (one unavailable recheck), as before.


def _fresh_alt_leaks(alt: str, then: dict, now: dict) -> list[str]:
    """alt_leaks against the observation as iNat says it is NOW; when it now hides its location, the place words come from the draft's own
    observation (iNat no longer reports the place)."""
    leaks = alt_leaks(alt, now)
    if not leaks and location_hidden(now):
        leaks = alt_leaks(alt, {**then, "geoprivacy": "private"})
    return leaks


def _unavailable_in_run(audit, fallback: int) -> int:
    try:
        return sum(1 for r in audit.records() if r.get("run") == audit.run and r.get("event") == "live_recheck_unavailable")
    except (OSError, ValueError):
        return fallback


def live_recheck(ctx: dict) -> Verdict | None:
    """Fail open ONLY when iNat cannot be reached (a transient API error, a network or socket error, a timeout): audited as
    live_recheck_unavailable. Any other exception (a coding error, an unexpected API answer) propagates, and the framework blocks the send.

    After the first unavailable answer the draft's remaining observations are not asked (each would spend the transport's whole retry budget,
    minutes): they are recorded as unavailable too (skipped=true). UNAVAILABLE_BLOCK_AFTER unavailable observations in one run block the send.
    The stored alt texts are checked against the observations as just fetched, so an alt that names a place is withdrawn when the observation
    turned private or obscured after drafting."""
    from .inat import InatNotFoundError

    draft, audit = ctx["draft"], ctx.get("audit") or Audit()
    text = social.segments_text(draft.get("text") or [])
    unavailable = _transient_errors()
    observed = draft_observations(draft)
    fetched: dict[int, dict] = {}
    down, count = None, 0
    for obs_id, then in observed.items():
        if down is not None:
            count += 1
            audit.log("live_recheck_unavailable", draft_id=ctx.get("draft_id"), obs_id=obs_id, error=down, skipped=True)
            continue
        try:
            now = ctx["fetch"](obs_id)
        except InatNotFoundError:
            return Verdict("withdraw", f"observation {obs_id} no longer exists on iNat")
        except unavailable as e:
            down = f"{type(e).__name__}: {e}"[:300]
            count += 1
            audit.log("live_recheck_unavailable", draft_id=ctx.get("draft_id"), obs_id=obs_id, error=down)
            continue
        problem = recheck_problem(obs_id, then, now, text)
        if problem:
            return Verdict("withdraw", problem)
        fetched[obs_id] = now
    single = next(iter(observed)) if len(observed) == 1 else None
    for i, image in enumerate(draft.get("images") or [], 1):
        obs_id = int(image["obs_id"]) if image.get("obs_id") is not None else (single if single is not None else int(draft.get("obs_id", -1)))
        if obs_id in fetched and obs_id in observed:
            leaks = _fresh_alt_leaks(image.get("alt") or "", observed[obs_id], fetched[obs_id])
            if leaks:
                return Verdict("withdraw", f"alt text of image {i} leaks {'; '.join(leaks)} now that observation {obs_id} hides its location")
    if count and _unavailable_in_run(audit, count) >= UNAVAILABLE_BLOCK_AFTER:
        return Verdict("block", f"iNat could not be asked about {_unavailable_in_run(audit, count)} observations in this run "
                                f"(limit {UNAVAILABLE_BLOCK_AFTER}); the live recheck cannot vouch for the draft, nothing is sent")
    return None


# ------------------------------------------------------------------------------------------------------------- gate_rescreen

def _image_observation(draft: dict, image: dict) -> dict:
    per_image = (draft.get("observations") or {}).get(str(image.get("obs_id")))
    return per_image or draft.get("observation") or {}


def _rescreen_withdraw(ctx: dict, gate: str, reason: str) -> Verdict:
    (ctx.get("audit") or Audit()).log("gate_rescreen", draft_id=ctx.get("draft_id"), gate=gate, reason=reason)
    return Verdict("withdraw", reason)


def rescreen_privacy(ctx: dict) -> Verdict | None:
    """The alt_text_privacy check on every alt text of the STORED draft (so drafts written before the gate, or before it was improved, are
    screened before they are sent). The reason names the image and the kind of leak, never the text."""
    draft = ctx["draft"]
    for i, image in enumerate(draft.get("images") or [], 1):
        leaks = alt_leaks(image.get("alt") or "", _image_observation(draft, image))
        if leaks:
            return _rescreen_withdraw(ctx, "alt_text_privacy", f"alt text of image {i} leaks " + "; ".join(leaks))
    return None


def rescreen_image(ctx: dict) -> Verdict | None:
    """The image_floor check on every image file of the STORED draft."""
    draft = ctx["draft"]
    for i, image in enumerate(draft.get("images") or [], 1):
        path = Path(ctx["draft_dir"]) / image["file"]
        try:
            problem = image_problem(measure_image(path))
        except OSError:
            problem = "image file is missing or unreadable"
        if problem:
            return _rescreen_withdraw(ctx, "image_floor", f"image {i} ({image['file']}): {problem}")
    return None


# -------------------------------------------------------------------------------------------------------------- extension

def register_hooks() -> None:
    social_hooks.register("alt_text_privacy", "draft", alt_text_privacy, rule="alt_text_privacy", priority=10, critical=True)
    social_hooks.register("image_floor", "draft", image_floor, rule="image_floor", priority=30, critical=True)
    social_hooks.register("live_recheck", "pre_send", live_recheck, rule="live_recheck", priority=20, critical=True)
    social_hooks.register("gate_rescreen_privacy", "pre_send", rescreen_privacy, rule="alt_text_privacy", priority=25, critical=True)
    social_hooks.register("gate_rescreen_image", "pre_send", rescreen_image, rule="image_floor", priority=25, critical=True)
