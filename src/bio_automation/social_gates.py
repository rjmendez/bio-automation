"""Content gates for the Bluesky posting flow: automatic checks that stand in for the owner's review of each post.

  alt_text_privacy  (draft, priority 10)    an alt text never carries coordinates, a street address, or (for an observation that hides its
                                            location) a token of its place name
  image_floor       (draft, priority 30)    a black, grossly under/over-exposed or far too blurry photo is not drafted
  live_recheck      (pre_send, priority 20) right before the send every observation of the draft is read from iNat again; a deleted, casual,
                                            taxon-less or re-identified one, or one whose location became hidden, withdraws the draft

Each gate is a hook (social_hooks) that runs only while its rule is on; register_hooks() is called by social_hooks.load_all()."""
from __future__ import annotations

import re
from pathlib import Path

from . import social, social_hooks
from .social_hooks import Verdict
from .social_rules import Audit

# ------------------------------------------------------------------------------------------------------ alt_text_privacy

# Two decimal numbers with 3+ decimals, comma-separated: "40.1234, -75.5678" (a measurement like "2.5, 3.1" has too few decimals to match).
COORDINATES = re.compile(r"(?<![\d.])-?\d{1,3}\.\d{3,}\s*[,;/]\s*-?\d{1,3}\.\d{3,}(?![\d.])")
# A house number, one to three capitalised words, then a street word: "1234 Old Mill Rd", "7 Oak Street". The street word must be capitalised
# so "2 long drive" or "3 ways" is not an address.
STREET_WORDS = "St|Street|Rd|Road|Ave|Avenue|Dr|Drive|Ln|Lane|Blvd|Way|Ct|Court"
ADDRESS = re.compile(rf"\b\d{{1,6}}\s+(?:[A-Z][\w'-]*\s+){{1,3}}(?:{STREET_WORDS})\b")
# Place tokens are only taken from a place name when 4+ letters long and not one of these generic words (which say nothing about where).
PLACE_TOKEN_MIN = 4
PLACE_STOPLIST = frozenset({"county", "state", "usa", "united", "states", "township", "park", "city", "town", "borough", "village",
                            "north", "south", "east", "west", "national", "forest", "near", "area", "region", "province"})
HIDING = ("private", "obscured-private")            # geoprivacy / taxon_geoprivacy values that keep the true place off the public record
TAXON_HIDING = HIDING + ("obscured",)               # a taxon-obscured observation: the owner's own coordinates must not leak either


def location_hidden(obs: dict) -> bool:
    """True when the observation hides its location: no public place, private/obscured-private geoprivacy, or a taxon geoprivacy that hides it."""
    return (social.public_place(obs) is None or (obs.get("geoprivacy") or "") in HIDING
            or (obs.get("taxon_geoprivacy") or "") in TAXON_HIDING)


def place_tokens(place_guess: str | None) -> list[str]:
    """The place name's words (split on commas and spaces) of PLACE_TOKEN_MIN+ letters that are not generic."""
    words = re.split(r"[,\s]+", place_guess or "")
    out = []
    for w in words:
        w = w.strip(".()'\"")
        if len(w) >= PLACE_TOKEN_MIN and w.isalpha() and w.casefold() not in PLACE_STOPLIST and w.casefold() not in [t.casefold() for t in out]:
            out.append(w)
    return out


def alt_leaks(alt: str, obs: dict) -> list[str]:
    """What the alt text gives away, as human reasons ([] = clean): coordinates, a street address, a hidden place's name."""
    found = []
    m = COORDINATES.search(alt)
    if m:
        found.append(f"coordinates ({m.group(0)!r})")
    m = ADDRESS.search(alt)
    if m:
        found.append(f"street address ({m.group(0)!r})")
    if location_hidden(obs):
        # place names are capitalised in prose; a lower-case "black" or "rock" is the colour or the thing, not "Black Rock Road"
        words = {w.casefold() for w in re.findall(r"\b[A-Z][a-z]+\b", alt)}
        hit = [t for t in place_tokens(obs.get("place_guess")) if t.casefold() in words]
        if hit:
            found.append(f"hidden place name ({', '.join(hit)})")
    return found


def alt_text_privacy(ctx: dict) -> Verdict | None:
    leaks = alt_leaks(ctx.get("alt") or "", ctx["obs"])
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


def recheck_problem(obs_id: int, then: dict, now: dict, text: str) -> str | None:
    """Why the draft written with `then` is no longer fine to send now that iNat says `now` (None = still fine)."""
    if now.get("quality_grade") == "casual":
        return f"observation {obs_id} is now casual"
    if not _taxon_name(now):
        return f"observation {obs_id} now has no taxon"
    if _taxon_name(then) and _taxon_name(now) != _taxon_name(then):
        return f"observation {obs_id} was re-identified: {_taxon_name(then)} -> {_taxon_name(now)}"
    shown = social.public_place(then)
    if shown and location_hidden(now) and shown in text:
        return f"observation {obs_id} now hides its location but the post still shows {shown!r}"
    return None


def live_recheck(ctx: dict) -> Verdict | None:
    from .inat import InatNotFoundError

    draft, audit = ctx["draft"], ctx.get("audit") or Audit()
    text = social.segments_text(draft.get("text") or [])
    for obs_id, then in draft_observations(draft).items():
        try:
            now = ctx["fetch"](obs_id)
        except InatNotFoundError:
            return Verdict("withdraw", f"observation {obs_id} no longer exists on iNat")
        except Exception as e:  # noqa: BLE001
            # FAIL OPEN, deliberately: an iNat outage or timeout must not stop all posting. The draft was already checked when it was
            # written; the recheck is an extra safety net, so when it cannot run we say so in the audit log and carry on.
            audit.log("live_recheck_unavailable", draft_id=ctx.get("draft_id"), obs_id=obs_id, error=f"{type(e).__name__}: {e}"[:300])
            continue
        problem = recheck_problem(obs_id, then, now, text)
        if problem:
            return Verdict("withdraw", problem)
    return None


# -------------------------------------------------------------------------------------------------------------- extension

def register_hooks() -> None:
    social_hooks.register("alt_text_privacy", "draft", alt_text_privacy, rule="alt_text_privacy", priority=10)
    social_hooks.register("image_floor", "draft", image_floor, rule="image_floor", priority=30)
    social_hooks.register("live_recheck", "pre_send", live_recheck, rule="live_recheck", priority=20)
