"""The owner's alt-text style, learned from their own Bluesky posts, for the alt text social.py drafts.

Source: the account's public post records (com.atproto.repo.listRecords on its PDS: no sign-in, no scope). Posts made by the old
inat_skeet bot are left out (their template text, and their alt text was model-written); what remains is alt text the owner wrote.

build_profile() is deterministic: measured statistics (length, sentences, openers, punctuation, person, emoji), the writing rules
that follow from them (each names the measurement behind it), and every owner alt text with an embedding from the pool's "embed"
model. Nothing in the profile is model-written. At draft time, style_for() picks the owner's alt texts closest to the photo's subject
(cosine over those embeddings) as examples, next to the rules.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import re
import statistics
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable

from . import settings

PROFILE_VERSION = 1
BOT_MARKERS = ("Quality grade:", "AI generated alt text")     # inat_skeet's fixed post template
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_FIRST_PERSON = re.compile(r"\b(I|my|me|we|our)\b")
_PHOTO_WORD = re.compile(r"\b(photo|image|picture)\b", re.IGNORECASE)
_EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
_MOOD_WORD = re.compile(r"\b(beautiful|stunning|gorgeous|lovely|amazing|cute|pretty|majestic|vibrant|delicate|serene|striking|breathtaking)\b",
                        re.IGNORECASE)
NUMBER_WORDS = {"two", "three", "four", "five", "six", "several", "multiple", "many"}


class StyleError(Exception):
    """The profile cannot be built or used; the message says what to do."""


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "bio-automation/social-style", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def resolve_account(handle: str, *, get_json: Callable[[str], dict] = _get_json) -> tuple[str, str]:
    """(did, pds endpoint) for a handle, from the public AppView and the DID document."""
    did = get_json(f"{settings.BSKY_APPVIEW}/xrpc/com.atproto.identity.resolveHandle?{urllib.parse.urlencode({'handle': handle})}")["did"]
    doc = get_json(f"{settings.PLC_URL}/{did}")
    for s in doc.get("service", []):
        if s.get("id", "").endswith("#atproto_pds"):
            return did, s["serviceEndpoint"].rstrip("/")
    raise StyleError(f"{did} names no #atproto_pds service")


def fetch_posts(did: str, pds: str, *, get_json: Callable[[str], dict] = _get_json) -> list[dict]:
    """Every app.bsky.feed.post record of the account (public listRecords, 100 per page)."""
    out, cursor = [], None
    while True:
        q = {"repo": did, "collection": "app.bsky.feed.post", "limit": 100, **({"cursor": cursor} if cursor else {})}
        page = get_json(f"{pds}/xrpc/com.atproto.repo.listRecords?{urllib.parse.urlencode(q)}")
        out += page.get("records", [])
        cursor = page.get("cursor")
        if not cursor or not page.get("records"):
            return out


def is_bot_post(text: str) -> bool:
    return any(m in text for m in BOT_MARKERS)


def owner_alt_texts(records: list[dict]) -> list[dict]:
    """[{alt, uri, created_at, post_text}] for every non-empty image alt text in posts the owner wrote, first occurrence only."""
    out, seen = [], set()
    for r in records:
        v = r.get("value", {})
        if is_bot_post(v.get("text", "")):
            continue
        embed = v.get("embed") or {}
        for img in embed.get("images") or (embed.get("media") or {}).get("images") or []:
            alt = (img.get("alt") or "").strip()
            if alt and alt not in seen:
                seen.add(alt)
                out.append({"alt": alt, "uri": r.get("uri"), "created_at": v.get("createdAt"), "post_text": v.get("text", "")})
    return out


def _share(items: list, pred) -> float:
    return round(sum(1 for x in items if pred(x)) / len(items), 2) if items else 0.0


def _pct(values: list[int], p: float) -> int:
    s = sorted(values)
    return s[min(len(s) - 1, int(len(s) * p))]


def measure(alts: list[str]) -> dict:
    """The statistics the rules are built from."""
    if not alts:
        raise StyleError("no owner-written alt text to learn from")
    chars = [len(a) for a in alts]
    sentences = [len([s for s in _SENTENCE.split(a) if s]) for a in alts]
    first = [a.split()[0].lower().strip(".,") for a in alts]
    return {"alt_texts": len(alts), "chars_median": int(statistics.median(chars)), "chars_p10": _pct(chars, 0.10),
            "chars_p90": _pct(chars, 0.90), "words_median": int(statistics.median(len(a.split()) for a in alts)),
            "sentences_median": int(statistics.median(sentences)), "sentences_p90": _pct(sentences, 0.90),
            "one_sentence_share": _share(sentences, lambda n: n == 1), "opens_with_article_share": _share(first, lambda w: w in ("a", "an")),
            "opens_with_count_share": _share(first, lambda w: w in NUMBER_WORDS), "ends_with_period_share": _share(alts, lambda a: a.endswith(".")),
            "first_person_share": _share(alts, lambda a: bool(_FIRST_PERSON.search(a))),
            "photo_word_share": _share(alts, lambda a: bool(_PHOTO_WORD.search(a))),
            "mood_word_share": _share(alts, lambda a: bool(_MOOD_WORD.search(a))), "emoji_count": sum(len(_EMOJI.findall(a)) for a in alts)}


def rules_from(stats: dict) -> list[str]:
    """Writing rules, each tied to the measurement it comes from."""
    r = [f"Length: about {stats['chars_median']} characters ({stats['words_median']} words); rarely over {stats['chars_p90']} "
         f"(median and 90th percentile of {stats['alt_texts']} of the photographer's own alt texts)."]
    if stats["one_sentence_share"] >= 0.5:
        r.append(f"Usually one sentence ({round(stats['one_sentence_share'] * 100)}% are), at most {max(2, stats['sentences_p90'])}.")
    if stats["opens_with_article_share"] >= 0.5:
        r.append(f"Start with the subject itself: 'A ...' or 'An ...' ({round(stats['opens_with_article_share'] * 100)}% do), "
                 "or with a count ('Two ...') when there are several.")
    if stats["photo_word_share"] < 0.15:
        r.append(f"Do not say 'photo', 'image' or 'picture' ({round(stats['photo_word_share'] * 100)}% do).")
    if stats["first_person_share"] < 0.15:
        r.append("Third person, no 'I' or 'my'.")
    if stats["ends_with_period_share"] >= 0.75:
        r.append("End with a period.")
    if stats["emoji_count"] == 0:
        r.append("No emoji.")
    if stats["mood_word_share"] < 0.1:
        r.append(f"Plain, concrete words (colour, shape, what it is on or doing); no words like 'beautiful', 'vibrant' or 'delicate' "
                 f"({round(stats['mood_word_share'] * 100)}% use one).")
    return r


def build_profile(handle: str, *, get_json: Callable[[str], dict] = _get_json, embed: Callable[[list[str]], list[list[float]]],
                  embed_model: str, now: dt.datetime | None = None) -> dict:
    did, pds = resolve_account(handle, get_json=get_json)
    records = fetch_posts(did, pds, get_json=get_json)
    alts = owner_alt_texts(records)
    stats = measure([a["alt"] for a in alts])
    vectors = embed([a["alt"] for a in alts])
    if len(vectors) != len(alts):
        raise StyleError(f"embedding returned {len(vectors)} vectors for {len(alts)} texts")
    return {"version": PROFILE_VERSION, "handle": handle, "did": did,
            "built_at": (now or dt.datetime.now(dt.timezone.utc)).isoformat(timespec="seconds"),
            "source": {"posts": len(records), "bot_posts_excluded": sum(1 for r in records if is_bot_post(r.get("value", {}).get("text", "")))},
            "stats": stats, "rules": rules_from(stats), "embed_model": embed_model,
            "examples": [{**a, "vector": v} for a, v in zip(alts, vectors)]}


def save_profile(profile: dict, path: Path | None = None) -> Path:
    path = Path(path or settings.SOCIAL_STYLE_PROFILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(profile, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return path


def load_profile(path: Path | None = None) -> dict:
    path = Path(path or settings.SOCIAL_STYLE_PROFILE)
    if not path.is_file():
        raise StyleError(f"no style profile at {path}: run `bio social-style` first")
    profile = json.loads(path.read_text(encoding="utf-8"))
    if profile.get("version") != PROFILE_VERSION:
        raise StyleError(f"style profile {path} is version {profile.get('version')}, expected {PROFILE_VERSION}: run `bio social-style` again")
    return profile


def _cosine(a: list[float], b: list[float]) -> float:
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0


def nearest_examples(profile: dict, query_vector: list[float], k: int = 5) -> list[str]:
    """The k owner alt texts closest to the query (highest cosine first; ties keep profile order)."""
    scored = sorted(enumerate(profile["examples"]), key=lambda ie: (-_cosine(query_vector, ie[1]["vector"]), ie[0]))
    return [e["alt"] for _, e in scored[:k]]


def style_for(profile: dict, subject: str, *, embed: Callable[[list[str]], list[list[float]]], k: int = 5) -> dict:
    """{"rules", "examples", "max_chars"} for one photo: the profile's rules and the owner's k alt texts nearest to `subject`."""
    return {"rules": profile["rules"], "examples": nearest_examples(profile, embed([subject])[0], k), "max_chars": profile["stats"]["chars_p90"]}


def loci_record(profile: dict, examples: int = 12) -> dict:
    """The profile as a structured memory record (no vectors): text for the finding, metadata for the structured fields."""
    ex = profile["examples"]
    picks = [e["alt"] for e in ex[::max(1, len(ex) // examples)][:examples]]     # spread over the whole history, not the newest few
    text = (f"Bluesky alt-text style of {profile['handle']} ({profile['did']}), learned {profile['built_at'][:10]} from "
            f"{profile['stats']['alt_texts']} alt texts the owner wrote (posts: {profile['source']['posts']}, inat_skeet bot posts "
            f"excluded: {profile['source']['bot_posts_excluded']}). Rules: " + " ".join(profile["rules"]) +
            " Examples: " + " | ".join(picks))
    meta = {"kind": "bsky_alt_text_style", "version": profile["version"], "handle": profile["handle"], "did": profile["did"],
            "built_at": profile["built_at"], "stats": profile["stats"], "rules": profile["rules"], "embed_model": profile["embed_model"],
            "examples_sample": picks, "profile_path": str(settings.SOCIAL_STYLE_PROFILE)}
    return {"text": text, "metadata": meta}
