"""social_style: the owner's alt-text style from their own public posts. The network (get_json) and the embedding model (embed) are the
stubbed dependencies; everything else runs for real."""
from __future__ import annotations

import json

import pytest

from bio_automation import social_style
from bio_automation.social_style import StyleError

BOT_TEXT = "Araneus thaddeus - Lattice Orbweaver\n\nQuality grade: research\n2024-10-01\nWarning! AI generated alt text."


def post(uri, text, alts, *, media=False, created="2025-01-01T00:00:00Z"):
    images = [{"alt": a, "image": {}} for a in alts]
    embed = {"$type": "app.bsky.embed.recordWithMedia", "media": {"images": images}} if media else {"$type": "app.bsky.embed.images", "images": images}
    return {"uri": uri, "value": {"text": text, "createdAt": created, "embed": embed}}


RECORDS = [
    post("at://me/p/1", "seed balls!", ["Wet mud balls full of seeds in a tray", ""]),
    post("at://me/p/2", BOT_TEXT, ["A spider sits in its web, model-written."]),
    post("at://me/p/3", "quote", ["A dry ball of mud with a milkweed seed attached."], media=True),
    post("at://me/p/4", "again", ["Wet mud balls full of seeds in a tray"]),
    {"uri": "at://me/p/5", "value": {"text": "no images"}},
]


class TestOwnerAltTexts:
    def test_bot_posts_empty_alts_and_repeats_are_left_out_and_quote_media_counts(self):
        assert social_style.owner_alt_texts(RECORDS) == [
            {"alt": "Wet mud balls full of seeds in a tray", "uri": "at://me/p/1", "created_at": "2025-01-01T00:00:00Z", "post_text": "seed balls!"},
            {"alt": "A dry ball of mud with a milkweed seed attached.", "uri": "at://me/p/3", "created_at": "2025-01-01T00:00:00Z", "post_text": "quote"}]

    def test_bot_marker_detection(self):
        assert social_style.is_bot_post(BOT_TEXT) is True
        assert social_style.is_bot_post("A Quality photo of a grade school") is False


ALTS = ["A small white beetle.", "An orange mushroom on a log. It is wet.", "Two green caterpillars on my hand",
        "A beautiful photo of a moth.", "A spider on a web."]


class TestMeasureAndRules:
    def test_statistics_are_exact(self):
        assert social_style.measure(ALTS) == {
            "alt_texts": 5, "chars_median": 28, "chars_p10": 18, "chars_p25": 21, "chars_p75": 33, "chars_p90": 39, "words_median": 6, "sentences_median": 1,
            "sentences_p90": 2, "one_sentence_share": 0.8, "opens_with_article_share": 0.8, "opens_with_count_share": 0.2,
            "ends_with_period_share": 0.8, "first_person_share": 0.2, "photo_word_share": 0.2, "mood_word_share": 0.2, "emoji_count": 0}

    def test_rules_follow_the_thresholds(self):
        loose = social_style.rules_from(social_style.measure(ALTS))
        assert loose == [
            "Length: 21-33 characters, typically about 28 (6 words); never under 18 or over 39 (middle half, median, 10th and 90th "
            "percentile of 5 of the photographer's own alt texts).",
            "Usually one sentence (80% are), at most 2.",
            "Start with the subject itself: 'A ...' or 'An ...' (80% do), or with a count ('Two ...') when there are several.",
            "End with a period.", "No emoji."]
        strict = social_style.rules_from(social_style.measure(["A beetle on a leaf.", "A moth on a wall.", "An ant on sand."]))
        assert "Third person, no 'I' or 'my'." in strict
        assert "Do not say 'photo', 'image' or 'picture' (0% do)." in strict
        assert strict[-1] == ("Plain, concrete words (colour, shape, what it is on or doing); no words like 'beautiful', 'vibrant' or 'delicate' "
                              "(0% use one).")

    def test_nothing_to_learn_from_raises(self):
        with pytest.raises(StyleError, match="no owner-written alt text to learn from"):
            social_style.measure([])


class RecordingGet:
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def __call__(self, url):
        self.calls.append(url)
        for prefix, answer in self.routes:
            if url.startswith(prefix):
                return answer(url) if callable(answer) else answer
        raise AssertionError(f"unexpected URL {url}")


def account_routes(pages):
    def list_records(url):
        return pages[1] if "cursor=c1" in url else pages[0]
    return [("https://public.api.bsky.app/xrpc/com.atproto.identity.resolveHandle", {"did": "did:plc:me"}),
            ("https://plc.directory/did:plc:me", {"service": [{"id": "#bsky_fg", "serviceEndpoint": "https://x"},
                                                              {"id": "#atproto_pds", "serviceEndpoint": "https://pds.example/"}]}),
            ("https://pds.example/xrpc/com.atproto.repo.listRecords", list_records)]


def fake_embed(texts):
    return [[float(len(t)), 1.0] for t in texts]


class TestBuild:
    def test_paged_fetch_and_profile(self):
        get = RecordingGet(account_routes([{"records": RECORDS[:3], "cursor": "c1"}, {"records": RECORDS[3:]}]))
        profile = social_style.build_profile("me.example", get_json=get, embed=fake_embed, embed_model="emb:1",
                                             now=__import__("datetime").datetime(2026, 10, 22, tzinfo=__import__("datetime").timezone.utc))
        assert (profile["did"], profile["source"], profile["embed_model"], profile["built_at"]) == (
            "did:plc:me", {"posts": 5, "bot_posts_excluded": 1}, "emb:1", "2026-10-22T00:00:00+00:00")
        assert [(e["alt"], e["vector"]) for e in profile["examples"]] == [
            ("Wet mud balls full of seeds in a tray", [37.0, 1.0]), ("A dry ball of mud with a milkweed seed attached.", [48.0, 1.0])]
        assert get.calls[2] == "https://pds.example/xrpc/com.atproto.repo.listRecords?repo=did%3Aplc%3Ame&collection=app.bsky.feed.post&limit=100"
        assert get.calls[3].endswith("&cursor=c1") and len(get.calls) == 4

    def test_a_short_embedding_answer_raises(self):
        get = RecordingGet(account_routes([{"records": RECORDS}]))
        with pytest.raises(StyleError, match=r"embedding returned 1 vectors for 2 texts"):
            social_style.build_profile("me.example", get_json=get, embed=lambda t: [[1.0]], embed_model="e")

    def test_a_did_without_a_pds_raises(self):
        get = RecordingGet([("https://public.api.bsky.app", {"did": "did:plc:x"}), ("https://plc.directory", {"service": []})])
        with pytest.raises(StyleError, match=r"did:plc:x names no #atproto_pds service"):
            social_style.resolve_account("x.example", get_json=get)


PROFILE = {"version": 2, "handle": "me.example", "did": "did:plc:me", "built_at": "2026-10-22T00:00:00+00:00",
           "source": {"posts": 9, "bot_posts_excluded": 4}, "embed_model": "emb:1", "rules": ["Rule one.", "Rule two."],
           "stats": {"alt_texts": 3, "chars_p25": 40, "chars_p75": 90, "chars_p90": 120},
           "examples": [{"alt": "far away", "vector": [0.0, 1.0]}, {"alt": "closest", "vector": [1.0, 0.1]},
                        {"alt": "second", "vector": [1.0, 0.5]}]}


class TestUse:
    def test_nearest_examples_rank_by_cosine_not_by_position(self):
        assert social_style.nearest_examples(PROFILE, [1.0, 0.0], k=2) == ["closest", "second"]
        assert social_style.nearest_examples(PROFILE, [0.0, 1.0], k=1) == ["far away"]

    def test_style_for_embeds_the_subject_once(self):
        calls = []

        def embed(texts):
            calls.append(texts)
            return [[1.0, 0.0]]

        assert social_style.style_for(PROFILE, "spider web", embed=embed, k=2) == {"rules": ["Rule one.", "Rule two."],
                                                                                    "examples": ["closest", "second"], "band": [40, 90]}
        assert calls == [["spider web"]]

    def test_profile_round_trip_and_refusals(self, tmp_path):
        path = social_style.save_profile(PROFILE, tmp_path / "s" / "p.json")
        assert social_style.load_profile(path) == PROFILE
        with pytest.raises(StyleError, match=r"no style profile at .*nope\.json: run `bio social-style` first"):
            social_style.load_profile(tmp_path / "nope.json")
        (tmp_path / "old.json").write_text(json.dumps({**PROFILE, "version": 0}))
        with pytest.raises(StyleError, match=r"is version 0, expected 2: run `bio social-style` again"):
            social_style.load_profile(tmp_path / "old.json")

    def test_loci_record_carries_the_structure_without_vectors(self):
        rec = social_style.loci_record(PROFILE, examples=2)
        assert rec["text"] == ("Bluesky alt-text style of me.example (did:plc:me), learned 2026-10-22 from 3 alt texts the owner wrote "
                               "(posts: 9, inat_skeet bot posts excluded: 4). Rules: Rule one. Rule two. Examples: far away | closest")
        assert rec["metadata"]["examples_sample"] == ["far away", "closest"]
        assert (rec["metadata"]["kind"], rec["metadata"]["stats"], rec["metadata"]["rules"]) == (
            "bsky_alt_text_style", {"alt_texts": 3, "chars_p25": 40, "chars_p75": 90, "chars_p90": 120}, ["Rule one.", "Rule two."])
        assert "vector" not in json.dumps(rec)
