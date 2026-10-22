"""social: gallery -> observation index, month selection, drafts with local-vision alt text, approval ledger, Bluesky posting.
Real Pillow JPEGs, real exiftool, real atproto TextBuilder; the stubbed dependencies are iNat (fetch), Ollama (api), the label reader and
the Bluesky client (recording fakes)."""
from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from bio_automation import social
from bio_automation.inat import InatNotFoundError
from bio_automation.inat_stock import embed_label
from bio_automation.social import Ledger, SkipObservation, SocialError, VisionError

TZ = dt.timezone(dt.timedelta(hours=-4))


def jpeg(path: Path, size=(64, 48), color=(10, 90, 10)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path, "JPEG")
    return path


def write_csv(path: Path, rows: list[dict]) -> Path:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def cand(obs, title, day, raw, id_, stock="True"):
    return {"id": id_, "stock_candidate": stock, "day": day, "raw": raw, "obs_id": obs, "title": title}


def observation(obs_id=42, grade="research", iconic="Arachnida", place="Bucks County, PA, USA", geoprivacy=None, name="Araneus thaddeus",
                common="Lattice Orbweaver", observed_on="2024-10-01"):
    return {"observation_id": obs_id, "source": "inaturalist",
            "taxon": {"id": 7, "name": name, "preferred_common_name": common, "common_name": common, "iconic_taxon_name": iconic, "rank": "species"},
            "place_guess": place, "observed_on": observed_on, "geoprivacy": geoprivacy, "license": "cc-by-nc", "quality_grade": grade,
            "keywords_seed": []}


class RecordingTitles:
    def __init__(self, titles):
        self.titles, self.calls = titles, []

    def __call__(self, paths):
        self.calls.append(sorted(p.name for p in paths))
        return {p: self.titles.get(p.name, "") for p in paths}


# ----------------------------------------------------------------------------------------------------------- index

class TestObsIndex:
    def gallery(self, tmp_path):
        g = tmp_path / "gallery"
        for rel in ("2024/2024-10/2024-10-01_DSC_1.jpg", "2025/2025-10/2025-10-02_DSC_2.jpg", "2025/2025-10/2025-10-02_DSC_3.jpg",
                    "2025/2025-10/2025-10-02_DSC_4.jpg", "2025/2025-10/2025-10-02_DSC_5.jpg", "2026/2026-03/2026-03-01_DSC_6.jpg"):
            jpeg(g / rel)
        return g

    def test_each_file_resolves_from_its_evidence_or_says_why_not(self, tmp_path):
        g = self.gallery(tmp_path)
        audit = write_csv(tmp_path / "audit.csv", [{"file": "2024-10-01_DSC_1.jpg", "obs_id": "100"}])
        old = write_csv(tmp_path / "old.csv", [cand("201", "Wrong Bee", "2025-10-02", "2025/2025-10-02/DSC_2.NEF", "1"),
                                               cand("300", "Moth", "2025-10-02", "2025/2025-10-02/DSC_4.NEF", "3"),
                                               cand("301", "Moth", "2025-10-02", "2025/2025-10-02/DSC_4.NEF", "4"),
                                               cand("400", "Fly", "2025-10-02", "2025/2025-10-02/DSC_5.NEF", "5")])
        new = write_csv(tmp_path / "new.csv", [cand("200", "Right Bee", "2025-10-02", "2025/2025-10-02/DSC_2.NEF", "1"),
                                               cand("999", "Ant", "2025-10-02", "2025/2025-10-02/DSC_3.NEF", "2", stock="False")])
        titles = RecordingTitles({"2025-10-02_DSC_2.jpg": "Right Bee", "2025-10-02_DSC_4.jpg": "Moth", "2025-10-02_DSC_5.jpg": "Spider"})

        index = social.build_obs_index(g, [audit], [old, new], titles=titles)

        assert index == {
            "files": {"2024/2024-10/2024-10-01_DSC_1.jpg": 100, "2025/2025-10/2025-10-02_DSC_2.jpg": 200},
            "unresolved": {"2025/2025-10/2025-10-02_DSC_3.jpg": "no csv row names this file",
                           "2025/2025-10/2025-10-02_DSC_4.jpg": "ambiguous: observations 300, 301",
                           "2025/2025-10/2025-10-02_DSC_5.jpg": "every csv row's title differs from the one embedded in the file",
                           "2026/2026-03/2026-03-01_DSC_6.jpg": "no csv row names this file"}}
        # titles are read only for files that have candidate-csv evidence
        assert titles.calls == [["2025-10-02_DSC_2.jpg", "2025-10-02_DSC_4.jpg", "2025-10-02_DSC_5.jpg"]]

    def test_enhanced_nr_raw_maps_through_render_name(self, tmp_path):
        g = tmp_path / "gallery"
        jpeg(g / "2025/2025-04/2025-04-25_DSC_3770_14081.jpg")
        c = write_csv(tmp_path / "c.csv", [cand("55", "Wasp", "2025-04-25", "2025/2025-04-25/DSC_3770 (2025_06_03 19_23_22 UTC)-Enhanced-NR.dng", "14081")])

        index = social.build_obs_index(g, [], [c], titles=RecordingTitles({"2025-04-25_DSC_3770_14081.jpg": "Wasp"}))

        assert index == {"files": {"2025/2025-04/2025-04-25_DSC_3770_14081.jpg": 55}, "unresolved": {}}

    def test_read_titles_reads_real_embedded_titles_with_accents(self, tmp_path):
        a, b = jpeg(tmp_path / "a dir" / "café.jpg"), jpeg(tmp_path / "b.jpg")
        embed_label(a, {"title": "Mélange of moths", "caption": "c", "keywords": ["k"]})

        assert social.read_titles([a, b]) == {a: "Mélange of moths", b: ""}

    def test_index_round_trips_and_a_missing_index_names_the_command(self, tmp_path):
        idx = {"files": {"2024/2024-10/x.jpg": 1}, "unresolved": {}}
        social.write_index(idx, tmp_path / "state" / "i.json")
        assert social.load_index(tmp_path / "state" / "i.json") == idx
        with pytest.raises(SocialError, match=r"run `bio social-index` first"):
            social.load_index(tmp_path / "nope.json")


# ----------------------------------------------------------------------------------------------------------- selection

class TestSelection:
    INDEX = {"files": {"2024/2024-10/a.jpg": 1, "2024/2024-10/b.jpg": 1, "2024/2024-10/c.jpg": 2, "2025/2025-10/d.jpg": 3,
                       "2025/2025-10/e.jpg": 4, "2026/2026-10/f.jpg": 5, "2025/2025-11/g.jpg": 6, "2025/2025-01/h.jpg": 7}}

    def test_month_takes_every_year_and_no_other_month(self):
        assert social.month_observations(self.INDEX, 10) == {1: ["2024/2024-10/a.jpg", "2024/2024-10/b.jpg"], 2: ["2024/2024-10/c.jpg"],
                                                             3: ["2025/2025-10/d.jpg"], 4: ["2025/2025-10/e.jpg"], 5: ["2026/2026-10/f.jpg"]}
        assert social.month_observations(self.INDEX, 1) == {7: ["2025/2025-01/h.jpg"]}

    def test_years_take_turns_newest_first_and_excluded_are_left_out(self):
        cands = social.month_observations(self.INDEX, 10)
        order = social.order_candidates(cands, exclude={2}, seed="2026-10-21")
        year = {1: 2024, 3: 2025, 4: 2025, 5: 2026}
        assert [year[o] for o in order] == [2026, 2025, 2024, 2025]
        assert sorted(order) == [1, 3, 4, 5]
        assert social.order_candidates(cands, exclude={2}, seed="2026-10-21") == order


# ----------------------------------------------------------------------------------------------------------- images

class TestPrepareImage:
    def test_smooth_image_fits_at_the_first_quality_and_the_size_cap(self, tmp_path):
        src = tmp_path / "s.jpg"
        grad = np.tile(np.linspace(0, 255, 3000, dtype=np.uint8), (2000, 1))
        Image.fromarray(np.stack([grad, grad, grad], -1)).save(src, "JPEG", quality=95)

        info = social.prepare_image(src, tmp_path / "o.jpg", max_bytes=1_000_000, max_edge=1200)

        assert (info["width"], info["height"], info["quality"]) == (1200, 800, 90)
        assert info["bytes"] == (tmp_path / "o.jpg").stat().st_size

    def test_noisy_image_is_squeezed_under_the_byte_limit(self, tmp_path):
        src = tmp_path / "n.jpg"
        Image.fromarray(np.random.default_rng(0).integers(0, 255, (2000, 3000, 3), dtype=np.uint8)).save(src, "JPEG", quality=95)

        info = social.prepare_image(src, tmp_path / "o.jpg", max_bytes=100_000, max_edge=1200)

        # 1200 px does not fit even at quality 60, so the size steps down (x0.85: 1020, 867) and the quality restarts at 90
        assert (info["width"], info["height"], info["quality"]) == (867, 578, 70)
        assert info["bytes"] <= 100_000
        assert info["bytes"] == (tmp_path / "o.jpg").stat().st_size

    def test_orientation_is_applied_and_no_metadata_survives(self, tmp_path):
        src = tmp_path / "e.jpg"
        im = Image.new("RGB", (300, 200), (200, 30, 30))
        exif = im.getexif()
        exif[0x0112] = 6                                   # rotate 90 CW on display
        exif[0x8825] = {1: "N", 2: (40.0, 1.0, 0.0)}       # GPS IFD
        im.save(src, "JPEG", exif=exif.tobytes())

        info = social.prepare_image(src, tmp_path / "o.jpg")

        assert (info["width"], info["height"]) == (200, 300)
        out = (tmp_path / "o.jpg").read_bytes()
        assert b"Exif" not in out
        assert dict(Image.open(tmp_path / "o.jpg").getexif()) == {}

    def test_impossible_limit_raises(self, tmp_path):
        src = jpeg(tmp_path / "x.jpg", size=(400, 300))
        with pytest.raises(SocialError, match=r"no JPEG of at least 256 px fits 10 bytes"):
            social.prepare_image(src, tmp_path / "o.jpg", max_bytes=10, max_edge=400)


# ----------------------------------------------------------------------------------------------------------- vision

class FakeOllama:
    def __init__(self, tags=(), content="A spider rests at the centre of its web.", size=100, size_vram=100, model="vis:1"):
        self.tags, self.content, self.size, self.size_vram, self.model = list(tags), content, size, size_vram, model
        self.calls = []

    def __call__(self, path, body=None, **kw):
        self.calls.append((path, body))
        if path == "/api/tags":
            return {"models": [{"name": n} for n in self.tags]}
        if path == "/api/chat":
            return {"message": {"role": "assistant", "content": self.content}}
        if path == "/api/ps":
            return {"models": [{"name": self.model, "size": self.size, "size_vram": self.size_vram}]}
        raise AssertionError(path)


POOL = """
[[models.pool]]
name = "big-vision:latest"
roles = ["vision"]
rank = 1

[[models.pool]]
name = "small-vision:latest"
roles = ["vision"]
rank = 2

[[models.pool]]
name = "chat:latest"
roles = ["gen"]
rank = 0
"""


class TestVisionModel:
    def test_env_wins(self, monkeypatch, tmp_path):
        monkeypatch.setenv("BIO_VISION_MODEL", "mine:7b")
        api = FakeOllama()
        assert social.vision_model(api=api, pool_file=tmp_path / "none.toml") == "mine:7b"
        assert api.calls == []

    def test_best_ranked_installed_vision_model_of_the_pool(self, monkeypatch, tmp_path):
        monkeypatch.delenv("BIO_VISION_MODEL", raising=False)
        (tmp_path / "p.toml").write_text(POOL, encoding="utf-8")
        assert social.vision_model(api=FakeOllama(tags=["big-vision:latest", "small-vision:latest", "chat:latest"]), pool_file=tmp_path / "p.toml") == "big-vision:latest"
        assert social.vision_model(api=FakeOllama(tags=["small-vision:latest", "chat:latest"]), pool_file=tmp_path / "p.toml") == "small-vision:latest"

    def test_no_installed_vision_model_raises(self, monkeypatch, tmp_path):
        monkeypatch.delenv("BIO_VISION_MODEL", raising=False)
        (tmp_path / "p.toml").write_text(POOL, encoding="utf-8")
        with pytest.raises(VisionError, match=r"no installed model has the 'vision' role"):
            social.vision_model(api=FakeOllama(tags=["chat:latest"]), pool_file=tmp_path / "p.toml")


CONTEXT = {"common_name": "Lattice Orbweaver", "scientific_name": "Araneus thaddeus", "rank": "species", "grade": "Research Grade",
           "observed_on": "October 1, 2024", "place": "Bucks County, PA, USA", "title": "Lattice Orbweaver on web",
           "caption": "Lattice Orbweaver on web.", "keywords": ["web", "autumn"]}


class TestDescribe:
    def test_alt_text_comes_from_the_model_with_every_fact_in_the_prompt(self, tmp_path):
        img = jpeg(tmp_path / "1.jpg", size=(2000, 1000))
        api = FakeOllama(content="  A spider rests\n at the centre of its web. ")

        assert social.describe(img, CONTEXT, model="vis:1", api=api) == "A spider rests at the centre of its web."
        path, body = api.calls[0]
        assert path == "/api/chat" and body["model"] == "vis:1" and body["think"] is False and body["stream"] is False
        prompt = body["messages"][0]["content"]
        for line in ("- Common name: Lattice Orbweaver", "- Scientific name: Araneus thaddeus", "- Date: October 1, 2024",
                     "- Place: Bucks County, PA, USA", "- Stock title: Lattice Orbweaver on web", "- Stock keywords: web, autumn"):
            assert line in prompt.splitlines()
        sent = Image.open(__import__("io").BytesIO(__import__("base64").b64decode(body["messages"][0]["images"][0])))
        assert sent.size == (1024, 512)

    def test_model_partly_in_system_memory_is_refused(self, tmp_path):
        img = jpeg(tmp_path / "1.jpg")
        with pytest.raises(VisionError, match=r"vis:1 is not fully on the GPU"):
            social.describe(img, CONTEXT, model="vis:1", api=FakeOllama(size=100, size_vram=60))

    def test_empty_answer_is_refused(self, tmp_path):
        img = jpeg(tmp_path / "1.jpg")
        with pytest.raises(VisionError, match=r"returned no alt text"):
            social.describe(img, CONTEXT, model="vis:1", api=FakeOllama(content="  "))

    def test_overlong_answer_is_capped_at_the_bluesky_limit(self, tmp_path):
        img = jpeg(tmp_path / "1.jpg")
        assert social.describe(img, CONTEXT, model="vis:1", api=FakeOllama(content="a" * 2500)) == "A" + "a" * 1999

    def test_cleanup_drops_the_opener_and_sentences_that_restate_date_or_place(self, tmp_path):
        img = jpeg(tmp_path / "1.jpg")
        answer = ("The image shows a lattice orbweaver at the hub of its web. Its abdomen is orange with a pale lattice. "
                  "The photo was taken on October 1, 2024, in Bucks County, PA, USA. The background is dark green foliage.")
        assert social.describe(img, CONTEXT, model="vis:1", api=FakeOllama(content=answer)) == (
            "A lattice orbweaver at the hub of its web. Its abdomen is orange with a pale lattice. The background is dark green foliage.")

    def test_cleanup_keeps_an_answer_that_would_otherwise_be_empty_and_leaves_clean_text_alone(self):
        assert social.clean_alt("Seen in Bucks County, PA, USA.", CONTEXT) == "Seen in Bucks County, PA, USA."
        assert social.clean_alt("A spider on a web.  Light from the left.", CONTEXT) == "A spider on a web. Light from the left."
        assert social.clean_alt("This photo is soft at the edges.", {}) == "This photo is soft at the edges."
        assert social.clean_alt("This photo depicts a moth.", {}) == "A moth."


# ----------------------------------------------------------------------------------------------------------- post text

class TestText:
    def test_full_post(self):
        segs = social.text_segments(observation())
        assert social.segments_text(segs) == ("Lattice Orbweaver (Araneus thaddeus)\nResearch Grade\nOctober 1, 2024 · Bucks County, PA, USA\n"
                                              "https://www.inaturalist.org/observations/42\nAlt text: local AI model, reviewed.\n"
                                              "#macrophotography #nature #inaturalist #Arachnida #arachnids")
        assert [s for s in segs if s[0] == "link"] == [["link", "https://www.inaturalist.org/observations/42", "https://www.inaturalist.org/observations/42"]]

    def test_lowercase_common_name_is_capitalised_for_display(self):
        text = social.segments_text(social.text_segments(observation(name="Verbena hastata", common="blue vervain", iconic="Plantae")))
        assert text.startswith("Blue vervain (Verbena hastata)\nResearch Grade\n")
        assert social.display_name({"name": "Verbena hastata"}) == "Verbena hastata"

    def test_private_location_is_left_out_and_shown_when_obscured(self):
        assert "Bucks County" not in social.segments_text(social.text_segments(observation(geoprivacy="private")))
        assert "October 1, 2024 · Bucks County, PA, USA\n" in social.segments_text(social.text_segments(observation(geoprivacy="obscured")))

    def test_unknown_iconic_taxon_gets_the_common_tags_and_its_name(self):
        text = social.segments_text(social.text_segments(observation(iconic="Chromista")))
        assert text.endswith("\n#macrophotography #nature #inaturalist #Chromista")
        assert social.segments_text(social.text_segments(observation(iconic=None))).endswith("\n#macrophotography #nature #inaturalist")

    def test_tags_are_dropped_from_the_end_to_fit_300(self):
        # with a 69-character place the post is exactly 300 characters with every Fungi tag ("fungi" is #Fungi already); one more drops the last
        exact = social.segments_text(social.text_segments(observation(iconic="Fungi", place="P" * 69)))
        assert len(exact) == 300
        assert exact.endswith("\n#macrophotography #nature #inaturalist #Fungi #mycology #mushrooms #fungifriends")
        over = social.segments_text(social.text_segments(observation(iconic="Fungi", place="P" * 70)))
        assert len(over) == 287
        assert over.endswith("\n#macrophotography #nature #inaturalist #Fungi #mycology #mushrooms")

    def test_too_long_without_tags_raises(self):
        with pytest.raises(SocialError, match=r"observation 42: post text is \d+ characters even without hashtags"):
            social.text_segments(observation(place="P" * 400))

    def test_real_text_builder_makes_link_and_tag_facets(self):
        tb = social.text_builder(social.text_segments(observation()))
        facets = tb.build_facets()
        kinds = [f.features[0].py_type for f in facets]
        assert kinds == ["app.bsky.richtext.facet#link"] + ["app.bsky.richtext.facet#tag"] * 5
        assert tb.build_text() == social.segments_text(social.text_segments(observation()))


# ----------------------------------------------------------------------------------------------------------- drafting

class RecordingFetch:
    def __init__(self, obs_by_id):
        self.obs_by_id, self.calls = obs_by_id, []

    def __call__(self, obs_id):
        self.calls.append(obs_id)
        v = self.obs_by_id[obs_id]
        if isinstance(v, Exception):
            raise v
        return v


class RecordingDescribe:
    def __init__(self):
        self.calls = []

    def __call__(self, image, context, *, model):
        self.calls.append((Path(image).name, context["title"], model))
        return f"alt for {context['title']}"


def labels(path):
    return {"title": f"title {Path(path).stem}", "caption": "cap", "keywords": ["k1", "k2"]}


def no_model():
    raise AssertionError("pick_model must not be called when a model is given or nothing is needed")


class TestDraft:
    def setup_gallery(self, tmp_path, n_files=5):
        g = tmp_path / "gallery"
        rels = [f"2024/2024-10/2024-10-01_DSC_{i}.jpg" for i in range(n_files)]
        for r in rels:
            jpeg(g / r, size=(300, 200))
        return g, rels

    def test_make_draft_writes_up_to_four_images_with_alt_text_and_the_post(self, tmp_path):
        g, rels = self.setup_gallery(tmp_path)
        desc = RecordingDescribe()
        draft = social.make_draft(42, rels, month="2026-10", gallery_dir=g, drafts_dir=tmp_path / "drafts", model="vis:1",
                                  fetch=RecordingFetch({42: observation()}), read_label=labels, describe_fn=desc,
                                  now=dt.datetime(2026, 10, 21, 9, 0, tzinfo=TZ))
        assert draft["draft_id"] == "2026-10-obs42"
        assert [(i["file"], i["source"], i["alt"], i["width"], i["height"]) for i in draft["images"]] == [
            (f"{n + 1}.jpg", rels[n], f"alt for title 2024-10-01_DSC_{n}", 300, 200) for n in range(4)]
        assert desc.calls == [(f"{n + 1}.jpg", f"title 2024-10-01_DSC_{n}", "vis:1") for n in range(4)]
        on_disk = json.loads((tmp_path / "drafts" / "2026-10-obs42" / "draft.json").read_text(encoding="utf-8"))
        assert on_disk == draft
        assert sorted(p.name for p in (tmp_path / "drafts" / "2026-10-obs42").iterdir()) == ["1.jpg", "2.jpg", "3.jpg", "4.jpg", "draft.json"]

    @pytest.mark.parametrize("value,reason", [
        (observation(grade="casual"), "observation 42 is casual"),
        (InatNotFoundError("gone"), "observation 42 no longer exists on iNat"),
        ({**observation(), "taxon": {}}, "observation 42 has no taxon")])
    def test_unpostable_observations_are_skipped_before_any_image_work(self, tmp_path, value, reason):
        g, rels = self.setup_gallery(tmp_path, 1)
        desc = RecordingDescribe()
        with pytest.raises(SkipObservation, match=reason):
            social.make_draft(42, rels, month="2026-10", gallery_dir=g, drafts_dir=tmp_path / "d", model="m",
                              fetch=RecordingFetch({42: value}), read_label=labels, describe_fn=desc)
        assert desc.calls == []
        assert not (tmp_path / "d").exists()

    def test_draft_month_tops_up_to_keep_pending_skipping_casual_and_seen(self, tmp_path):
        g = tmp_path / "gallery"
        index = {"files": {}}
        for obs, year in ((1, 2024), (2, 2025), (3, 2026), (4, 2025), (5, 2024)):
            rel = f"{year}/{year}-10/{year}-10-0{obs}_DSC_{obs}.jpg"
            jpeg(g / rel, size=(300, 200))
            index["files"][rel] = obs
        index["files"]["2025/2025-11/2025-11-01_DSC_9.jpg"] = 9
        ledger = Ledger(tmp_path / "ledger.jsonl")
        ledger.append("drafted", "2026-10-obs4", obs_id=4, month="2026-10")      # pending already
        ledger.append("drafted", "2025-10-obs5", obs_id=5, month="2025-10")
        ledger.append("posted", "2025-10-obs5", obs_id=5, month="2025-10")       # posted last year: never again
        fetch = RecordingFetch({1: observation(1), 2: observation(2, grade="casual"), 3: observation(3)})

        report = social.draft_month(index, ledger, today=dt.date(2026, 10, 21), gallery_dir=g, drafts_dir=tmp_path / "drafts",
                                    keep_pending=3, model="vis:1", fetch=fetch, read_label=labels, describe_fn=RecordingDescribe(),
                                    pick_model=no_model)

        assert report == {"month": "2026-10", "pending": 1, "drafted": ["2026-10-obs3", "2026-10-obs1"],
                          "skipped": {2: "observation 2 is casual"}}
        assert fetch.calls == [3, 2, 1]
        assert {k: v["state"] for k, v in ledger.drafts().items()} == {"2026-10-obs4": "drafted", "2025-10-obs5": "posted",
                                                                      "2026-10-obs3": "drafted", "2026-10-obs1": "drafted"}

    def test_an_expired_draft_of_an_earlier_month_comes_back_but_rejected_posted_and_posting_never_do(self, tmp_path):
        g = tmp_path / "gallery"
        index = {"files": {}}
        for obs in (1, 2, 3, 4, 5):
            rel = f"2024/2024-10/2024-10-0{obs}_DSC_{obs}.jpg"
            jpeg(g / rel, size=(300, 200))
            index["files"][rel] = obs
        ledger = Ledger(tmp_path / "ledger.jsonl")
        for obs, events in ((1, ["drafted"]), (2, ["drafted", "approved"]), (3, ["drafted", "rejected"]),
                            (4, ["drafted", "approved", "posted"]), (5, ["drafted", "approved", "posting"])):
            for e in events:
                ledger.append(e, f"2025-10-obs{obs}", obs_id=obs, month="2025-10")
        fetch = RecordingFetch({o: observation(o) for o in (1, 2, 3, 4, 5)})

        report = social.draft_month(index, ledger, today=dt.date(2026, 10, 1), gallery_dir=g, drafts_dir=tmp_path / "drafts",
                                    keep_pending=10, model="vis:1", fetch=fetch, read_label=labels, describe_fn=RecordingDescribe(),
                                    pick_model=no_model)

        assert sorted(report["drafted"]) == ["2026-10-obs1", "2026-10-obs2"]
        assert sorted(fetch.calls) == [1, 2]

    def test_draft_month_does_nothing_when_enough_are_pending(self, tmp_path):
        ledger = Ledger(tmp_path / "ledger.jsonl")
        ledger.append("approved", "2026-10-obs4", obs_id=4, month="2026-10")
        fetch = RecordingFetch({})
        report = social.draft_month({"files": {"2024/2024-10/a.jpg": 1}}, ledger, today=dt.date(2026, 10, 21), gallery_dir=tmp_path,
                                    drafts_dir=tmp_path / "d", keep_pending=1, fetch=fetch, pick_model=no_model)
        assert report == {"month": "2026-10", "pending": 1, "drafted": [], "skipped": {}}
        assert fetch.calls == []


# ----------------------------------------------------------------------------------------------------------- approval + posting

class FakeResp:
    uri = "at://did:plc:me/app.bsky.feed.post/3abc"
    cid = "bafyabc"


class RecordingClient:
    def __init__(self, fail=None):
        self.calls, self.fail = [], fail

    def send_images(self, **kw):
        self.calls.append(kw)
        if self.fail:
            raise self.fail
        return FakeResp()


def write_draft(drafts_dir: Path, draft_id: str, obs_id: int, month: str) -> Path:
    d = drafts_dir / draft_id
    jpeg(d / "1.jpg", size=(300, 200))
    jpeg(d / "2.jpg", size=(200, 300), color=(1, 2, 3))
    draft = {"draft_id": draft_id, "month": month, "obs_id": obs_id, "text": social.text_segments(observation(obs_id)),
             "images": [{"file": "1.jpg", "alt": "first alt", "width": 300, "height": 200}, {"file": "2.jpg", "alt": "second alt", "width": 200, "height": 300}]}
    (d / "draft.json").write_text(json.dumps(draft), encoding="utf-8")
    return d


NOW = dt.datetime(2026, 10, 21, 9, 0, tzinfo=TZ)


class TestPosting:
    def ready(self, tmp_path, n=2):
        ledger, drafts = Ledger(tmp_path / "l.jsonl"), tmp_path / "drafts"
        for i in range(1, n + 1):
            write_draft(drafts, f"2026-10-obs{i}", i, "2026-10")
            ledger.append("drafted", f"2026-10-obs{i}", obs_id=i, month="2026-10", now=NOW - dt.timedelta(days=2, minutes=-i))
        return ledger, drafts

    def test_dry_run_shows_the_oldest_approved_draft_and_records_nothing(self, tmp_path):
        ledger, drafts = self.ready(tmp_path)
        social.approve(ledger, drafts, "2026-10-obs2")
        social.approve(ledger, drafts, "2026-10-obs1")
        before = ledger.records()

        def no_client():
            raise AssertionError("dry-run must not log in")

        result = social.post_next(ledger, drafts, now=NOW, client_factory=no_client)

        assert result["status"] == "dry-run" and result["draft_id"] == "2026-10-obs2"
        assert result["alts"] == ["first alt", "second alt"]
        assert result["image_bytes"] == [(drafts / "2026-10-obs2" / f).stat().st_size for f in ("1.jpg", "2.jpg")]
        assert ledger.records() == before

    def test_live_post_sends_text_images_alts_ratios_and_records_the_uri(self, tmp_path):
        ledger, drafts = self.ready(tmp_path, 1)
        social.approve(ledger, drafts, "2026-10-obs1")
        client = RecordingClient()

        result = social.post_next(ledger, drafts, now=NOW, live=True, client_factory=lambda: client)

        assert result["status"] == "posted" and result["uri"] == FakeResp.uri
        (call,) = client.calls
        assert call["text"].build_text() == social.segments_text(social.text_segments(observation(1)))
        assert call["images"] == [(drafts / "2026-10-obs1" / f).read_bytes() for f in ("1.jpg", "2.jpg")]
        assert call["image_alts"] == ["first alt", "second alt"] and call["langs"] == ["en"]
        assert [(r.width, r.height) for r in call["image_aspect_ratios"]] == [(300, 200), (200, 300)]
        assert [(r["event"], r.get("uri")) for r in ledger.records()][-2:] == [("posting", None), ("posted", FakeResp.uri)]

    def test_one_post_a_day(self, tmp_path):
        ledger, drafts = self.ready(tmp_path)
        social.approve(ledger, drafts, "2026-10-obs1")
        social.approve(ledger, drafts, "2026-10-obs2")
        client = RecordingClient()
        social.post_next(ledger, drafts, now=NOW, live=True, client_factory=lambda: client)

        again = social.post_next(ledger, drafts, now=NOW + dt.timedelta(hours=3), live=True, client_factory=lambda: client)
        assert again == {"status": "done for today", "posted_today": ["2026-10-obs1"]}
        assert len(client.calls) == 1
        tomorrow = social.post_next(ledger, drafts, now=NOW + dt.timedelta(days=1), live=True, client_factory=lambda: client)
        assert tomorrow["status"] == "posted" and tomorrow["draft_id"] == "2026-10-obs2"

    def test_only_this_months_approved_drafts_are_posted(self, tmp_path):
        ledger, drafts = self.ready(tmp_path, 1)
        social.approve(ledger, drafts, "2026-10-obs1")
        assert social.post_next(ledger, drafts, now=NOW + dt.timedelta(days=15)) == {"status": "nothing approved", "month": "2026-11"}
        assert social.post_next(ledger, drafts, now=NOW)["status"] == "dry-run"

    def test_unapproved_drafts_are_never_posted(self, tmp_path):
        ledger, drafts = self.ready(tmp_path)
        social.approve(ledger, drafts, "2026-10-obs1")
        social.reject(ledger, "2026-10-obs1", "blurry")
        assert social.post_next(ledger, drafts, now=NOW) == {"status": "nothing approved", "month": "2026-10"}

    def test_a_draft_edited_after_approval_is_refused_until_approved_again(self, tmp_path):
        ledger, drafts = self.ready(tmp_path, 1)
        social.approve(ledger, drafts, "2026-10-obs1")
        d = json.loads((drafts / "2026-10-obs1" / "draft.json").read_text())
        d["images"][0]["alt"] = "edited alt"
        (drafts / "2026-10-obs1" / "draft.json").write_text(json.dumps(d))
        with pytest.raises(SocialError, match=r"draft 2026-10-obs1 changed after it was approved"):
            social.post_next(ledger, drafts, now=NOW)
        social.approve(ledger, drafts, "2026-10-obs1")
        assert social.post_next(ledger, drafts, now=NOW)["alts"] == ["edited alt", "second alt"]

    def test_failed_login_is_a_clean_failure_that_can_be_approved_again(self, tmp_path):
        ledger, drafts = self.ready(tmp_path, 1)
        social.approve(ledger, drafts, "2026-10-obs1")

        def bad_login():
            raise RuntimeError("bad app password")

        with pytest.raises(RuntimeError, match="bad app password"):
            social.post_next(ledger, drafts, now=NOW, live=True, client_factory=bad_login)
        assert ledger.drafts()["2026-10-obs1"]["state"] == "failed"
        assert ledger.records()[-1]["error"] == "login: RuntimeError: bad app password"
        social.approve(ledger, drafts, "2026-10-obs1")
        client = RecordingClient()
        assert social.post_next(ledger, drafts, now=NOW, live=True, client_factory=lambda: client)["status"] == "posted"

    def test_a_send_that_raises_blocks_posting_until_resolved(self, tmp_path):
        ledger, drafts = self.ready(tmp_path)
        social.approve(ledger, drafts, "2026-10-obs1")
        social.approve(ledger, drafts, "2026-10-obs2")
        with pytest.raises(SocialError, match=r"posting 2026-10-obs1 failed with an unknown outcome \(ConnectionError: reset\)"):
            social.post_next(ledger, drafts, now=NOW, live=True, client_factory=lambda: RecordingClient(fail=ConnectionError("reset")))
        assert ledger.drafts()["2026-10-obs1"]["state"] == "posting"
        client = RecordingClient()
        with pytest.raises(SocialError, match=r"outcome unknown for 2026-10-obs1"):
            social.post_next(ledger, drafts, now=NOW + dt.timedelta(days=1), live=True, client_factory=lambda: client)
        assert client.calls == []

        social.resolve(ledger, "2026-10-obs1", posted_uri="at://did:plc:me/app.bsky.feed.post/3zzz")
        assert ledger.drafts()["2026-10-obs1"]["state"] == "posted"
        assert social.post_next(ledger, drafts, now=NOW + dt.timedelta(days=1), live=True, client_factory=lambda: client)["draft_id"] == "2026-10-obs2"

    def test_resolve_not_posted_makes_it_approvable_again(self, tmp_path):
        ledger, drafts = self.ready(tmp_path, 1)
        social.approve(ledger, drafts, "2026-10-obs1")
        with pytest.raises(SocialError, match="unknown outcome"):
            social.post_next(ledger, drafts, now=NOW, live=True, client_factory=lambda: RecordingClient(fail=TimeoutError("t")))
        social.resolve(ledger, "2026-10-obs1", posted_uri=None)
        assert ledger.drafts()["2026-10-obs1"]["state"] == "failed"
        social.approve(ledger, drafts, "2026-10-obs1")
        assert ledger.drafts()["2026-10-obs1"]["state"] == "approved"

    def test_state_rules(self, tmp_path):
        ledger, drafts = self.ready(tmp_path, 1)
        with pytest.raises(SocialError, match=r"no draft 'nope'"):
            social.approve(ledger, drafts, "nope")
        with pytest.raises(SocialError, match=r"draft 2026-10-obs1 is not awaiting an outcome \(state drafted\)"):
            social.resolve(ledger, "2026-10-obs1", posted_uri=None)
        social.reject(ledger, "2026-10-obs1")
        with pytest.raises(SocialError, match=r"draft 2026-10-obs1 is rejected; only a drafted or failed draft can be approved"):
            social.approve(ledger, drafts, "2026-10-obs1")
        with pytest.raises(SocialError, match=r"unknown ledger event 'bogus'"):
            ledger.append("bogus", "x")

    def test_missing_credentials_name_the_variables(self, monkeypatch):
        monkeypatch.delenv("BSKY_HANDLE", raising=False)
        monkeypatch.setenv("BSKY_APP_PASSWORD", "x")
        with pytest.raises(SocialError, match=r"set BSKY_HANDLE and BSKY_APP_PASSWORD"):
            social.bluesky_client()


# ----------------------------------------------------------------------------------------------------------- CLI

class TestCli:
    def test_approve_through_the_cli_and_a_bad_id_is_a_one_line_error(self, tmp_path, monkeypatch, capsys):
        from bio_automation import cli, settings

        monkeypatch.setattr(settings, "SOCIAL_LEDGER", tmp_path / "l.jsonl")
        monkeypatch.setattr(settings, "SOCIAL_DIR", tmp_path / "social")
        write_draft(tmp_path / "social" / "drafts", "2026-10-obs1", 1, "2026-10")
        Ledger().append("drafted", "2026-10-obs1", obs_id=1, month="2026-10")

        monkeypatch.setattr("sys.argv", ["bio", "social-approve", "2026-10-obs1"])
        assert cli.main() == 0
        assert Ledger().drafts()["2026-10-obs1"]["state"] == "approved"

        monkeypatch.setattr("sys.argv", ["bio", "social-approve", "nope"])
        assert cli.main() == 1
        assert capsys.readouterr().err == f"bio social-approve: no draft 'nope' in {tmp_path / 'l.jsonl'}\n"
