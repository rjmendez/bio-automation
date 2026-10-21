"""stock_select: Stock candidate measurement, tagging and publishing.

Image maths is tested on synthetic arrays whose answers are computable by hand (a 0/255 checkerboard has a Laplacian of +-1020 everywhere, so a
tile of it has variance 1020**2 = 1040400). Raw decode / exiftool / the Stock renderer are the dependencies and are stubbed where a test is about
the selection logic; one test per raw helper runs against the committed NEF fixtures."""
from __future__ import annotations

import csv
import logging
import os
from pathlib import Path

import numpy as np
import pytest

from bio_automation import cli, pipeline, stock_select
from bio_automation.io import read_json, write_json
from bio_automation.models import Asset, Observation
from bio_automation.paths import metadata_dir, source_raw_dir
from bio_automation.render import ExportBoundsError
from bio_automation.stock_select import FrameScore, StockFloorMissing

FIXTURES = Path(__file__).parent / "fixtures"
FLOOR = {"n": 100, "p10": 18.4, "p25": 49.7, "median": 114.0, "luma_p10": 35.0, "clip_p95": 0.02}
CHECKER_VAR = 1020.0 ** 2


def checker(h, w, cols=None):
    """0/255 checkerboard RGB image; cols limits the checkerboard to image columns [0, cols), the rest is flat mid-grey."""
    yy, xx = np.indices((h, w))
    v = np.where((yy + xx) % 2 == 0, 255, 0).astype(np.uint8)
    if cols is not None:
        v[:, cols:] = 128
    return np.stack([v, v, v], axis=2)


class TestTileSharpness:
    def test_checkerboard_tiles_have_the_hand_computed_variance(self):
        s = stock_select.tile_sharpness(checker(66, 66))
        assert s.sharp == pytest.approx(CHECKER_VAR, rel=1e-5)
        assert s.clip == pytest.approx(0.5, rel=1e-6)           # half the pixels are 255

    def test_flat_image_has_zero_sharpness(self):
        s = stock_select.tile_sharpness(np.full((66, 66, 3), 128, np.uint8))
        assert (s.sharp, s.clip) == (0.0, 0.0)

    def test_sharp_quarter_of_the_frame_counts_but_a_single_sharp_tile_does_not(self):
        quarter = stock_select.tile_sharpness(checker(66, 66, cols=41))           # 4 of 8 tile columns -> 32 of 64 tiles
        assert quarter.sharp == pytest.approx(CHECKER_VAR, rel=1e-5)
        one_tile = np.full((66, 66, 3), 128, np.uint8)
        one_tile[26:34, 26:34] = checker(8, 8)[:, :, :]                          # 1 of 64 tiles: below the 90th percentile
        assert stock_select.tile_sharpness(one_tile).sharp == 0.0

    def test_clip_boundary_is_250_inclusive(self):
        img = np.full((66, 66, 3), 249, np.uint8)
        assert stock_select.tile_sharpness(img).clip == 0.0
        img[:, :33] = 250
        assert stock_select.tile_sharpness(img).clip == 0.5

    def test_clip_uses_the_brightest_channel(self):
        img = np.zeros((66, 66, 3), np.uint8)
        img[..., 1] = 255
        assert stock_select.tile_sharpness(img).clip == 1.0

    def test_wrong_shape_is_refused(self):
        with pytest.raises(ValueError, match=r"rgb must be HxWx3, got shape \(66, 66\)"):
            stock_select.tile_sharpness(np.zeros((66, 66), np.uint8))

    def test_too_small_image_is_refused_and_the_minimum_size_works(self):
        with pytest.raises(ValueError, match="image 23x23 is too small for an 8x8 tile grid"):
            stock_select.tile_sharpness(np.zeros((23, 23, 3), np.uint8))
        assert stock_select.tile_sharpness(np.zeros((24, 24, 3), np.uint8)).sharp == 0.0


class TestMedianLuma:
    def test_uniform_colour_uses_rec601_weights(self):
        img = np.zeros((4, 4, 3), np.uint8)
        img[...] = (100, 0, 0)
        assert stock_select.median_luma(img) == pytest.approx(29.9, rel=1e-5)
        img[...] = (0, 100, 0)
        assert stock_select.median_luma(img) == pytest.approx(58.7, rel=1e-5)

    def test_median_not_mean(self):
        img = np.zeros((10, 10, 3), np.uint8)
        img[:, :4] = 255                                                  # 40% white: mean 102, median 0
        assert stock_select.median_luma(img) == 0.0
        img[:, :6] = 255                                                  # 60% white: median flips to white
        assert stock_select.median_luma(img) == pytest.approx(255.0, rel=1e-5)

    def test_wrong_shape_refused(self):
        with pytest.raises(ValueError, match=r"rgb must be HxWx3, got shape \(4, 4\)"):
            stock_select.median_luma(np.zeros((4, 4), np.uint8))


class TestFloor:
    def test_percentiles_of_1_to_100(self):
        f = stock_select.calibrate_floor([float(i) for i in range(1, 101)])
        assert f["n"] == 100
        assert f["p10"] == pytest.approx(10.9)
        assert f["p25"] == pytest.approx(25.75)
        assert f["median"] == pytest.approx(50.5)

    def test_luma_floor_is_added_when_measured(self):
        f = stock_select.calibrate_floor([1.0, 2.0], [float(i) for i in range(1, 101)])
        assert f["luma_p10"] == pytest.approx(10.9)
        assert "luma_p10" not in stock_select.calibrate_floor([1.0, 2.0])

    def test_clip_floor_is_the_95th_percentile(self):
        f = stock_select.calibrate_floor([1.0, 2.0], [1.0], [float(i) / 100 for i in range(1, 101)])
        assert f["clip_p95"] == pytest.approx(0.9505)
        assert "clip_p95" not in stock_select.calibrate_floor([1.0, 2.0])

    def test_complete_floor_passes_and_an_old_one_names_what_is_missing(self):
        assert stock_select.require_complete_floor(FLOOR) is FLOOR
        with pytest.raises(StockFloorMissing, match="has no clip_p95; re-run"):
            stock_select.require_complete_floor({k: v for k, v in FLOOR.items() if k != "clip_p95"})

    def test_empty_calibration_is_refused(self):
        with pytest.raises(ValueError, match="cannot calibrate a floor from zero measurements"):
            stock_select.calibrate_floor([])

    def test_missing_floor_names_the_path_and_the_fix(self, tmp_path):
        with pytest.raises(StockFloorMissing, match=r"no sharpness floor at .*nofloor\.json; run `bio stock-calibrate"):
            stock_select.load_floor(tmp_path / "nofloor.json")

    def test_saved_floor_loads_back_identically(self, tmp_path):
        write_json(tmp_path / "f.json", FLOOR)
        assert stock_select.load_floor(tmp_path / "f.json") == FLOOR

    def test_calibrate_from_folder_measures_halved_images_and_saves(self, tmp_path):
        from PIL import Image

        folder = tmp_path / "approved" / "sub"
        folder.mkdir(parents=True)
        big = np.kron(checker(66, 66), np.ones((2, 2, 1), np.uint8))                                # 2x2-pixel blocks: halves to the 66x66 checker
        Image.fromarray(big).save(folder / "a.jpg", quality=100, subsampling=0)
        (tmp_path / "approved" / "notes.txt").write_text("not an image")
        floor = stock_select.calibrate_from_folder(tmp_path / "approved", tmp_path / "floor.json")
        assert floor["n"] == 1
        assert floor["p10"] == floor["median"] == floor["p25"]
        assert floor["p10"] == pytest.approx(CHECKER_VAR, rel=0.02)                                 # JPEG q100 is the only deviation from exact
        assert floor["clip_p95"] == pytest.approx(0.5, abs=0.02)                                    # half of the pixels are 255
        assert floor["luma_p10"] == pytest.approx(127.5, abs=1.0)                                   # half the pixels 255, half 0: the median sits at the 127/128 midpoint
        assert read_json(tmp_path / "floor.json") == floor

    def test_excluded_names_do_not_calibrate_the_floor(self, tmp_path):
        from PIL import Image

        folder = tmp_path / "approved"
        folder.mkdir()
        sharp = np.kron(checker(66, 66), np.ones((2, 2, 1), np.uint8))
        Image.fromarray(sharp).save(folder / "keep.jpg", quality=100, subsampling=0)
        Image.fromarray(np.full((132, 132, 3), 128, np.uint8)).save(folder / "mine.jpg", quality=100)         # flat: sharpness 0
        both = stock_select.calibrate_from_folder(folder, tmp_path / "a.json")
        only = stock_select.calibrate_from_folder(folder, tmp_path / "b.json", exclude_names={"mine.jpg"})
        assert both["n"] == 2 and both["median"] == pytest.approx(CHECKER_VAR / 2, rel=0.02)
        assert only["n"] == 1 and only["median"] == pytest.approx(CHECKER_VAR, rel=0.02)

    def test_calibrate_from_folder_without_jpegs(self, tmp_path):
        with pytest.raises(ValueError, match="no JPEGs under"):
            stock_select.calibrate_from_folder(tmp_path, tmp_path / "f.json")


def passing(**kw):
    base = dict(mp=45.4, iso=800, score=FrameScore(50.0, 0.0), floor=FLOOR)
    base.update(kw)
    return stock_select.technical_reasons(**base)


class TestTechnicalReasons:
    def test_clean_frame_has_no_reasons(self):
        assert passing() == []

    @pytest.mark.parametrize("mp,ok", [(4.0, True), (3.99, False), (100.0, True), (100.01, False)])
    def test_megapixel_bounds(self, mp, ok):
        assert (passing(mp=mp) == []) is ok

    def test_iso_boundary_and_message(self):
        assert passing(iso=3200) == []
        assert passing(iso=3201) == ["ISO 3201 > 3200"]
        assert passing(iso=6400, ) == ["ISO 6400 > 3200"]

    def test_unknown_iso_is_a_reason_not_a_pass(self):
        assert passing(iso=None) == ["ISO unknown"]

    def test_sharpness_floor_boundary(self):
        assert passing(score=FrameScore(18.4, 0.0)) == []
        assert passing(score=FrameScore(18.3, 0.0)) == ["sharpness 18.3 < floor 18.4"]

    def test_clipping_boundary(self):
        assert passing(score=FrameScore(50.0, 0.02)) == []
        assert passing(score=FrameScore(50.0, 0.0201)) == ["2.0% clipped highlights"]

    def test_the_clipping_limit_is_the_floors_not_a_constant(self):
        loose = {**FLOOR, "clip_p95": 0.04}
        assert passing(score=FrameScore(50.0, 0.033), floor=loose) == []
        assert passing(score=FrameScore(50.0, 0.033)) == ["3.3% clipped highlights"]

    def test_location_is_a_reason(self):
        assert passing(located=True) == ["carries a location"]

    def test_unscored_frame_is_only_that_reason(self):
        assert passing(score=None, iso=9999, mp=1.0) == ["unscored"]

    def test_reasons_accumulate_in_a_fixed_order(self):
        assert passing(mp=2.0, iso=6400, score=FrameScore(1.0, 0.5), located=True) == [
            "2.0 MP outside 4-100", "ISO 6400 > 3200", "sharpness 1.0 < floor 18.4", "50.0% clipped highlights", "carries a location"]

    def test_custom_max_iso(self):
        assert passing(iso=6400, max_iso=6400) == []


def frame(i, group, iso=800, mp=45.4, located=False):
    return {"id": i, "group": group, "mp": mp, "iso": iso, "located": located}


class TestDecide:
    def test_sharpest_two_per_group_win_even_when_listed_last(self):
        frames = [frame(1, "a"), frame(2, "a"), frame(3, "a"), frame(4, "b")]
        scores = {1: FrameScore(40, 0), 2: FrameScore(90, 0), 3: FrameScore(60, 0), 4: FrameScore(25, 0)}
        rows = {r["id"]: r for r in stock_select.decide(frames, scores, FLOOR, top_per_group=2)}
        assert [i for i, r in rows.items() if r["stock_candidate"]] == [2, 3, 4]
        assert rows[1]["reasons"] == ["burst: not among the 2 sharpest of its group"]
        assert rows[1]["tech_pass"] is True and rows[1]["stock_candidate"] is False

    def test_by_default_every_frame_that_passes_is_kept(self):
        frames = [frame(1, "a"), frame(2, "a"), frame(3, "a")]
        scores = {1: FrameScore(40, 0), 2: FrameScore(90, 0), 3: FrameScore(60, 0)}
        rows = stock_select.decide(frames, scores, FLOOR)
        assert [r["stock_candidate"] for r in rows] == [True, True, True]

    def test_sharpest_frame_failing_the_bar_does_not_take_a_slot(self):
        frames = [frame(1, "a", iso=6400), frame(2, "a"), frame(3, "a")]
        scores = {1: FrameScore(500, 0), 2: FrameScore(30, 0), 3: FrameScore(20, 0)}
        rows = {r["id"]: r for r in stock_select.decide(frames, scores, FLOOR, top_per_group=1)}
        assert [i for i, r in rows.items() if r["stock_candidate"]] == [2]
        assert rows[1]["reasons"] == ["ISO 6400 > 3200"] and rows[1]["tech_pass"] is False

    def test_flagged_group_is_excluded_even_when_its_frames_are_the_sharpest(self):
        frames = [frame(1, "a"), frame(2, "a"), frame(3, "b"), frame(4, "b")]
        scores = {1: FrameScore(400, 0), 2: FrameScore(300, 0), 3: FrameScore(30, 0), 4: FrameScore(25, 0)}
        rows = {r["id"]: r for r in stock_select.decide(frames, scores, FLOOR, excluded_groups={"a": "flagged people_visible, sign_or_text"})}
        assert [i for i, r in rows.items() if r["stock_candidate"]] == [3, 4]
        assert rows[1]["reasons"] == ["group excluded: flagged people_visible, sign_or_text"] and rows[1]["tech_pass"] is False
        unflagged = {r["id"]: r for r in stock_select.decide(frames, scores, FLOOR)}
        assert [i for i, r in unflagged.items() if r["stock_candidate"]] == [1, 2, 3, 4]       # positive twin: without the exclusion they win

    def test_every_row_is_marked_release_unscreened_and_scores_are_rounded(self):
        rows = stock_select.decide([frame(1, "a")], {1: FrameScore(50.126, 0.012345678)}, FLOOR)
        assert (rows[0]["release"], rows[0]["sharp"], rows[0]["clip"], rows[0]["mp"]) == ("unscreened", 50.13, 0.01235, 45.4)

    def test_unscored_frame_is_reported_not_picked(self):
        rows = stock_select.decide([frame(1, "a")], {}, FLOOR)
        assert rows[0]["stock_candidate"] is False and rows[0]["reasons"] == ["unscored"] and rows[0]["sharp"] is None


class TestRawHelpers:
    def test_real_nef_fixture_measures_to_pinned_values(self):
        raw = FIXTURES / "sample_raw.NEF"
        s = stock_select.score_raw(raw)
        assert s.sharp == pytest.approx(48.284, rel=1e-3)
        assert s.clip == pytest.approx(0.04104, rel=1e-3)
        assert stock_select.raw_megapixels(raw) == pytest.approx(45.74976, rel=1e-6)
        assert stock_select.raw_iso(raw) == 64

    def test_second_fixture_is_distinguishable(self):
        s = stock_select.score_raw(FIXTURES / "sample_raw_fungus.NEF")
        assert s.sharp == pytest.approx(70.648, rel=1e-3)
        assert s.clip == pytest.approx(0.013399, rel=1e-3)


def seed_folder(tmp_path, *, observation_id=408509974, quality_grade="research", rank="species"):
    from bio_automation.pipeline import save_observation

    pipeline.ensure_layout(tmp_path)
    raw_dir = source_raw_dir(tmp_path)
    for name in ("DSC_0001.NEF", "DSC_0002.NEF", "DSC_0003.NEF"):
        (raw_dir / name).write_bytes(b"x")
    pipeline.save_assets(tmp_path, [
        Asset(file="source/raw/DSC_0001.NEF", role="original", status="kept", selected=True),
        Asset(file="source/raw/DSC_0002.NEF", role="original", status="kept", selected=True),
        Asset(file="source/raw/DSC_0003.NEF", role="original", status="rejected", selected=False),
    ])
    save_observation(tmp_path, Observation(
        observation_id=observation_id, obsdate="2026-06-13", place_guess="Pennsylvania, US", quality_grade=quality_grade,
        taxon={"name": "Leiobunum verrucosum", "common_name": "Warty Harvestman", "iconic_taxon_name": "Arachnida", "rank": rank}))
    write_json(tmp_path / "floor.json", FLOOR)


def stub_raws(monkeypatch, measured=None):
    scored = []
    measured = measured or {"DSC_0001.NEF": FrameScore(80.0, 0.0), "DSC_0002.NEF": FrameScore(5.0, 0.0)}
    monkeypatch.setattr(stock_select, "score_raw", lambda p: scored.append(p.name) or measured[p.name])
    monkeypatch.setattr(stock_select, "raw_megapixels", lambda p: 45.4)
    monkeypatch.setattr(stock_select, "raw_iso", lambda p: 400)
    return scored


class TestTagObservation:
    def test_kept_raws_are_scored_labelled_and_written(self, tmp_path, monkeypatch):
        seed_folder(tmp_path)
        scored = stub_raws(monkeypatch)
        result = stock_select.tag_observation(tmp_path, tmp_path / "floor.json")
        assert scored == ["DSC_0001.NEF", "DSC_0002.NEF"]                      # the rejected raw is never read
        by_id = {f["id"]: f for f in result["frames"]}
        assert by_id["source/raw/DSC_0001.NEF"]["stock_candidate"] is True
        assert by_id["source/raw/DSC_0002.NEF"]["reasons"] == ["sharpness 5.0 < floor 18.4"]
        assert result["label_error"] is None
        assert result["label"]["title"] == "Warty Harvestman Leiobunum verrucosum - Pennsylvania"
        assert read_json(metadata_dir(tmp_path) / "stock_tags.json") == result

    def test_observation_not_linked_to_inat_has_no_candidates(self, tmp_path, monkeypatch):
        seed_folder(tmp_path, observation_id=None)
        stub_raws(monkeypatch)
        result = stock_select.tag_observation(tmp_path, tmp_path / "floor.json")
        assert result["label"] is None and result["label_error"] == "no iNat observation linked (run bio fetch-inat first)"
        assert [f["stock_candidate"] for f in result["frames"]] == [False, False]
        assert result["frames"][0]["reasons"] == ["group excluded: no iNat observation linked (run bio fetch-inat first)"]

    def test_observation_the_label_policy_refuses_has_no_candidates(self, tmp_path, monkeypatch):
        seed_folder(tmp_path, quality_grade="casual")
        stub_raws(monkeypatch)
        result = stock_select.tag_observation(tmp_path, tmp_path / "floor.json")
        assert result["label_error"] == "observation 408509974: casual observation"
        assert [f["stock_candidate"] for f in result["frames"]] == [False, False]

    def test_a_raw_that_cannot_be_decoded_is_tagged_unscored(self, tmp_path, monkeypatch):
        seed_folder(tmp_path)

        def decode(path):
            if path.name == "DSC_0001.NEF":
                raise RuntimeError("bad raw")
            return FrameScore(80.0, 0.0)

        stub_raws(monkeypatch)
        monkeypatch.setattr(stock_select, "score_raw", decode)
        by_id = {f["id"]: f for f in stock_select.tag_observation(tmp_path, tmp_path / "floor.json")["frames"]}
        assert by_id["source/raw/DSC_0001.NEF"]["reasons"] == ["unscored"]
        assert by_id["source/raw/DSC_0002.NEF"]["stock_candidate"] is True

    def test_missing_floor_raises(self, tmp_path):
        seed_folder(tmp_path)
        with pytest.raises(StockFloorMissing, match="no sharpness floor at"):
            stock_select.tag_observation(tmp_path, tmp_path / "absent.json")


class TestPipelineHook:
    def test_missing_floor_is_reported_and_logged(self, tmp_path, monkeypatch, caplog):
        def raiser(folder):
            raise StockFloorMissing("no sharpness floor at X; run `bio stock-calibrate`")

        monkeypatch.setattr(stock_select, "tag_observation", raiser)
        with caplog.at_level(logging.WARNING):
            out = pipeline._tag_stock(tmp_path)
        assert out == {"skipped": "no sharpness floor at X; run `bio stock-calibrate`"}
        assert "stock tagging skipped: no sharpness floor at X" in caplog.text

    def test_tags_are_returned_when_a_floor_exists(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(stock_select, "tag_observation", lambda folder: calls.append(folder) or {"frames": [1]})
        assert pipeline._tag_stock(tmp_path) == {"frames": [1]}
        assert calls == [tmp_path]

    def test_other_errors_propagate(self, tmp_path, monkeypatch):
        def boom(folder):
            raise OSError("disk gone")

        monkeypatch.setattr(stock_select, "tag_observation", boom)
        with pytest.raises(OSError, match="disk gone"):
            pipeline._tag_stock(tmp_path)


def epoch(iso):
    import datetime as dt

    return dt.datetime.fromisoformat(iso).timestamp()


BASE = "2026-07-31T09:00:00-04:00"


def queue_item(lc_id, raw, seconds, loc="none", w=8256, h=5504, tmp=None):
    """A LightCraft queue frame captured `seconds` after BASE."""
    return {"lc_id": lc_id, "group": 99, "raw": raw, "day": "2026-07-31", "t": epoch(BASE) + seconds, "w": w, "h": h, "loc_source": loc,
            "copy": str(tmp / f"{lc_id}.jpg")}


def make_copy(path, iso):
    from PIL import Image

    ex = Image.Exif()
    ex.get_ifd(0x8769)[0x8827] = iso
    Image.new("RGB", (8, 8)).save(path, "JPEG", exif=ex.tobytes())


def fake_fetch(calls, overrides=None):
    overrides = overrides or {}

    def fetch(obs_id):
        calls.append(obs_id)
        base = {"observation_id": obs_id, "source": "inaturalist", "place_guess": "Pennsylvania, US", "license": "cc-by", "quality_grade": "research",
                "keywords_seed": [], "taxon": {"id": 1, "name": "Leiobunum verrucosum", "preferred_common_name": "Warty Harvestman",
                                                  "common_name": "Warty Harvestman", "iconic_taxon_name": "Arachnida", "rank": "species"}}
        return {**base, **overrides.get(obs_id, {})}

    return fetch


OBS = [{"id": 11, "time": "2026-07-31T09:00:10-04:00", "observed_on": "2026-07-31",
        "photos": [{"id": 1101, "url": "https://x/photos/1101/square.jpg"}, {"id": 1102, "url": "https://x/photos/1102/square.jpg"},
                   {"id": 1103, "url": "https://x/photos/1103/square.jpg"}]},
       {"id": 22, "time": "2026-07-31T09:10:00-04:00", "observed_on": "2026-07-31",
        "photos": [{"id": 2201, "url": "https://x/photos/2201/square.jpg"}]}]

# Unit vectors standing in for DA-CLIP embeddings, keyed by file stem. Observation 11 has three photos that ARE frames 1, 2, 3; observation 22's photo is
# frame 4. Frame 6 is a sibling shot (similar to photo 1101, 0.8) that was never uploaded; frame 5 is far from every observation in time.
E = [(1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0)]
VEC = {"1101": E[0], "1102": E[1], "1103": E[2], "2201": E[3], "1": E[0], "2": E[1], "3": E[2], "4": E[3], "5": (0.5, 0.5, 0.5, 0.5), "6": (0.8, 0.6, 0.0, 0.0)}


class TestTagQueue:
    @pytest.fixture(autouse=True)
    def fake_visual_check(self, monkeypatch):
        """The embedder (GPU model) and the photo download are the dependencies; the matching, labelling and gating run for real."""
        import numpy as np

        from bio_automation import inat_stock

        self.downloaded = []
        monkeypatch.setattr(inat_stock, "default_embedder", lambda: (lambda path: np.array(VEC[Path(path).stem], dtype=np.float32)))
        monkeypatch.setattr(inat_stock, "cached_photo", lambda pid, url, cache_dir, **kw: self.downloaded.append(pid) or Path(cache_dir) / f"{pid}.jpg")

    def seed(self, tmp_path):
        items = [queue_item(1, "2026/d/DSC_1.NEF", 0, tmp=tmp_path), queue_item(2, "2026/d/DSC_2.NEF", 30, tmp=tmp_path),
                 queue_item(3, "2026/d/DSC_3.NEF", 20, tmp=tmp_path),                       # three uploaded photos of observation 11
                 queue_item(4, "2026/d/DSC_4.NEF", 600, tmp=tmp_path),                      # the photo of observation 22
                 queue_item(5, "2026/d/DSC_5.NEF", 3000, tmp=tmp_path),                     # nowhere near an observation
                 queue_item(6, "2026/d/DSC_6.NEF", 40, tmp=tmp_path)]                       # a sibling shot that was never uploaded
        for i, iso in ((1, 100), (2, 200), (3, 6400), (4, 100), (5, 100), (6, 100)):
            make_copy(tmp_path / f"{i}.jpg", iso)
        write_json(tmp_path / "q.json", items)
        write_json(tmp_path / "floor.json", FLOOR)

    def run(self, tmp_path, monkeypatch, sharp=None, fetch=None, calls=None, **kw):
        self.seed(tmp_path)
        sharp = sharp or {"DSC_1.NEF": 30.0, "DSC_2.NEF": 90.0, "DSC_3.NEF": 500.0, "DSC_4.NEF": 40.0, "DSC_5.NEF": 900.0, "DSC_6.NEF": 800.0}
        monkeypatch.setattr(stock_select, "score_raw", lambda p: FrameScore(sharp[p.name], 0.0))
        calls = [] if calls is None else calls
        rows = stock_select.tag_queue(tmp_path / "q.json", tmp_path / "raws", tmp_path / "out.csv", observations=OBS, fetch=fetch or fake_fetch(calls),
                                      floor_path=tmp_path / "floor.json", workers=1, **kw)
        return {r["id"]: r for r in rows}, calls

    def test_exactly_the_uploaded_photos_are_labelled_candidates(self, tmp_path, monkeypatch):
        by, calls = self.run(tmp_path, monkeypatch)
        assert [i for i, r in by.items() if r["stock_candidate"]] == [1, 2, 4]       # 3 is ISO 6400; 5 and 6 were never uploaded, however sharp
        assert by[3]["reasons"] == "ISO 6400 > 3200"
        assert (by[2]["group"], by[2]["obs_id"], by[2]["taxon"], by[2]["grade"], by[2]["rank"], by[2]["similarity"]) == (11, 11, "Leiobunum verrucosum", "research", "species", 1.0)
        assert by[2]["title"] == "Warty Harvestman Leiobunum verrucosum - Pennsylvania"
        assert by[2]["keywords"].split("|")[:4] == ["warty", "harvestman", "leiobunum", "verrucosum"] and len(by[2]["keywords"].split("|")) == 16
        assert (by[2]["category"], by[2]["category_id"]) == ("Animals", 1)
        assert by[2]["caption"] == "Warty Harvestman (Leiobunum verrucosum) photographed at Pennsylvania, US on 2026-07-31."

    def test_every_photo_of_an_observation_is_used_not_just_the_sharpest_two(self, tmp_path, monkeypatch):
        by, _ = self.run(tmp_path, monkeypatch, sharp={"DSC_1.NEF": 30.0, "DSC_2.NEF": 90.0, "DSC_3.NEF": 500.0, "DSC_4.NEF": 40.0, "DSC_5.NEF": 9.0, "DSC_6.NEF": 8.0})
        assert by[1]["stock_candidate"] is True and by[2]["stock_candidate"] is True        # observation 11 keeps both of its passing photos

    def test_the_burst_cap_is_optional(self, tmp_path, monkeypatch):
        by, _ = self.run(tmp_path, monkeypatch, top_per_group=1)
        assert [i for i, r in by.items() if r["stock_candidate"]] == [2, 4]
        assert by[1]["reasons"] == "burst: not among the 1 sharpest of its group"

    def test_a_frame_that_is_not_an_uploaded_photo_is_never_a_candidate(self, tmp_path, monkeypatch):
        by, _ = self.run(tmp_path, monkeypatch)
        for i in (5, 6):
            assert by[i]["stock_candidate"] is False
            assert by[i]["reasons"] == "group excluded: not one of the photos uploaded to an iNat observation"
            assert (by[i]["obs_id"], by[i]["title"], by[i]["similarity"]) == ("", "", "")

    def test_a_photo_with_no_look_alike_frame_claims_nothing(self, tmp_path, monkeypatch):
        VEC["2"] = (0.0, 0.5, 0.5, 0.7071)                      # frame 2 no longer looks like photo 1102 (similarity 0.5)
        try:
            by, _ = self.run(tmp_path, monkeypatch)
        finally:
            VEC["2"] = E[1]
        assert by[2]["stock_candidate"] is False and by[2]["reasons"] == "group excluded: not one of the photos uploaded to an iNat observation"
        assert by[1]["stock_candidate"] is True and by[1]["obs_id"] == 11                       # the other photos still match

    def test_only_frames_that_can_still_be_candidates_are_measured(self, tmp_path, monkeypatch):
        scored = []
        self.seed(tmp_path)
        monkeypatch.setattr(stock_select, "score_raw", lambda p: scored.append(p.name) or FrameScore(90.0, 0.0))
        stock_select.tag_queue(tmp_path / "q.json", tmp_path, tmp_path / "out.csv", observations=OBS, fetch=fake_fetch([], {22: {"quality_grade": "casual"}}),
                               floor_path=tmp_path / "floor.json", workers=1)
        assert scored == ["DSC_1.NEF", "DSC_2.NEF", "DSC_3.NEF"]          # 4 belongs to a refused observation; 5 and 6 are not uploaded photos

    def test_every_photo_is_fetched_once_and_each_observation_once(self, tmp_path, monkeypatch):
        _, calls = self.run(tmp_path, monkeypatch)
        assert calls == [11, 22]
        assert sorted(self.downloaded) == ["1101", "1102", "1103", "2201"]

    def test_an_observation_the_policy_refuses_excludes_its_frames_with_the_reason(self, tmp_path, monkeypatch):
        by, _ = self.run(tmp_path, monkeypatch, fetch=fake_fetch([], {22: {"quality_grade": "casual"}}))
        assert by[4]["stock_candidate"] is False and by[4]["reasons"] == "group excluded: observation 22: casual observation"
        assert by[2]["stock_candidate"] is True                                  # positive twin: the other observation is untouched

    def test_a_label_too_thin_to_ship_excludes_the_observation(self, tmp_path, monkeypatch):
        thin = {"taxon": {"id": 2, "name": "Amanita muscaria", "preferred_common_name": None, "common_name": None, "iconic_taxon_name": "Fungi",
                          "rank": "species"}}
        by, _ = self.run(tmp_path, monkeypatch, fetch=fake_fetch([], {22: thin}))
        assert by[4]["reasons"] == "group excluded: label: 13 keywords (need at least 15)"
        assert (by[4]["stock_candidate"], by[4]["title"]) == (False, "Amanita muscaria - Pennsylvania")      # the label stays visible so the owner sees why

    def test_a_failed_fetch_excludes_the_observation(self, tmp_path, monkeypatch):
        def fetch(i):
            if i == 22:
                raise ConnectionError("offline")
            return fake_fetch([])(i)

        by, _ = self.run(tmp_path, monkeypatch, fetch=fetch)
        assert by[4]["reasons"] == "group excluded: observation 22: fetch failed: ConnectionError: offline"

    def test_hand_flagged_observation_is_excluded(self, tmp_path, monkeypatch):
        by, _ = self.run(tmp_path, monkeypatch, excluded_groups={11: "flagged people_visible"})
        assert by[2]["stock_candidate"] is False and by[2]["reasons"] == "group excluded: flagged people_visible"

    def test_csv_is_written_with_the_label_columns(self, tmp_path, monkeypatch):
        self.run(tmp_path, monkeypatch)
        with open(tmp_path / "out.csv", newline="", encoding="utf-8") as f:
            rows = {r["id"]: r for r in csv.DictReader(f)}
        assert rows["2"]["obs_id"] == "11" and rows["2"]["category_id"] == "1" and rows["2"]["stock_candidate"] == "True"
        assert rows["5"]["obs_id"] == "" and rows["5"]["day"] == "2026-07-31" and rows["5"]["raw"] == "2026/d/DSC_5.NEF"

    def test_default_fetch_asks_for_the_lineage_and_shares_one_cache(self, tmp_path, monkeypatch):
        from bio_automation import inat

        seen = []

        def real(obs_id, **kw):
            seen.append((obs_id, kw["with_lineage"], id(kw["lineage_cache"])))
            return fake_fetch([])(obs_id)

        monkeypatch.setattr(inat, "fetch_inat_observation", real)
        self.seed(tmp_path)
        monkeypatch.setattr(stock_select, "score_raw", lambda p: FrameScore(90.0, 0.0))
        stock_select.tag_queue(tmp_path / "q.json", tmp_path, tmp_path / "out.csv", observations=OBS, floor_path=tmp_path / "floor.json", workers=1)
        assert [(o, w) for o, w, _ in seen] == [(11, True), (22, True)]
        assert len({c for _, _, c in seen}) == 1                                      # one shared cache, so a taxon is looked up once

    def test_iso_comes_from_the_raw_when_the_picture_has_no_exif(self, tmp_path, monkeypatch):
        from PIL import Image

        self.seed(tmp_path)
        Image.new("RGB", (8, 8)).save(tmp_path / "2.jpg", "JPEG")                       # frame 2's picture carries no EXIF at all
        asked = []
        monkeypatch.setattr(stock_select, "raw_iso", lambda p: asked.append(p.name) or 6400)
        monkeypatch.setattr(stock_select, "score_raw", lambda p: FrameScore(90.0, 0.0))
        rows = {r["id"]: r for r in stock_select.tag_queue(tmp_path / "q.json", tmp_path / "raws", tmp_path / "out.csv", observations=OBS,
                                                           fetch=fake_fetch([]), floor_path=tmp_path / "floor.json", workers=1)}
        assert asked == ["DSC_2.NEF"]                                                    # only the picture without EXIF needed the raw
        assert rows[2]["iso"] == 6400 and rows[2]["reasons"] == "ISO 6400 > 3200"
        assert rows[1]["iso"] == 100

    def test_rerun_uses_the_cache_and_does_not_rescore(self, tmp_path, monkeypatch):
        self.run(tmp_path, monkeypatch)
        calls = []
        monkeypatch.setattr(stock_select, "score_raw", lambda p: calls.append(p.name))
        stock_select.tag_queue(tmp_path / "q.json", tmp_path, tmp_path / "out.csv", observations=OBS, fetch=fake_fetch([]),
                               floor_path=tmp_path / "floor.json", workers=1)
        assert calls == []

    def test_unreadable_raw_is_an_unscored_row(self, tmp_path, monkeypatch):
        self.seed(tmp_path)

        def decode(p):
            if p.name == "DSC_2.NEF":
                raise RuntimeError("corrupt")
            return FrameScore(90.0, 0.0)

        monkeypatch.setattr(stock_select, "score_raw", decode)
        rows = stock_select.tag_queue(tmp_path / "q.json", tmp_path, tmp_path / "out.csv", observations=OBS, fetch=fake_fetch([]),
                                      floor_path=tmp_path / "floor.json", workers=1)
        assert {r["id"]: r["reasons"] for r in rows}[2] == "unscored"

    def test_limit_scores_only_that_many_and_marks_the_rest_unscored(self, tmp_path, monkeypatch):
        self.seed(tmp_path)
        monkeypatch.setattr(stock_select, "score_raw", lambda p: FrameScore(90.0, 0.0))
        rows = stock_select.tag_queue(tmp_path / "q.json", tmp_path, tmp_path / "out.csv", observations=OBS, fetch=fake_fetch([]),
                                      floor_path=tmp_path / "floor.json", workers=1, limit=2)
        assert [r["reasons"] for r in rows if r["id"] in (3, 4)] == ["unscored", "unscored"]

    def test_observations_are_required(self, tmp_path):
        self.seed(tmp_path)
        with pytest.raises(TypeError, match="observations"):
            stock_select.tag_queue(tmp_path / "q.json", tmp_path, tmp_path / "out.csv", floor_path=tmp_path / "floor.json", workers=1)


LABEL_COLS = {"title": "Warty Harvestman Leiobunum verrucosum - Pennsylvania", "caption": "Warty Harvestman photographed at Pennsylvania, US.",
              "keywords": "|".join(f"kw{i}" for i in range(16)), "category": "Animals", "category_id": "1"}


class TestPublish:
    ROWS = [{"raw": "2026/d/DSC_1.NEF", "day": "2026-07-31", "stock_candidate": "True", **LABEL_COLS},
            {"raw": "2026/d/DSC_2.NEF", "day": "2026-07-31", "stock_candidate": "False", **LABEL_COLS},
            {"raw": "2026/d/DSC_3.NEF", "day": "2026-08-01", "stock_candidate": "True", **LABEL_COLS}]

    @pytest.fixture(autouse=True)
    def floor_file(self, tmp_path, monkeypatch):
        write_json(tmp_path / "pubfloor.json", FLOOR)
        monkeypatch.setattr(stock_select, "DEFAULT_FLOOR_PATH", tmp_path / "pubfloor.json")

    def csv_file(self, tmp_path, rows=None):
        p = tmp_path / "c.csv"
        rows = rows or self.ROWS
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        return p

    def render(self, tmp_path, day, name, size=(2000, 2000), quality=95, gps=False, color=(90, 120, 60)):
        from PIL import Image, ImageCms

        p = tmp_path / "render" / day[:4] / day / name
        p.parent.mkdir(parents=True, exist_ok=True)
        kw = {"icc_profile": ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()}
        if gps:
            ex = Image.Exif()
            g = ex.get_ifd(0x8825)
            g[1], g[2], g[3], g[4] = "N", (40.0, 0.0, 0.0), "W", (75.0, 0.0, 0.0)
            kw["exif"] = ex.tobytes()
        Image.new("RGB", size, color).save(p, "JPEG", quality=quality, **kw)
        return p

    def publish(self, tmp_path, rows=None, **kw):
        return stock_select.publish_candidates(self.csv_file(tmp_path, rows), tmp_path / "raws", tmp_path / "render", tmp_path / "gallery", **kw)

    def test_render_name_carries_the_day(self):
        assert stock_select.render_name(self.ROWS[0]) == "2026-07-31_DSC_1.jpg"

    def test_lightroom_export_names_are_shortened_and_made_unique_with_the_frame_id(self):
        row = {"raw": "2025/2025-04-25/DSC_3770 (2025_06_03 19_23_22 UTC)-Enhanced-NR.dng", "day": "2025-04-25", "id": "14081"}
        name = stock_select.render_name(row)
        assert name == "2025-04-25_DSC_3770_14081.jpg" and len(name) <= 30
        assert stock_select.render_name({**row, "id": "14082"}) == "2025-04-25_DSC_3770_14082.jpg"          # two frames of one day never collide

    def test_labelled_renders_are_embedded_linked_and_listed_in_the_adobe_csv(self, tmp_path):
        from bio_automation import inat_stock

        src1 = self.render(tmp_path, "2026-07-31", "2026-07-31_DSC_1.jpg")
        self.render(tmp_path, "2026-07-31", "2026-07-31_DSC_2.jpg")                       # a render exists for the non-candidate: still not shipped
        self.render(tmp_path, "2026-08-01", "2026-08-01_DSC_3.jpg")
        out = self.publish(tmp_path)
        assert out == {"candidates": 2, "plain_decoded": 0, "linked": 2, "missing": [], "failed": [], "adobe_csv_rows": 2}
        link = tmp_path / "gallery/2026/2026-07/2026-07-31_DSC_1.jpg"
        assert os.path.samefile(src1, link)
        assert inat_stock.read_label(link) == {"title": LABEL_COLS["title"], "caption": LABEL_COLS["caption"], "keywords": LABEL_COLS["keywords"].split("|")}
        assert not (tmp_path / "gallery/2026/2026-07/2026-07-31_DSC_2.jpg").exists()
        assert inat_stock.read_label(tmp_path / "render/2026/2026-07-31/2026-07-31_DSC_2.jpg")["title"] == ""        # untouched, not a candidate
        rows = list(csv.DictReader(open(tmp_path / "render" / "adobe_stock_metadata.csv", newline="", encoding="utf-8-sig")))
        assert [(r["Filename"], r["Category"], r["Releases"]) for r in rows] == [("2026-07-31_DSC_1.jpg", "1", ""), ("2026-08-01_DSC_3.jpg", "1", "")]
        assert rows[0]["Keywords"] == ",".join(f"kw{i}" for i in range(16))

    def test_a_label_that_does_not_read_back_is_failed_and_never_linked(self, tmp_path, monkeypatch):
        from bio_automation import inat_stock

        embedded = []
        monkeypatch.setattr(inat_stock, "embed_label", lambda jpeg, label: embedded.append(jpeg.name))      # the write silently does nothing
        self.render(tmp_path, "2026-07-31", "2026-07-31_DSC_1.jpg")
        out = self.publish(tmp_path, [self.ROWS[0]])
        assert embedded == ["2026-07-31_DSC_1.jpg"]
        assert out["failed"] == [{"raw": "2026/d/DSC_1.NEF", "error": "embedded metadata does not read back as written"}]
        assert out["linked"] == 0 and out["adobe_csv_rows"] == 0 and not (tmp_path / "gallery").exists()

    def test_rerun_keeps_one_link_per_file_and_refreshes_a_stale_one(self, tmp_path):
        self.render(tmp_path, "2026-07-31", "2026-07-31_DSC_1.jpg")
        self.publish(tmp_path, [self.ROWS[0]])
        again = self.publish(tmp_path, [self.ROWS[0]])
        assert again["failed"] == [] and again["adobe_csv_rows"] == 1
        link = tmp_path / "gallery/2026/2026-07/2026-07-31_DSC_1.jpg"
        assert os.path.samefile(tmp_path / "render/2026/2026-07-31/2026-07-31_DSC_1.jpg", link)       # re-embedding replaced the inode; the link followed

    def test_a_candidate_without_a_label_is_failed_and_never_linked(self, tmp_path):
        self.render(tmp_path, "2026-07-31", "2026-07-31_DSC_1.jpg")
        bare = {"raw": "2026/d/DSC_1.NEF", "day": "2026-07-31", "stock_candidate": "True"}
        out = self.publish(tmp_path, [bare])
        assert out["linked"] == 0 and out["adobe_csv_rows"] == 0
        assert out["failed"] == [{"raw": "2026/d/DSC_1.NEF", "error": "no title; 0 keywords (need at least 15); no caption; no Adobe category number for ''"}]
        assert not (tmp_path / "gallery").exists() and not (tmp_path / "render" / "adobe_stock_metadata.csv").exists()

    def test_a_candidate_without_a_retouched_render_is_missing_and_never_decoded_plain(self, tmp_path, monkeypatch):
        from bio_automation import render

        decoded = []
        monkeypatch.setattr(render, "export_stock_jpeg", lambda raw, dest: decoded.append(raw))
        self.render(tmp_path, "2026-08-01", "2026-08-01_DSC_3.jpg")
        out = self.publish(tmp_path)
        assert out["missing"] == ["2026/d/DSC_1.NEF"] and out["linked"] == 1
        assert decoded == []
        assert not (tmp_path / "gallery/2026/2026-07").exists()

    def test_plain_decode_only_runs_when_asked_for(self, tmp_path, monkeypatch):
        from PIL import Image

        from bio_automation import render

        decoded = []

        def fake(raw, dest):
            decoded.append(raw.name)
            self.render(tmp_path, "2026-07-31", dest.name)

        monkeypatch.setattr(render, "export_stock_jpeg", fake)
        out = self.publish(tmp_path, [self.ROWS[0]], plain_decode=True)
        assert decoded == ["DSC_1.NEF"]
        assert (out["plain_decoded"], out["linked"], out["missing"]) == (1, 1, [])
        assert Image.open(tmp_path / "gallery/2026/2026-07/2026-07-31_DSC_1.jpg").size == (2000, 2000)

    def test_renders_that_break_the_stock_standard_are_failed_not_linked(self, tmp_path):
        self.render(tmp_path, "2026-07-31", "2026-07-31_DSC_1.jpg", size=(100, 100))                 # under 4 MP
        self.render(tmp_path, "2026-08-01", "2026-08-01_DSC_3.jpg", gps=True)                        # carries a GPS block
        out = self.publish(tmp_path)
        assert out["linked"] == 0
        assert out["failed"] == [
            {"raw": "2026/d/DSC_1.NEF", "error": "100x100 = 0.01 MP is below the 4 MP minimum"},
            {"raw": "2026/d/DSC_3.NEF", "error": "contains a GPS block (Stock files must not carry coordinates)"}]
        assert not (tmp_path / "gallery").exists()

    def test_dark_render_is_failed_and_one_just_bright_enough_is_linked(self, tmp_path):
        self.render(tmp_path, "2026-07-31", "2026-07-31_DSC_1.jpg", color=(10, 10, 10))              # median luma 10 < 35
        self.render(tmp_path, "2026-08-01", "2026-08-01_DSC_3.jpg", color=(36, 36, 36))              # median luma 36 >= 35
        out = self.publish(tmp_path)
        assert out["failed"] == [{"raw": "2026/d/DSC_1.NEF", "error": "under-exposed: median luma 10.0 < floor 35.0"}]
        assert out["linked"] == 1 and (tmp_path / "gallery/2026/2026-08/2026-08-01_DSC_3.jpg").is_file()

    def test_floor_without_luma_or_clip_is_refused(self, tmp_path):
        write_json(tmp_path / "pubfloor.json", {k: v for k, v in FLOOR.items() if k not in ("luma_p10", "clip_p95")})
        with pytest.raises(StockFloorMissing, match="the sharpness floor has no luma_p10, clip_p95; re-run `bio stock-calibrate"):
            self.publish(tmp_path)

    def test_low_quality_render_is_failed(self, tmp_path):
        self.render(tmp_path, "2026-07-31", "2026-07-31_DSC_1.jpg", quality=80)
        out = self.publish(tmp_path, [self.ROWS[0]])
        assert out["failed"] == [{"raw": "2026/d/DSC_1.NEF", "error": "JPEG quality below 95 (quantisation table coarser than q95)"}]


class TestCli:
    def args(self, *extra):
        return cli.build_parser().parse_args(["stock-tag-queue", "q.json", "--raw-root", "r", "--out", "o.csv", *extra])

    def test_observations_flag_is_required(self, capsys):
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(["stock-tag-queue", "q.json", "--raw-root", "r", "--out", "o.csv"])
        assert "--observations" in capsys.readouterr().err

    def test_tag_queue_without_a_floor_exits_with_the_fix(self, tmp_path, monkeypatch):
        write_json(tmp_path / "q.json", [])
        write_json(tmp_path / "obs.json", {"observations": []})
        monkeypatch.setattr(stock_select, "DEFAULT_FLOOR_PATH", tmp_path / "none.json")
        args = cli.build_parser().parse_args(["stock-tag-queue", str(tmp_path / "q.json"), "--raw-root", str(tmp_path), "--out", str(tmp_path / "o.csv"),
                                               "--observations", str(tmp_path / "obs.json")])
        with pytest.raises(SystemExit, match=r"no sharpness floor at .*none\.json; run `bio stock-calibrate"):
            cli.cmd_stock_tag_queue(args)

    def test_stale_observation_snapshot_exits_with_the_fix(self, tmp_path):
        import time

        p = tmp_path / "obs.json"
        write_json(p, {"observations": []})
        old = time.time() - 5 * 86400
        os.utime(p, (old, old))
        with pytest.raises(SystemExit, match=r"obs\.json is 5\.0 days old \(limit 3\); refresh it from the live iNat account first"):
            cli.cmd_stock_tag_queue(self.args("--observations", str(p)))

    def test_tag_queue_passes_snapshot_and_notes_groups(self, tmp_path, monkeypatch):
        seen = {}
        monkeypatch.setattr(stock_select, "tag_queue", lambda *a, **k: seen.update(k) or [{"stock_candidate": False, "group": 1}])
        write_json(tmp_path / "obs.json", {"observations": [{"id": 1, "time": "2026-07-31T09:00:00-04:00"}]})
        notes = tmp_path / "n.json"
        notes.write_text('{"92": {"flags": ["not_wildlife", "electronics"], "note": "board"}}', encoding="utf-8")
        cli.cmd_stock_tag_queue(self.args("--observations", str(tmp_path / "obs.json"), "--notes", str(notes)))
        assert seen["observations"] == [{"id": 1, "time": "2026-07-31T09:00:00-04:00"}]
        assert seen["excluded_groups"] == {92: "flagged not_wildlife, electronics"}

    def test_created_since_keeps_only_the_newer_observations_and_top_per_group_is_passed(self, tmp_path, monkeypatch):
        seen = {}
        monkeypatch.setattr(stock_select, "tag_queue", lambda *a, **k: seen.update(k) or [{"stock_candidate": False, "group": 1}])
        write_json(tmp_path / "obs.json", {"observations": [{"id": 1, "created_at": "2026-10-11T23:59:59-04:00"},
                                                              {"id": 2, "created_at": "2026-10-12T00:00:01-04:00"}, {"id": 3}]})
        cli.cmd_stock_tag_queue(self.args("--observations", str(tmp_path / "obs.json"), "--created-since", "2026-10-12", "--top-per-group", "3"))
        assert [o["id"] for o in seen["observations"]] == [2] and seen["top_per_group"] == 3
        cli.cmd_stock_tag_queue(self.args("--observations", str(tmp_path / "obs.json")))
        assert [o["id"] for o in seen["observations"]] == [1, 2, 3] and seen["top_per_group"] is None

    def test_calibrate_command_leaves_out_the_files_named_in_the_csv(self, tmp_path, monkeypatch, capsys):
        seen = {}
        monkeypatch.setattr(stock_select, "calibrate_from_folder", lambda folder, out, sample, exclude: seen.update(exclude=exclude) or {"n": 1})
        (tmp_path / "a.csv").write_text("Filename,Title\nx.jpg,T\ny.jpg,U\n", encoding="utf-8-sig")
        cli.cmd_stock_calibrate(cli.build_parser().parse_args(["stock-calibrate", str(tmp_path), "--exclude-csv", str(tmp_path / "a.csv")]))
        assert seen["exclude"] == {"x.jpg", "y.jpg"}
        cli.cmd_stock_calibrate(cli.build_parser().parse_args(["stock-calibrate", str(tmp_path)]))
        assert seen["exclude"] is None

    def test_min_similarity_is_passed_through(self, tmp_path, monkeypatch):
        seen = {}
        monkeypatch.setattr(stock_select, "tag_queue", lambda *a, **k: seen.update(k) or [{"stock_candidate": False, "group": 1}])
        write_json(tmp_path / "obs.json", {"observations": []})
        cli.cmd_stock_tag_queue(self.args("--observations", str(tmp_path / "obs.json"), "--min-similarity", "0.93"))
        assert seen["min_similarity"] == 0.93
        cli.cmd_stock_tag_queue(self.args("--observations", str(tmp_path / "obs.json")))
        assert seen["min_similarity"] is None

    def test_tag_queue_prints_the_counts(self, tmp_path, monkeypatch, capsys):
        rows = [{"stock_candidate": True, "group": 1}, {"stock_candidate": False, "group": 1}, {"stock_candidate": True, "group": 2},
                {"stock_candidate": False, "group": 3}]
        monkeypatch.setattr(stock_select, "tag_queue", lambda *a, **k: rows)
        write_json(tmp_path / "obs.json", {"observations": []})
        assert cli.cmd_stock_tag_queue(self.args("--observations", str(tmp_path / "obs.json"))) == 0
        assert capsys.readouterr().out == "2 Stock candidates of 4 frames in 2 of 3 groups -> o.csv\n"

    def test_fetch_inat_stores_the_lineage_with_the_observation(self, tmp_path, monkeypatch):
        from bio_automation.pipeline import ensure_layout, load_observation, save_observation

        ensure_layout(tmp_path)
        save_observation(tmp_path, Observation(obsdate="2026-06-13"))
        asked = []
        monkeypatch.setattr(cli, "fetch_inat_observation", lambda oid, **kw: asked.append((oid, kw)) or {
            "observation_id": oid, "source": "inaturalist", "place_guess": "PA", "license": None, "quality_grade": "research", "keywords_seed": [],
            "taxon": {"name": "Ramaria", "lineage": [{"rank": "family", "name": "Gomphaceae", "common_name": None}]}})
        cli.cmd_fetch_inat(cli.build_parser().parse_args(["fetch-inat", str(tmp_path), "--observation-id", "408509974"]))
        assert asked == [(408509974, {"with_lineage": True})]
        assert load_observation(tmp_path).taxon["lineage"] == [{"rank": "family", "name": "Gomphaceae", "common_name": None}]

    def test_publish_exit_code_reflects_failures_and_missing_renders(self, monkeypatch):
        seen = []
        result = {"candidates": 1, "plain_decoded": 0, "linked": 0, "missing": [], "failed": [{"raw": "x", "error": "e"}]}
        monkeypatch.setattr(stock_select, "publish_candidates", lambda *a, **k: seen.append(k) or result)
        argv = ["stock-publish", "c.csv", "--raw-root", "r", "--render-dir", "d", "--gallery-dir", "g"]
        args = cli.build_parser().parse_args(argv)
        assert cli.cmd_stock_publish(args) == 1
        result = {**result, "failed": [], "missing": ["m"]}
        assert cli.cmd_stock_publish(args) == 1                                           # a missing render is also not a clean run
        result = {**result, "missing": []}
        assert cli.cmd_stock_publish(args) == 0
        assert seen == [{"plain_decode": False, "floor_path": None, "adobe_csv": None}] * 3
        cli.cmd_stock_publish(cli.build_parser().parse_args(argv + ["--plain-decode"]))
        assert seen[-1] == {"plain_decode": True, "floor_path": None, "adobe_csv": None}
        cli.cmd_stock_publish(cli.build_parser().parse_args(argv + ["--floor", "f.json"]))
        assert seen[-1] == {"plain_decode": False, "floor_path": Path("f.json"), "adobe_csv": None}
        cli.cmd_stock_publish(cli.build_parser().parse_args(argv + ["--adobe-csv", "out.csv"]))
        assert seen[-1]["adobe_csv"] == Path("out.csv")
