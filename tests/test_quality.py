"""Unit tests for bio_automation.quality.

Hermetic: every test here uses synthetic numpy arrays / PIL images built by
the test itself (no real raw decode, no rawpy/exiftool call) so the right
answer is hand-computable and pinned exactly, per AGENTS.md rule 1/7. The one
function that touches a real raw file, `_decode_preview_rgb`, is exercised
for real in tests/test_quality_live.py instead (same split test_preclean.py
vs test_preclean_live.py already uses in this repo).

suggest_culling()'s own dependency on disk I/O (_compute_asset_metrics) is
stubbed in TestSuggestCulling -- "stub only its dependencies" (rule 2) -- so
suggest_culling()'s own filtering/assembly/write logic is what's actually
exercised, not a reimplementation of it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from bio_automation import quality
from bio_automation.io import read_json
from bio_automation.models import Asset
from bio_automation.pipeline import ensure_layout, save_assets, scan_assets
from bio_automation.quality import (
    PRESORT_FILENAME,
    _build_verdicts,
    _compute_asset_metrics,
    difference_hash,
    exposure_clipping,
    hamming_distance,
    sharpness_score,
    suggest_culling,
)


# --------------------------------------------------------------------- sharpness_score

class TestSharpnessScore:
    def test_perfectly_flat_image_has_exactly_zero_variance(self):
        """A constant image has an identical 4-neighbor Laplacian response
        (0) at every pixel by construction -- the variance of an all-zero
        array is exactly 0.0, regardless of the fill value chosen."""
        flat = np.full((6, 6), 100.0, dtype=np.float32)
        assert sharpness_score(flat) == 0.0

    def test_sharp_vertical_edge_has_known_nonzero_variance(self):
        """Positive twin to the flat case above, on a fixture whose exact
        answer is hand-derivable: a hard step edge (cols 0-2 = 0, cols 3-5 =
        200) is constant along every row, so each row's 2D Laplacian reduces
        to the 1D case left+right-2*center with edge padding. Working that
        out by hand for the 6-value row [0,0,0,200,200,200] gives per-pixel
        Laplacian values [0,0,200,-200,0,0], repeated identically for all 6
        rows (the image is constant vertically): 24 zeros, 6 values of +200,
        6 of -200, mean 0, so variance = sum(x^2)/36 = (6*200^2 + 6*200^2)/36
        = 480000/36 = 40000/3 (~13333.33). float32 rounding moves this by a
        few parts in 1e4, hence approx with a tight relative tolerance."""
        row = [0.0, 0.0, 0.0, 200.0, 200.0, 200.0]
        edge_image = np.array([row] * 6, dtype=np.float32)

        result = sharpness_score(edge_image)

        assert result == pytest.approx(40000 / 3, rel=1e-3)
        # The actual property under test: real detail scores far above a
        # flat field, not a specific constant chosen to make the test pass.
        assert result > 0.0

    def test_accepts_non_float_input_without_crashing(self):
        """Boundary case: an 8-bit uint array (what callers in this module
        actually pass) must be cast internally, not raise on unsigned
        subtraction overflow -- same flat-image exact-zero fixture as above,
        at the actual input dtype real callers use."""
        flat_u8 = np.full((4, 4), 10, dtype=np.uint8)
        assert sharpness_score(flat_u8) == 0.0


# --------------------------------------------------------------------- exposure_clipping

class TestExposureClipping:
    def test_well_exposed_image_has_zero_clipping_both_ends(self):
        mid_gray = np.full((10, 10), 128, dtype=np.uint8)
        result = exposure_clipping(mid_gray)
        assert result == {"shadow_clipped_fraction": 0.0, "highlight_clipped_fraction": 0.0}

    def test_exact_shadow_clipped_fraction(self):
        """30 of 100 pixels at the shadow clip point -> exactly 0.30, not an
        estimate."""
        gray = np.full((10, 10), 128, dtype=np.uint8)
        gray[:3, :] = 0  # 3 rows * 10 cols = 30 pixels
        result = exposure_clipping(gray)
        assert result["shadow_clipped_fraction"] == pytest.approx(0.30)
        assert result["highlight_clipped_fraction"] == 0.0

    def test_exact_highlight_clipped_fraction(self):
        """Positive twin to the shadow case: 80 of 100 pixels at the
        highlight clip point -> exactly 0.80."""
        gray = np.full((10, 10), 128, dtype=np.uint8)
        gray[:8, :] = 255  # 8 rows * 10 cols = 80 pixels
        result = exposure_clipping(gray)
        assert result["highlight_clipped_fraction"] == pytest.approx(0.80)
        assert result["shadow_clipped_fraction"] == 0.0

    def test_shadow_threshold_boundary_is_inclusive(self):
        """Boundary case (rule 7): a pixel exactly at shadow_thresh (5) must
        count as clipped; one at 6 must not. 2x2 = 4 pixels, one of them at
        the boundary value -> exact 0.25."""
        gray = np.array([[5, 200], [200, 200]], dtype=np.uint8)
        result = exposure_clipping(gray, shadow_thresh=5)
        assert result["shadow_clipped_fraction"] == pytest.approx(0.25)

    def test_value_one_above_shadow_threshold_is_not_clipped(self):
        """Negative twin to the boundary test above, same fixture shape, one
        value higher: must drop to exactly 0.0, not 0.25."""
        gray = np.array([[6, 200], [200, 200]], dtype=np.uint8)
        result = exposure_clipping(gray, shadow_thresh=5)
        assert result["shadow_clipped_fraction"] == 0.0


# --------------------------------------------------------------------- difference_hash

def _gradient_image(ascending: bool) -> Image.Image:
    """An 8-row x 9-col grayscale image, every row identical, strictly
    monotonic left-to-right: ascending (0,10,...,80) or its reverse. Already
    exactly the size difference_hash resizes to for n=8, so its internal
    resize is an identity (BILINEAR resampling onto the same grid returns the
    same values -- verified independently before relying on it here) and the
    hash bits are fully hand-computable: every adjacent-column comparison is
    True for the ascending row (giving all 64 bits set, 2**64-1), and False
    for the descending row (giving 0)."""
    row = list(range(0, 90, 10))
    if not ascending:
        row = row[::-1]
    arr = np.array([row] * 8, dtype=np.uint8)
    return Image.fromarray(arr, mode="L")


class TestDifferenceHash:
    def test_strictly_ascending_row_hashes_to_all_ones(self):
        assert difference_hash(_gradient_image(ascending=True)) == 2**64 - 1

    def test_strictly_descending_row_hashes_to_all_zeros(self):
        """Negative twin: every comparison flips from True to False."""
        assert difference_hash(_gradient_image(ascending=False)) == 0

    def test_identical_images_hash_identically(self):
        """Byte-identical frames -> the same hash (hamming distance 0 via
        hamming_distance, covered below) -- the near-duplicate case."""
        a = _gradient_image(ascending=True)
        b = _gradient_image(ascending=True)
        assert difference_hash(a) == difference_hash(b)

    def test_resize_from_a_larger_image_still_produces_a_valid_64_bit_hash(self):
        """Not just the identity-resize fixture above: a real-sized (100x80)
        image must still resize down and produce an in-range hash."""
        arr = np.tile(np.linspace(0, 255, 100, dtype=np.uint8), (80, 1))
        value = difference_hash(Image.fromarray(arr, mode="L"))
        assert 0 <= value <= 2**64 - 1


class TestHammingDistance:
    def test_identical_values_have_zero_distance(self):
        assert hamming_distance(0b1010, 0b1010) == 0

    def test_known_bit_difference_is_counted_exactly(self):
        """0b1010 ^ 0b0110 = 0b1100 -> exactly 2 differing bits."""
        assert hamming_distance(0b1010, 0b0110) == 2

    def test_maximally_different_dhashes_from_the_gradient_fixture(self):
        """Genuinely different frames (ascending vs descending gradient,
        see TestDifferenceHash) are maximally far apart: every one of the 64
        bits differs."""
        a = difference_hash(_gradient_image(ascending=True))
        b = difference_hash(_gradient_image(ascending=False))
        assert hamming_distance(a, b) == 64


# --------------------------------------------------------------------- _build_verdicts

class TestBuildVerdicts:
    def test_empty_input_returns_empty_list(self):
        assert _build_verdicts({}) == []

    def test_exact_verdicts_for_one_of_each_case(self):
        """One synthetic folder of 5 already-scored assets, each isolating
        exactly one condition (sharp+clean, blurry, underexposed,
        overexposed, near-duplicate-of-the-first), asserted as one exact
        list -- reasons text, verdict, and near_duplicate_of all pinned."""
        file_metrics = {
            "A.NEF": {"sharpness": 100.0, "dhash": 0x0,
                      "shadow_clipped_fraction": 0.0, "highlight_clipped_fraction": 0.0},
            "B.NEF": {"sharpness": 1.0, "dhash": 0xFF,
                      "shadow_clipped_fraction": 0.0, "highlight_clipped_fraction": 0.0},
            "C.NEF": {"sharpness": 50.0, "dhash": 0xFF00,
                      "shadow_clipped_fraction": 0.5, "highlight_clipped_fraction": 0.0},
            "D.NEF": {"sharpness": 50.0, "dhash": 0xFF0000,
                      "shadow_clipped_fraction": 0.0, "highlight_clipped_fraction": 0.5},
            # Same dhash as A.NEF -> hamming distance 0 -> paired with A, the
            # earlier file in iteration order.
            "E.NEF": {"sharpness": 50.0, "dhash": 0x0,
                      "shadow_clipped_fraction": 0.0, "highlight_clipped_fraction": 0.0},
        }
        # median of [100, 1, 50, 50, 50] = 50.0; blur cutoff = 50.0 * 0.35 = 17.5

        result = _build_verdicts(file_metrics)

        assert result == [
            {"file": "A.NEF", "verdict": "keep_candidate",
             "reasons": ["sharp, well-exposed, and not a near-duplicate"],
             "sharpness": 100.0, "shadow_clipped_fraction": 0.0,
             "highlight_clipped_fraction": 0.0, "near_duplicate_of": None},
            {"file": "B.NEF", "verdict": "cull_candidate",
             "reasons": ["blurry: sharpness 1.00 vs folder median 50.00"],
             "sharpness": 1.0, "shadow_clipped_fraction": 0.0,
             "highlight_clipped_fraction": 0.0, "near_duplicate_of": None},
            {"file": "C.NEF", "verdict": "cull_candidate",
             "reasons": ["underexposed: 50% clipped shadows"],
             "sharpness": 50.0, "shadow_clipped_fraction": 0.5,
             "highlight_clipped_fraction": 0.0, "near_duplicate_of": None},
            {"file": "D.NEF", "verdict": "cull_candidate",
             "reasons": ["overexposed: 50% clipped highlights"],
             "sharpness": 50.0, "shadow_clipped_fraction": 0.0,
             "highlight_clipped_fraction": 0.5, "near_duplicate_of": None},
            {"file": "E.NEF", "verdict": "cull_candidate",
             "reasons": ["near-duplicate of A.NEF"],
             "sharpness": 50.0, "shadow_clipped_fraction": 0.0,
             "highlight_clipped_fraction": 0.0, "near_duplicate_of": "A.NEF"},
        ]

    def test_blur_cutoff_boundary_is_exclusive(self):
        """Boundary case: sharpness exactly at median*0.35 is NOT blurry
        (strict less-than); one float below it is. Two files so the median
        is driven by a known second value."""
        file_metrics = {
            "HIGH.NEF": {"sharpness": 100.0, "dhash": 0x0,
                         "shadow_clipped_fraction": 0.0, "highlight_clipped_fraction": 0.0},
            "AT_CUTOFF.NEF": {"sharpness": 17.5, "dhash": 0xFF,
                              "shadow_clipped_fraction": 0.0, "highlight_clipped_fraction": 0.0},
        }
        # median([100.0, 17.5]) = 58.75; cutoff = 58.75 * 0.35 = 20.5625 --
        # recompute with a cutoff value tied exactly to the median instead,
        # by using two equal highs so the median itself is unambiguous:
        file_metrics = {
            "HIGH_A.NEF": {"sharpness": 100.0, "dhash": 0x0,
                           "shadow_clipped_fraction": 0.0, "highlight_clipped_fraction": 0.0},
            "HIGH_B.NEF": {"sharpness": 100.0, "dhash": 0xFF,
                           "shadow_clipped_fraction": 0.0, "highlight_clipped_fraction": 0.0},
            "AT_CUTOFF.NEF": {"sharpness": 35.0, "dhash": 0xFF00,
                              "shadow_clipped_fraction": 0.0, "highlight_clipped_fraction": 0.0},
        }
        # median([100, 100, 35]) = 100.0; cutoff = 100.0 * 0.35 = 35.0 exactly.

        result = _build_verdicts(file_metrics)
        at_cutoff = next(v for v in result if v["file"] == "AT_CUTOFF.NEF")
        assert at_cutoff["verdict"] == "keep_candidate"
        assert at_cutoff["reasons"] == ["sharp, well-exposed, and not a near-duplicate"]

    def test_just_below_blur_cutoff_is_flagged(self):
        """Negative twin to the boundary test above, same fixture, one float
        below the cutoff."""
        file_metrics = {
            "HIGH_A.NEF": {"sharpness": 100.0, "dhash": 0x0,
                           "shadow_clipped_fraction": 0.0, "highlight_clipped_fraction": 0.0},
            "HIGH_B.NEF": {"sharpness": 100.0, "dhash": 0xFF,
                           "shadow_clipped_fraction": 0.0, "highlight_clipped_fraction": 0.0},
            "BELOW_CUTOFF.NEF": {"sharpness": 34.999, "dhash": 0xFF00,
                                 "shadow_clipped_fraction": 0.0, "highlight_clipped_fraction": 0.0},
        }
        result = _build_verdicts(file_metrics)
        below = next(v for v in result if v["file"] == "BELOW_CUTOFF.NEF")
        assert below["verdict"] == "cull_candidate"
        assert below["reasons"] == ["blurry: sharpness 35.00 vs folder median 100.00"]


# --------------------------------------------------------------------- suggest_culling

def _stub_metrics_by_name(mapping: dict[str, dict]):
    def fake(raw_path: Path) -> dict:
        return mapping[Path(raw_path).name]
    return fake


class TestSuggestCulling:
    def _seed(self, tmp_path):
        ensure_layout(tmp_path)
        assets = [
            Asset(file="source/raw/DSC_INGESTED_1.NEF", role="original", status="ingested", selected=False),
            Asset(file="source/raw/DSC_INGESTED_2.NEF", role="original", status="ingested", selected=False),
            # Must be excluded: already decided.
            Asset(file="source/raw/DSC_ALREADY_KEPT.NEF", role="original", status="kept", selected=True),
            Asset(file="source/raw/DSC_ALREADY_REJECTED.NEF", role="original", status="rejected", selected=False),
            # Must be excluded: not an "original" raw.
            Asset(file="review/DSC_INGESTED_1.jpg", role="review", status="ingested", selected=None),
        ]
        save_assets(tmp_path, assets)
        return assets

    def test_only_scores_ingested_original_raws(self, tmp_path, monkeypatch):
        self._seed(tmp_path)
        seen_files: list[str] = []

        def fake_compute(raw_path: Path) -> dict:
            seen_files.append(Path(raw_path).name)
            return {"sharpness": 50.0, "dhash": hash(str(raw_path)) & 0xFF,
                    "shadow_clipped_fraction": 0.0, "highlight_clipped_fraction": 0.0}

        monkeypatch.setattr(quality, "_compute_asset_metrics", fake_compute)

        suggest_culling(tmp_path)

        assert sorted(seen_files) == ["DSC_INGESTED_1.NEF", "DSC_INGESTED_2.NEF"]

    def test_writes_exact_result_to_presort_suggestions_json_and_returns_it(self, tmp_path, monkeypatch):
        self._seed(tmp_path)
        monkeypatch.setattr(quality, "_compute_asset_metrics", _stub_metrics_by_name({
            "DSC_INGESTED_1.NEF": {"sharpness": 100.0, "dhash": 0x0,
                                   "shadow_clipped_fraction": 0.0, "highlight_clipped_fraction": 0.0},
            "DSC_INGESTED_2.NEF": {"sharpness": 1.0, "dhash": 0xFF,
                                   "shadow_clipped_fraction": 0.0, "highlight_clipped_fraction": 0.0},
        }))

        result = suggest_culling(tmp_path)

        expected = {
            "folder": str(tmp_path),
            "assets": [
                {"file": "source/raw/DSC_INGESTED_1.NEF", "verdict": "keep_candidate",
                 "reasons": ["sharp, well-exposed, and not a near-duplicate"],
                 "sharpness": 100.0, "shadow_clipped_fraction": 0.0,
                 "highlight_clipped_fraction": 0.0, "near_duplicate_of": None},
                {"file": "source/raw/DSC_INGESTED_2.NEF", "verdict": "cull_candidate",
                 "reasons": ["blurry: sharpness 1.00 vs folder median 50.50"],
                 "sharpness": 1.0, "shadow_clipped_fraction": 0.0,
                 "highlight_clipped_fraction": 0.0, "near_duplicate_of": None},
            ],
            "summary": {"total": 2, "keep_candidates": 1, "cull_candidates": 1},
        }
        assert result == expected
        written_path = tmp_path / PRESORT_FILENAME
        assert written_path.exists()
        assert read_json(written_path) == expected

    def test_never_mutates_assets_jsonl(self, tmp_path, monkeypatch):
        """The core "pre-sort, do not cull yourself" property: assets.jsonl
        must be byte-identical before and after -- not just unchanged in the
        fields suggest_culling happens to look at."""
        self._seed(tmp_path)
        monkeypatch.setattr(quality, "_compute_asset_metrics", _stub_metrics_by_name({
            "DSC_INGESTED_1.NEF": {"sharpness": 100.0, "dhash": 0x0,
                                   "shadow_clipped_fraction": 0.0, "highlight_clipped_fraction": 0.0},
            "DSC_INGESTED_2.NEF": {"sharpness": 1.0, "dhash": 0xFF,
                                   "shadow_clipped_fraction": 0.0, "highlight_clipped_fraction": 0.0},
        }))
        assets_path = tmp_path / "metadata" / "assets.jsonl"
        before_text = assets_path.read_text(encoding="utf-8")
        before_assets = scan_assets(tmp_path)

        suggest_culling(tmp_path)

        assert assets_path.read_text(encoding="utf-8") == before_text
        assert scan_assets(tmp_path) == before_assets

    def test_never_calls_set_asset_status_or_delete_rejected_raws(self, tmp_path, monkeypatch):
        """Rule 4, and the single most important test given the owner's
        literal 'do not cull yourself' instruction: recording stubs on the
        exact mutating functions named in the spec, proven empty after a
        real (stub-decoded) suggest_culling() run. quality.py's own import
        block never imports either name (see the module docstring), so this
        patches them where they live, which is where a future regression
        that added such a call would have to reach them from."""
        self._seed(tmp_path)
        monkeypatch.setattr(quality, "_compute_asset_metrics", _stub_metrics_by_name({
            "DSC_INGESTED_1.NEF": {"sharpness": 50.0, "dhash": 0x0,
                                   "shadow_clipped_fraction": 0.0, "highlight_clipped_fraction": 0.0},
            "DSC_INGESTED_2.NEF": {"sharpness": 50.0, "dhash": 0xFF,
                                   "shadow_clipped_fraction": 0.0, "highlight_clipped_fraction": 0.0},
        }))
        # Structural proof first, on the unpatched module: the import block
        # never names either function, so there is no call site for them to
        # begin with.
        assert not hasattr(quality, "set_asset_status")
        assert not hasattr(quality, "delete_rejected_raws")

        set_status_calls: list[tuple] = []
        delete_calls: list[tuple] = []
        # Patched in both places a call could resolve from: the defining
        # module (catches a qualified `pipeline.set_asset_status(...)` call)
        # and quality's own module namespace, raising=False so the patch
        # applies whether or not quality.py currently imports the name --
        # if a regression adds `from .pipeline import set_asset_status` to
        # quality.py, that bare-name call resolves through quality.py's own
        # module globals at call time, which this second patch overrides
        # regardless of when the name was bound there.
        monkeypatch.setattr("bio_automation.pipeline.set_asset_status",
                             lambda *a, **k: set_status_calls.append((a, k)))
        monkeypatch.setattr("bio_automation.pipeline.delete_rejected_raws",
                             lambda *a, **k: delete_calls.append(a))
        monkeypatch.setattr(quality, "set_asset_status",
                             lambda *a, **k: set_status_calls.append((a, k)), raising=False)
        monkeypatch.setattr(quality, "delete_rejected_raws",
                             lambda *a, **k: delete_calls.append(a), raising=False)

        suggest_culling(tmp_path)

        assert set_status_calls == []
        assert delete_calls == []

    def test_empty_folder_writes_empty_summary_without_erroring(self, tmp_path):
        """No stub needed -- there are no candidates, so _compute_asset_metrics
        is never reached at all."""
        ensure_layout(tmp_path)
        save_assets(tmp_path, [])

        result = suggest_culling(tmp_path)

        assert result == {
            "folder": str(tmp_path), "assets": [],
            "summary": {"total": 0, "keep_candidates": 0, "cull_candidates": 0},
        }
