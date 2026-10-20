"""Live, unmocked proof that bio_automation.quality's raw-decode code path
and suggest_culling() actually work against real raw files -- not gated
behind an opt-in env var or marker, same rationale and pattern as
tests/test_preclean_live.py (see its module docstring): a check that doesn't
run by default doesn't run. Uses both committed NEF fixtures
(tests/fixtures/sample_raw.NEF, sample_raw_fungus.NEF) so suggest_culling()
runs over a real multi-asset folder, not a single file.

half_size decode keeps this fast (~1-2s per raw, measured) -- see
quality._decode_preview_rgb's docstring for why full resolution is
unnecessary for these heuristics.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from bio_automation.models import Asset
from bio_automation.pipeline import ensure_layout, save_assets
from bio_automation.quality import (
    PRESORT_FILENAME,
    _compute_asset_metrics,
    _decode_preview_rgb,
    suggest_culling,
)

FIXTURE_SPIDER = Path(__file__).parent / "fixtures" / "sample_raw.NEF"
FIXTURE_FUNGUS = Path(__file__).parent / "fixtures" / "sample_raw_fungus.NEF"


class TestDecodePreviewRgbLive:
    def test_real_nef_decodes_to_sane_small_rgb(self):
        assert FIXTURE_SPIDER.exists(), f"missing committed fixture: {FIXTURE_SPIDER}"
        rgb = _decode_preview_rgb(FIXTURE_SPIDER)

        assert rgb.dtype == np.uint8
        assert rgb.ndim == 3
        assert rgb.shape[2] == 3
        assert max(rgb.shape[:2]) <= 768  # downsampled, see module docstring
        # a real daylight macro exposure: not blank, not saturated, real variance
        assert rgb.min() < rgb.max()
        assert float(rgb.std()) > 5.0

    def test_real_decode_is_reproducible(self):
        first = _decode_preview_rgb(FIXTURE_SPIDER)
        second = _decode_preview_rgb(FIXTURE_SPIDER)
        assert np.array_equal(first, second)

    def test_different_real_fixtures_decode_to_different_images(self):
        """Positive-twin sanity check that two distinct real raws actually
        produce distinct pixel data, not two copies of the same buffer."""
        spider = _decode_preview_rgb(FIXTURE_SPIDER)
        fungus = _decode_preview_rgb(FIXTURE_FUNGUS)
        assert spider.shape != fungus.shape or not np.array_equal(spider, fungus)


class TestComputeAssetMetricsLive:
    def test_real_fixture_yields_in_range_metrics(self):
        metrics = _compute_asset_metrics(FIXTURE_SPIDER)

        assert set(metrics) == {"sharpness", "dhash", "shadow_clipped_fraction",
                                 "highlight_clipped_fraction"}
        assert metrics["sharpness"] > 0.0  # real sensor detail, not a flat frame
        assert 0 <= metrics["dhash"] <= 2**64 - 1
        assert 0.0 <= metrics["shadow_clipped_fraction"] <= 1.0
        assert 0.0 <= metrics["highlight_clipped_fraction"] <= 1.0


class TestSuggestCullingLive:
    def _build_observation(self, tmp_path) -> Path:
        folder = tmp_path / "obs-live"
        ensure_layout(folder)
        raw_dir = folder / "source" / "raw"
        shutil.copy2(FIXTURE_SPIDER, raw_dir / FIXTURE_SPIDER.name)
        shutil.copy2(FIXTURE_FUNGUS, raw_dir / FIXTURE_FUNGUS.name)
        save_assets(folder, [
            Asset(file=f"source/raw/{FIXTURE_SPIDER.name}", role="original",
                  status="ingested", selected=False),
            Asset(file=f"source/raw/{FIXTURE_FUNGUS.name}", role="original",
                  status="ingested", selected=False),
        ])
        return folder

    def test_runs_against_real_raws_without_erroring_and_produces_a_real_shaped_result(self, tmp_path):
        folder = self._build_observation(tmp_path)

        result = suggest_culling(folder)

        assert result["folder"] == str(folder)
        assert result["summary"]["total"] == 2
        assert result["summary"]["keep_candidates"] + result["summary"]["cull_candidates"] == 2
        assert {a["file"] for a in result["assets"]} == {
            f"source/raw/{FIXTURE_SPIDER.name}", f"source/raw/{FIXTURE_FUNGUS.name}",
        }
        for asset_result in result["assets"]:
            assert asset_result["verdict"] in ("keep_candidate", "cull_candidate")
            assert asset_result["reasons"], "every verdict must carry at least one reason"
            assert isinstance(asset_result["sharpness"], float)

        written = folder / PRESORT_FILENAME
        assert written.exists()

    def test_does_not_touch_assets_jsonl(self, tmp_path):
        """Same property as the hermetic test, proven once more end to end
        against real files."""
        folder = self._build_observation(tmp_path)
        assets_path = folder / "metadata" / "assets.jsonl"
        before = assets_path.read_text(encoding="utf-8")

        suggest_culling(folder)

        assert assets_path.read_text(encoding="utf-8") == before
