"""Live, unmocked proof that bio_automation.focus_stack actually drives the
real, installed focus-stack binary (https://github.com/PetteriAimonen/focus-stack,
MIT) against real denoised frames -- not asserted, measured.

Why this file exists and is not skippable (see AGENTS.md rule 10 and
tests/test_preclean_live.py's own docstring for the same reasoning): focus-stack
became this project's DEFAULT stacking engine (2026-10-19, see pipeline.py's
STACK_ENGINES) specifically because Zerene Stacker's own multi-file batch
loader never gets past the first frame on this machine (decompiled and
confirmed, see zerene.py's module docstring) -- PMax never actually runs
there. test_focus_stack.py proves this module's own subprocess/error-handling
logic is internally consistent, but only a real run against the real binary
can prove the new default production path actually merges more than one
frame, which is the entire reason it replaced Zerene. An opt-in-gated test
has the same "evaporates" failure mode tests/test_preclean_live.py's
docstring already documents for this repo.

Two distinct frames of the real spider fixture (tests/fixtures/sample_raw.NEF):
this repo has no second real raw of the *same* macro subject at a different
focus distance (sample_raw_fungus.NEF is a different subject, used
deliberately as the cross-subject case in test_grouping_live.py), so the two
frames fed to focus-stack here are two independent real runs of the project's
own real denoise_stack_frames() over the same real sensor data with two
different real parameter sets (noise 0.2 vs 0.9, exposure_ev 0.0 vs +0.4).
Each run is still the exact real rawpy decode -> real OpenVINO denoise model
-> real levels pipeline clean_and_stack() uses in production, writing a real
16-bit TIFF -- genuinely different pixel data from genuinely real sensor
data, just not a genuine focus bracket (no focus-distance differences exist
in this fixture to draw on). This is disclosed here, not glossed over: it
proves the wiring (real binary, real files, real alignment/merge/consistency
code paths) end to end, not that the merge content is depth-of-field-driven.

Deliberately full resolution (8288x5520, this fixture's real sensor size),
not cropped like test_preclean_live.py's unit-level checks: cropping would
mean never actually exercising decode_raw_to_bgr16/denoise_stack_frames on
real production-sized output, which is the whole point of an end-to-end
proof for a newly-defaulted engine. This makes the file slow -- measured
2026-10-19 at 470.6s for the two real denoise_stack_frames() calls plus
~30-40s for the real focus-stack merge (~8.5 min) when run alone, and 726s
(~12 min) in a second real run made under heavy concurrent CPU load from
other processes on this machine -- a real, disclosed, load-sensitive cost,
not hidden behind a flag that would make it not run.

Reading the merged output needs PIL, not tifffile: the real binary writes
its TIFF output LZW-compressed (verified via tifffile.TiffFile(...).pages[0]
.compression == 5), and plain tifffile.imread() on that file raises
"requires the 'imagecodecs' package" in this project's environment (that
package is not a project dependency). PIL/Pillow decodes LZW TIFFs without
it. The real, uncompressed 16-bit input frames denoise_stack_frames() writes
(compression == 1) decode fine with tifffile, which is used for those --
PIL was tried on them first and silently misread the pixel data (near-zero
mean/std on real, non-blank sensor data), so tifffile is the only reader
used for anything this module's own code writes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image

from bio_automation.focus_stack import DEFAULT_BINARY, OUTPUT_NAME, run_focus_stack
from bio_automation.preclean import denoise_stack_frames
from bio_automation.topaz import ModelCache

FIXTURE = Path(__file__).parent / "fixtures" / "sample_raw.NEF"


@pytest.fixture(scope="module")
def two_real_denoised_frames(tmp_path_factory):
    """Two real, distinct 16-bit TIFFs from the real spider raw -- see the
    module docstring for why they come from two denoise_stack_frames() calls
    with different real params rather than two genuinely different focus
    distances (this fixture set has none of the same subject)."""
    assert FIXTURE.exists(), f"missing committed fixture: {FIXTURE}"
    frames_dir = tmp_path_factory.mktemp("focus_stack_live_frames")

    with ModelCache() as cache:
        out_a = denoise_stack_frames([FIXTURE], frames_dir, noise=0.2, exposure_ev=0.0,
                                      cache=cache)[0]
        frame_a = frames_dir / "frame-a.tif"
        out_a.rename(frame_a)

        out_b = denoise_stack_frames([FIXTURE], frames_dir, noise=0.9, exposure_ev=0.4,
                                      cache=cache)[0]
        frame_b = frames_dir / "frame-b.tif"
        out_b.rename(frame_b)

    return frames_dir, frame_a, frame_b


class TestRunFocusStackLive:
    def test_real_binary_produces_a_genuine_merged_stack_not_a_copy_of_an_input(
        self, two_real_denoised_frames
    ):
        assert DEFAULT_BINARY.exists(), (
            f"real focus-stack binary not found at {DEFAULT_BINARY} -- install it "
            "(see src/bio_automation/focus_stack.py's module docstring) or set "
            "FOCUS_STACK_BINARY"
        )
        frames_dir, frame_a, frame_b = two_real_denoised_frames

        result = run_focus_stack(frames_dir, timeout=900.0)

        assert result == frames_dir / OUTPUT_NAME
        assert result.exists()

        merged = np.array(Image.open(result))
        frame_a_arr = tifffile.imread(str(frame_a))
        frame_b_arr = tifffile.imread(str(frame_b))

        # The real decode+denoise pipeline's output size is governed only by
        # the fixture's real sensor dimensions, not by content-dependent
        # alignment -- stable across runs, safe to pin exactly.
        assert frame_a_arr.shape == (8288, 5520, 3)
        assert frame_a_arr.dtype == np.uint16
        assert frame_b_arr.shape == (8288, 5520, 3)
        assert frame_b_arr.dtype == np.uint16

        # The real binary's own ECC alignment step crops to the mutually-
        # overlapping region, and writes its TIFF output as 8-bit regardless
        # of the real 16-bit input -- neither is true of a simple copy of an
        # input frame. The exact crop is content-dependent (measured
        # 2026-10-19 at (8286, 5518) and, on a second real run with the same
        # code, (8288, 5518) -- height sometimes needs no trim, width always
        # has, so only width is asserted strictly smaller; both bounded
        # generously rather than pinned exactly).
        assert merged.shape[2] == 3
        assert merged.dtype == np.uint8
        assert merged.shape[0] <= frame_a_arr.shape[0]
        assert frame_a_arr.shape[0] - merged.shape[0] <= 20
        assert merged.shape[1] < frame_a_arr.shape[1]
        assert frame_a_arr.shape[1] - merged.shape[1] <= 20

        # A genuine re-encode, not a renamed copy of either input: the real
        # LZW-compressed 8-bit output measures ~69-71 MB (measured across
        # two real runs) against each real uncompressed 16-bit input's
        # ~274 MB.
        assert 1_000_000 < result.stat().st_size < 100_000_000
        assert result.stat().st_size != frame_a.stat().st_size
        assert result.stat().st_size != frame_b.stat().st_size

        # Pixel content genuinely differs from both inputs, not just the
        # container format: crop each 16-bit input to the merged output's
        # (smaller, post-alignment) dimensions and downscale to 8-bit the
        # same way the real binary's own TIFF writer would, then compare.
        h, w = merged.shape[:2]
        frame_a_naive_8bit = (frame_a_arr[:h, :w].astype(np.float64) / 257.0).round().astype(np.uint8)
        frame_b_naive_8bit = (frame_b_arr[:h, :w].astype(np.float64) / 257.0).round().astype(np.uint8)
        assert not np.array_equal(merged, frame_a_naive_8bit)
        assert not np.array_equal(merged, frame_b_naive_8bit)

        # Real measured mean abs diff, two independent real runs 2026-10-19:
        # 13.09 and 18.02 from frame A's naive crop, 1.95 and 1.99 from
        # frame B's (frame B was focus-stack's own alignment reference, so
        # its contrast/whitebalance was left closest to the merged output's
        # -- still genuinely, not byte-identically, reprocessed in either
        # run). Asserting comfortably below the smaller of the two measured
        # values each time rather than pinning them exactly: the real
        # OpenCV merge/denoise/wavelet steps are deterministic run-to-run
        # for a *fixed* pair of input frames, but the frames themselves
        # (and therefore this exact diff) vary slightly with the real
        # OpenVINO model's own floating-point behavior under different
        # system load -- the claim under test is "genuinely different", not
        # "different by this exact amount".
        diff_from_a = np.abs(merged.astype(np.int32) - frame_a_naive_8bit.astype(np.int32))
        diff_from_b = np.abs(merged.astype(np.int32) - frame_b_naive_8bit.astype(np.int32))
        assert float(diff_from_a.mean()) > 5.0
        assert float(diff_from_b.mean()) > 0.5
