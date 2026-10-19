"""Pre-stack cleanup: raw denoise, then levels, in a fixed order with one
fixed parameter set shared across the whole stack, before Zerene aligns/
merges the frames. "Other processing" (sharpen/upscale) stays a later,
separate stage -- the existing post-stack topaz.enhance()/lce pass run
once on the merged output, not per frame here.

Why fixed params, not per-frame auto-params: Zerene's PMax picks the
sharpest pixel at each location by comparing local contrast across
frames. If each frame's denoise strength varies -- topaz.sharpen()'s
default auto_params estimates (blur, noise) per image via a centre-crop
autoparam model -- the cleanup pass itself becomes a source of
frame-to-frame inconsistency that PMax reads as a real detail difference,
risking misaligned pixel picks. Computing (blur, noise) once from a
single representative frame and reusing it for every frame keeps that
signal consistent: noise compounds across a stack (more frames reading
the same sensor-noise floor differently makes PMax's contrast comparison
noisier, not sharper), so suppressing it uniformly before stacking is the
point; letting the suppression itself vary frame to frame would undo it.
The same reasoning applies to exposure: a frame-to-frame brightness
difference is exactly the kind of inconsistency this module exists to
avoid feeding into PMax, so the EV pull-down is a fixed dial, not
estimated per frame, and no_auto_bright=True is passed unconditionally so
LibRaw's own per-image auto-brightness never introduces that variance.

Why denoise before levels, not the reverse: the denoise model sees the
sensor's native exposure, which is where its noise statistics are best
behaved (shadow noise especially changes character once a gain is
applied). Levels (a fixed EV pull-down, to leave headroom under clipping
for specular highlights -- wet leaves, reflective exoskeletons -- that
would otherwise blow out and lose detail no post-stack recovery can
restore) is applied as a plain linear gain *after* denoising, to the
model's float32 output directly, not folded into rawpy's own exp_shift at
decode time.

Source is the camera raw (NEF/CR2/...), decoded via rawpy, not the
embedded preview -- a pre-stack cleanup pass that throws away raw detail
before denoising defeats its own purpose. Decode and inference both stay
in float32 [0,1] end to end (16-bit TIFF output, not 8-bit), since
topaz.sharpen()'s uint8 round-trip (/255 in, *255-clip-uint8 out) would
throw away exactly the precision a raw decode has over an 8-bit JPEG --
the reason to decode raw at all for this step.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rawpy
import tifffile

from .topaz import MODELS_DIR, ModelCache, _estimate_sharpen_params, _run_tiled


def ev_to_linear_gain(exposure_ev: float) -> float:
    """Convert an EV offset to a linear multiplier, clamped to rawpy's
    exp_shift range (0.25..8.0 -- 2-stop darken to 3-stop lighten) even
    though it's applied as a plain multiply here, not handed to rawpy, so
    the two code paths stay comparable."""
    return float(np.clip(2.0 ** exposure_ev, 0.25, 8.0))


def decode_raw_to_bgr16(raw_path: Path) -> np.ndarray:
    """Demosaic *raw_path* via rawpy at the sensor's native exposure (no
    exp_shift -- levels is a separate, later stage; see module docstring).

    no_auto_bright=True always, so LibRaw's own per-image auto-brightness
    never introduces frame-to-frame variance this module is trying to
    eliminate.

    Returns uint16 HxWx3 in BGR channel order, matching the channel
    convention topaz.py's functions use (they reverse to RGB internally).
    """
    with rawpy.imread(str(raw_path)) as raw:
        rgb16 = raw.postprocess(output_bps=16, no_auto_bright=True, use_camera_wb=True)
    return rgb16[:, :, ::-1]


def _to_chw_rgb_f32(img_bgr_hwc_u16: np.ndarray) -> np.ndarray:
    img_rgb = img_bgr_hwc_u16[:, :, ::-1].astype(np.float32) / 65535.0
    return np.transpose(img_rgb, (2, 0, 1))


def _sharpen_f32(img_chw_f32: np.ndarray, cache: ModelCache,
                  blur: float, noise: float) -> np.ndarray:
    """Run the slb lens-blur/denoise model on float32 [0,1] CHW input and
    return float32 [0,1] CHW output, skipping topaz.sharpen()'s uint8
    round-trip. Mirrors its tiling/feed logic exactly; kept separate
    rather than adding a dtype branch to sharpen() itself, which is
    covered by its own tests at uint8 and used elsewhere at that depth.
    """
    compiled = cache.load(MODELS_DIR / "slb-v1-fp32-512x512-ov.tz")
    blur_val = np.array([blur], dtype=np.float32)
    noise_val = np.array([noise], dtype=np.float32)

    def feed(tile: np.ndarray, mdl) -> np.ndarray:
        results = mdl({"netInput": tile[np.newaxis], "t_param": blur_val, "t_param1": noise_val})
        return next(iter(results.values()))[0]

    return _run_tiled(compiled, img_chw_f32, 512, 512, 64, 1, feed, "tl_unet/netOutput")


def apply_levels(img_chw_f32: np.ndarray, exposure_ev: float) -> np.ndarray:
    """Apply a fixed linear exposure gain (stage 2: levels) to float32
    [0,1] data already denoised by _sharpen_f32, clipped back into range.
    """
    return np.clip(img_chw_f32 * ev_to_linear_gain(exposure_ev), 0.0, 1.0)


def compute_fixed_params(raw_paths: list[Path], cache: ModelCache,
                          noise: float = 1.0) -> tuple[float, float]:
    """Estimate (blur, noise) once from the middle frame of *raw_paths* and
    return it for reuse across the whole stack.

    *noise* overrides the estimated noise value directly: denoise strength
    is the point of this pass, so it is a dial the caller sets (clamped to
    [0, 1], the model's trained range), not a per-frame estimate. *blur* --
    lens-blur sharpening strength -- still comes from the autoparam model,
    since that is a real optical property of the lens/aperture the whole
    stack shares, not an artifact of any one frame. Estimated at native
    exposure (levels hasn't been applied yet at this stage).
    """
    if not raw_paths:
        raise ValueError("compute_fixed_params requires at least one raw path")
    representative = Path(raw_paths[len(raw_paths) // 2])
    chw = _to_chw_rgb_f32(decode_raw_to_bgr16(representative))
    _, lensblur, _, _ = _estimate_sharpen_params(chw, cache)
    return (float(lensblur), float(np.clip(noise, 0.0, 1.0)))


def denoise_stack_frames(raw_paths: list[Path], dest_dir: Path,
                          noise: float = 1.0, exposure_ev: float = 0.0,
                          cache: ModelCache | None = None) -> list[Path]:
    """Run the fixed pre-stack pipeline -- raw denoise, then levels -- over
    every raw in *raw_paths* with one shared (blur, noise, exposure_ev)
    triple, and write each result as a 16-bit RGB TIFF into *dest_dir*,
    named after the raw's stem. Returns the written paths in the same
    order as *raw_paths*. "Other processing" (sharpen/upscale) is not done
    here -- see module docstring.
    """
    raw_paths = [Path(p) for p in raw_paths]
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    own_cache = cache is None
    if own_cache:
        cache = ModelCache()

    try:
        blur, fixed_noise = compute_fixed_params(raw_paths, cache, noise=noise)
        written: list[Path] = []
        for raw_path in raw_paths:
            chw = _to_chw_rgb_f32(decode_raw_to_bgr16(raw_path))
            denoised_chw = _sharpen_f32(chw, cache, blur=blur, noise=fixed_noise)  # 1. raw denoise
            leveled_chw = apply_levels(denoised_chw, exposure_ev)                  # 2. levels
            result_hwc = np.transpose(leveled_chw, (1, 2, 0))
            result_u16 = (result_hwc * 65535.0).round().clip(0, 65535).astype(np.uint16)
            out_path = dest_dir / f"{raw_path.stem}.tif"
            tifffile.imwrite(str(out_path), result_u16)
            written.append(out_path)
        return written
    finally:
        if own_cache:
            cache.__exit__(None, None, None)
