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

import shutil
from pathlib import Path

import numpy as np
import rawpy
import tifffile
from PIL import Image

from .topaz import MODELS_DIR, ModelCache, _estimate_sharpen_params, _run_tiled

SUBJECT_MASK_MODEL = "s_mask_l_flexible-v2-fp16-320x320-ov.tz2"


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


def subject_mask(img_chw_f32: np.ndarray, cache: ModelCache) -> np.ndarray:
    """Return a float32 [0,1] saliency mask, same H,W as *img_chw_f32*,
    from Topaz Photo AI's s_mask_l_flexible model (already installed
    locally for Photo AI's own masked-sharpen feature -- see
    sharpen-subject-masking-new.json in the same models directory -- not
    Meta's SAM; this codebase explicitly does not use the samenc/samdec
    pair also present there).

    The model takes a single 320x320 RGB tile (whole-frame context, not
    tiled like _sharpen_f32) and returns raw logits, not a [0,1] mask --
    confirmed live against a real frame: output range was approximately
    -9.9 to +2.8, not already bounded. A sigmoid is applied here before
    the mask is usable as a blend weight. The 320x320 output is resized
    back up to the input's full resolution with PIL's bilinear filter,
    which also supplies the mask's edge feathering for free -- no separate
    blur/feather step is applied.
    """
    _, h, w = img_chw_f32.shape
    hwc_u8 = (np.transpose(img_chw_f32, (1, 2, 0)) * 255.0).clip(0, 255).astype(np.uint8)
    small = Image.fromarray(hwc_u8).resize((320, 320), Image.BILINEAR)
    in320 = np.transpose(np.array(small, dtype=np.float32) / 255.0, (2, 0, 1))[np.newaxis]

    compiled = cache.load(MODELS_DIR / SUBJECT_MASK_MODEL)
    out = compiled({"netInput": in320})
    logits = next(iter(out.values()))[0, 0]
    sig = 1.0 / (1.0 + np.exp(-logits))

    mask_img = Image.fromarray((sig * 255.0).clip(0, 255).astype(np.uint8), mode="L")
    mask_full = mask_img.resize((w, h), Image.BILINEAR)
    return np.array(mask_full, dtype=np.float32) / 255.0


def apply_levels_masked(img_chw_f32: np.ndarray, mask_hw: np.ndarray,
                         exposure_ev: float, subject_boost_ev: float) -> np.ndarray:
    """Like apply_levels(), but blends two exposures by *mask_hw*: the
    normal (possibly highlight-protecting, negative) *exposure_ev* gain
    outside the subject, and exposure_ev + subject_boost_ev inside it --
    so a dark subject can be lifted without also lifting (and clipping)
    bright background/highlights the global exposure_ev exists to protect.

    subject_boost_ev=0.0 short-circuits to a bit-exact call to
    apply_levels() -- not just a blend that happens to be mathematically
    equal -- since float32's x*(1-m) + x*m is not bit-identical to x for
    an arbitrary mask (non-associative rounding), and the point of this
    being a documented identity is that the mask genuinely has zero effect
    when there's no boost to apply, not an effect smaller than 1 ULP.
    mask_hw must already be the image's full H,W (see subject_mask()) and
    is applied identically per RGB channel.
    """
    if subject_boost_ev == 0.0:
        return apply_levels(img_chw_f32, exposure_ev)
    global_result = apply_levels(img_chw_f32, exposure_ev)
    subject_result = apply_levels(img_chw_f32, exposure_ev + subject_boost_ev)
    mask = mask_hw[np.newaxis, :, :]  # broadcast over the 3 channels
    return np.clip(global_result * (1.0 - mask) + subject_result * mask, 0.0, 1.0)


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
                          subject_boost_ev: float = 0.0,
                          cache: ModelCache | None = None,
                          min_free_gb: float = 5.0) -> list[Path]:
    """Run the fixed pre-stack pipeline -- raw denoise, then levels -- over
    every raw in *raw_paths* with one shared (blur, noise, exposure_ev,
    subject mask) set, and write each result as a 16-bit RGB TIFF into
    *dest_dir*, named after the raw's stem. Returns the written paths in
    the same order as *raw_paths*. "Other processing" (sharpen/upscale) is
    not done here -- see module docstring.

    *subject_boost_ev* (default 0.0, no-op) adds an extra EV lift inside
    the subject only, on top of *exposure_ev*, via subject_mask()/
    apply_levels_masked() -- see their docstrings. Like blur/noise/
    exposure_ev, the mask is computed ONCE, from the same representative
    (middle-index) frame compute_fixed_params() uses, and reused for every
    frame: a mask that shifts frame to frame would reintroduce exactly the
    inconsistency this module exists to keep out of Zerene's cross-frame
    contrast comparison. Skipped entirely (no mask model call) when
    subject_boost_ev is 0.0, since it would be a no-op per
    apply_levels_masked()'s identity property anyway.

    Each written TIFF is one full-resolution 16-bit frame (hundreds of MB
    for a modern raw), so disk space is checked against *dest_dir* before
    any model loading or raw decoding happens, same as topaz.py's
    enhance() -- a doomed batch fails fast and cheap instead of filling the
    disk partway through.
    """
    raw_paths = [Path(p) for p in raw_paths]
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    free_gb = shutil.disk_usage(dest_dir).free / 1e9
    if free_gb < min_free_gb:
        raise OSError(
            f"Insufficient disk space: {free_gb:.1f} GB free < {min_free_gb:.1f} GB required"
        )

    own_cache = cache is None
    if own_cache:
        cache = ModelCache()

    try:
        blur, fixed_noise = compute_fixed_params(raw_paths, cache, noise=noise)

        mask_hw = None
        if subject_boost_ev != 0.0:
            representative = Path(raw_paths[len(raw_paths) // 2])
            rep_chw = _to_chw_rgb_f32(decode_raw_to_bgr16(representative))
            mask_hw = subject_mask(rep_chw, cache)

        written: list[Path] = []
        for raw_path in raw_paths:
            chw = _to_chw_rgb_f32(decode_raw_to_bgr16(raw_path))
            denoised_chw = _sharpen_f32(chw, cache, blur=blur, noise=fixed_noise)  # 1. raw denoise
            if mask_hw is not None:
                leveled_chw = apply_levels_masked(denoised_chw, mask_hw, exposure_ev,
                                                   subject_boost_ev)                # 2. levels (masked)
            else:
                leveled_chw = apply_levels(denoised_chw, exposure_ev)              # 2. levels
            result_hwc = np.transpose(leveled_chw, (1, 2, 0))
            result_u16 = (result_hwc * 65535.0).round().clip(0, 65535).astype(np.uint16)
            out_path = dest_dir / f"{raw_path.stem}.tif"
            tifffile.imwrite(str(out_path), result_u16)
            written.append(out_path)
        return written
    finally:
        if own_cache:
            cache.__exit__(None, None, None)
