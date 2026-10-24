"""Standalone port of bio_automation/preclean.py's subject_mask() + refine_mask_by_color()
(Topaz s_mask_l_flexible path), for comparing against SAM3 concept-prompted segmentation on
a real spider test fixture. Ported, not imported -- production preclean.py is read-only
reference (live-reloading tree, never edited by this session). Logic copied verbatim from
D:\\apps\\bio-automation\\src\\bio_automation\\preclean.py (lines ~141-287), 2026-10-23.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

MODELS_DIR = Path(r"C:\ProgramData\Topaz Labs LLC\Topaz Photo AI\models")
CACHE_DIR = Path(r"C:\TopazModels")
SUBJECT_MASK_MODEL = "s_mask_l_flexible-v2-fp16-320x320-ov.tz2"


def _rgb_to_hue_sat(img_chw_f32: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r, g, b = img_chw_f32[0], img_chw_f32[1], img_chw_f32[2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc
    s = np.where(maxc > 0, delta / np.where(maxc > 0, maxc, 1), 0.0)
    rc = np.where(delta > 0, (maxc - r) / np.where(delta > 0, delta, 1), 0.0)
    gc = np.where(delta > 0, (maxc - g) / np.where(delta > 0, delta, 1), 0.0)
    bc = np.where(delta > 0, (maxc - b) / np.where(delta > 0, delta, 1), 0.0)
    is_r = (maxc == r) & (delta > 0)
    is_g = (maxc == g) & (delta > 0) & ~is_r
    is_b = (maxc == b) & (delta > 0) & ~is_r & ~is_g
    h = np.zeros_like(r)
    h = np.where(is_r, bc - gc, h)
    h = np.where(is_g, 2.0 + rc - bc, h)
    h = np.where(is_b, 4.0 + gc - rc, h)
    h = (h / 6.0) % 1.0
    return h, s


def _hue_distance(h1, h2):
    d = np.abs(h1 - h2)
    return np.minimum(d, 1.0 - d)


def _local_background_hue(img_chw_f32, prob, roi_prob_threshold=0.3, bg_prob_threshold=0.05, pad=400):
    _, H, W = img_chw_f32.shape
    roi = prob > roi_prob_threshold
    if not roi.any():
        return None
    ys, xs = np.where(roi)
    y0, y1 = max(0, int(ys.min()) - pad), min(H, int(ys.max()) + pad)
    x0, x1 = max(0, int(xs.min()) - pad), min(W, int(xs.max()) + pad)
    region_img = img_chw_f32[:, y0:y1, x0:x1]
    region_prob = prob[y0:y1, x0:x1]
    bg_mask = region_prob < bg_prob_threshold
    if bg_mask.sum() < 200:
        return None
    hue, _sat = _rgb_to_hue_sat(region_img)
    return float(hue[bg_mask].mean())


def refine_mask_by_color(img_chw_f32, prob, hue_ramp_low_deg=6.0, hue_ramp_high_deg=20.0, blur_radius=10.0):
    ref_hue = _local_background_hue(img_chw_f32, prob)
    if ref_hue is None:
        return prob
    hue, _sat = _rgb_to_hue_sat(img_chw_f32)
    hue_dist_deg = _hue_distance(hue, ref_hue) * 360.0
    soft_gate = np.clip((hue_dist_deg - hue_ramp_low_deg) / (hue_ramp_high_deg - hue_ramp_low_deg), 0.0, 1.0)
    gated = prob * soft_gate
    gate_img = Image.fromarray((gated * 255.0).clip(0, 255).astype(np.uint8), mode="L")
    gate_blurred = gate_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return np.array(gate_blurred, dtype=np.float32) / 255.0


class ModelCache:
    """Minimal port of topaz.py's ModelCache.load for a single already-cached model."""

    def __init__(self, device="CPU", cache_dir=CACHE_DIR):
        self._device = device
        self._cache_dir = Path(cache_dir)

    def load(self, tz_path: Path):
        import openvino as ov

        model_dir = self._cache_dir / tz_path.stem
        xmls = list(model_dir.glob("*.xml"))
        if not xmls:
            raise FileNotFoundError(
                f"No cached .xml for {tz_path.stem} under {model_dir} -- "
                f"this model has never been extracted (topaz.py's ModelCache "
                f"would decrypt it from {tz_path}, which this standalone port "
                f"deliberately does not do -- run it once via the real app/pipeline first)"
            )
        xml_path = xmls[0]
        bin_path = xml_path.with_suffix(".bin")
        core = ov.Core()
        model = core.read_model(str(xml_path), str(bin_path)) if bin_path.exists() else core.read_model(str(xml_path))
        return core.compile_model(model, self._device)


def subject_mask(img_chw_f32: np.ndarray, cache: ModelCache) -> np.ndarray:
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
    prob_full = np.array(mask_full, dtype=np.float32) / 255.0
    return refine_mask_by_color(img_chw_f32, prob_full)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--out-mask-png", type=Path, required=True)
    ap.add_argument("--out-stats-json", type=Path, required=True)
    args = ap.parse_args()

    img = Image.open(args.image).convert("RGB")
    img_chw_f32 = np.transpose(np.array(img, dtype=np.float32) / 255.0, (2, 0, 1))

    cache = ModelCache()
    prob = subject_mask(img_chw_f32, cache)

    Image.fromarray((prob * 255.0).clip(0, 255).astype(np.uint8), mode="L").save(args.out_mask_png)
    stats = {
        "model": SUBJECT_MASK_MODEL,
        "image": str(args.image),
        "image_size": [img.width, img.height],
        "mask_mean": float(prob.mean()),
        "mask_area_frac_gt_0.5": float((prob > 0.5).mean()),
    }
    args.out_stats_json.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
