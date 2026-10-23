"""bio-sr2x-v1: the owner's own 2x crop-upscale model (TinySR2x), trained on real
Stock-ready photos, AWS g4dn.xlarge, 10,684 steps. Held-out measurement (80 images,
never trained on): +5.01 dB PSNR vs plain bicubic. Topaz's ghqv2 measured -5 dB vs
bicubic on the same kind of real-frame test (tools/topaz_lab), so this is not an
incremental tweak -- it is the one 2x model in this codebase shown to help at all.

No GAN/adversarial loss in training (see tools/sr_train/model.py docstring): it does
not invent detail, so it stays safe for iNat-adjacent crops, not just Stock sale copies.

Selected via settings.UPSCALE_MODEL ("custom"); never swapped in silently -- see
topaz.enhance()'s mode="upscale" branch and settings.UPSCALE_MODEL's own comment.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import cuda_dlls, ort_providers, settings
from .topaz import _run_tiled

_MISSING_CUDA_DIRS = cuda_dlls.add_cuda_dll_dirs()       # before onnxruntime loads its CUDA provider
import onnxruntime as ort  # noqa: E402

CACHE_DIR = settings.MODEL_CACHE_DIR
_MODEL_DIR = CACHE_DIR / "bio-sr2x-v1"
_ONNX_NAME = "bio-sr2x-v1-fp32-128x128-2x.onnx"

_session: ort.InferenceSession | None = None  # one model: a lazy singleton, not topaz.ModelCache's LRU


def load_session(cache_dir: Path = CACHE_DIR) -> ort.InferenceSession:
    """Load bio-sr2x-v1 on CUDA, or raise: GPU only, never a silent CPU fallback."""
    model_path = Path(cache_dir) / "bio-sr2x-v1" / _ONNX_NAME
    if not model_path.exists():
        raise FileNotFoundError(f"bio-sr2x-v1 model not found: {model_path}")
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if _MISSING_CUDA_DIRS:
        raise RuntimeError(f"sr_model: CUDA DLL folders not found: {[str(d) for d in _MISSING_CUDA_DIRS]} (set BIO_CUDA_DLL_DIRS)")
    return ort_providers.open_session(ort, model_path, sess_options=opts, label="bio-sr2x-v1", require_cuda=True)


def get_session() -> ort.InferenceSession:
    """The process-wide session, created on first use."""
    global _session
    if _session is None:
        _session = load_session()
    return _session


def upscale_2x(img_bgr: np.ndarray, session: ort.InferenceSession | None = None) -> np.ndarray:
    """2x upscale via bio-sr2x-v1, 128x128 tiles -- same tiling contract as
    topaz.upscale_2x (ghqv2), so callers can swap between them without other changes."""
    session = session or get_session()
    input_name = session.get_inputs()[0].name

    img_rgb = img_bgr[:, :, ::-1].astype(np.float32) / 255.0
    img_chw = np.transpose(img_rgb, (2, 0, 1))

    def feed(tile: np.ndarray, _sess) -> np.ndarray:
        return session.run(None, {input_name: tile[np.newaxis].astype(np.float32)})[0][0]

    result_chw = _run_tiled(session, img_chw, 128, 128, 16, 2, feed)

    result_rgb = np.transpose(result_chw, (1, 2, 0))
    result_bgr = result_rgb[:, :, ::-1]
    return (result_bgr * 255.0).round().clip(0, 255).astype(np.uint8)
