"""
DA-CLIP degradation-aware CLIP embeddings.

Outputs (content_vec[512], degradation_vec[512]) per image.
content_vec  — semantic content embedding (like standard CLIP)
degradation_vec — describes image quality degradation type and severity

Uses the extracted daclip-v2-fp32-224x224-1x-ox.onnx from C:\\TopazModels\\.
Requires: onnxruntime, numpy, Pillow  (no openvino needed for ONNX models).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from lce import cuda_dlls

from bio_automation import ort_providers, settings

_MISSING_CUDA_DIRS = cuda_dlls.add_cuda_dll_dirs(settings.CUDA_DLL_DIRS)  # before onnxruntime loads its CUDA provider
import onnxruntime as ort  # noqa: E402

CACHE_DIR = settings.MODEL_CACHE_DIR
_MODEL_DIR = CACHE_DIR / "daclip-v2-fp32-224x224-1x-ox"
_ONNX_NAME = "daclip-v2-fp32-224x224-1x.onnx"

# CLIP normalisation constants (RGB)
_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_STD  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

_SIZE = 224


def load_session(cache_dir: Path = CACHE_DIR) -> ort.InferenceSession:
    """Load the DA-CLIP ONNX model on CUDA, or raise: GPU only, never a silent CPU fallback.

    The CUDA device comes from BIO_ORT_CUDA_DEVICE (default 0); the CUDA DLL folders from BIO_CUDA_DLL_DIRS."""
    model_path = Path(cache_dir) / "daclip-v2-fp32-224x224-1x-ox" / _ONNX_NAME
    if not model_path.exists():
        raise FileNotFoundError(f"DA-CLIP model not found: {model_path}")
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if _MISSING_CUDA_DIRS:
        raise RuntimeError(f"daclip: CUDA DLL folders not found: {[str(d) for d in _MISSING_CUDA_DIRS]} (set BIO_CUDA_DLL_DIRS)")
    return ort_providers.open_session(ort, model_path, sess_options=opts, label="daclip", require_cuda=True)


def preprocess(img_bgr_uint8: np.ndarray) -> np.ndarray:
    """Resize and normalise a BGR uint8 HxWx3 array to [1,224,224,3] float32.

    Converts BGR→RGB, resizes to 224×224, scales to [0,1], then applies
    CLIP mean/std normalisation.  Returns shape [1,224,224,3].
    """
    # BGR → RGB via Pillow for high-quality Lanczos resize
    rgb = img_bgr_uint8[:, :, ::-1]  # flip channel axis
    pil = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    pil = pil.resize((_SIZE, _SIZE), Image.LANCZOS)
    arr = np.array(pil, dtype=np.float32) / 255.0          # [224,224,3] in [0,1]
    arr = (arr - _MEAN) / _STD                              # CLIP normalisation
    return arr[np.newaxis]                                  # [1,224,224,3]


def embed(img_bgr: np.ndarray, session: ort.InferenceSession) -> tuple[np.ndarray, np.ndarray]:
    """Return (content_vec[512], degrad_vec[512]) for a BGR uint8 image array."""
    inp = preprocess(img_bgr)
    input_name = session.get_inputs()[0].name
    out = session.run(None, {input_name: inp})
    # output shape [2, 512]: row 0 = content, row 1 = degradation
    vecs = out[0]  # [2, 512]
    return vecs[0], vecs[1]


_DRAFT_SIZE = 512   # preprocess() resizes to 224x224 regardless of input size, so decoding a source bigger than this wastes time for no
                    # accuracy: draft(512) measured cosine similarity >=0.9997 against a full decode on real 8256x5504 camera-preview JPEGs,
                    # far inside every similarity threshold this package uses (0.90-0.97), at ~13x the decode speed (0.64s -> 0.05s/file).


def embed_path(path: Path, session: ort.InferenceSession) -> tuple[np.ndarray, np.ndarray]:
    """Load an image from *path* (any format Pillow supports) and embed it.

    Uses Image.draft() to let a JPEG decode at a reduced DCT scale when the source is bigger than _DRAFT_SIZE (a no-op for anything already
    smaller, and for non-JPEG formats draft() does nothing so they just decode normally): camera raw-preview JPEGs run 40+ MP and this is the
    difference between embedding thousands of them in minutes versus hours.
    """
    pil = Image.open(path)
    pil.draft("RGB", (_DRAFT_SIZE, _DRAFT_SIZE))
    pil = pil.convert("RGB")
    # Convert RGB→BGR for the shared preprocess() convention
    bgr = np.array(pil, dtype=np.uint8)[:, :, ::-1]
    return embed(bgr, session)
