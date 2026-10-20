"""Open an ONNX Runtime session on the GPU, and say so loudly when that does not happen.

Why this exists: ``ort.InferenceSession(path)`` with no providers runs on CPU and does not raise;
``providers=['CUDAExecutionProvider', ...]`` falls back to CPU just as quietly when the CUDA/cuDNN DLLs cannot be
loaded.  Both cost ~10-50x latency.  ``open_session`` therefore preloads the CUDA DLLs, requests the provider list,
then CHECKS ``session.get_providers()`` and reports every degraded outcome: a WARNING (rate limited), a counter
(``degraded_counts()``) and, with ``require_cuda=True``, a ``RuntimeError``.

The CUDA device index is configuration (``BIO_ORT_CUDA_DEVICE`` or the caller's config value), never a literal.
Note CUDA device order is not nvidia-smi PCI order on this workstation: check with ``nvidia-smi`` /
``torch.cuda.get_device_name`` before choosing an index.

``pipeline/stages/ort_providers.py`` is a byte-identical copy for the flat stage scripts that run from
``F:\\bio-automation-data`` without this package on ``sys.path``; ``tests/test_ort_providers.py`` keeps them in sync.
"""
from __future__ import annotations

import logging
import os
import time
from collections import Counter

log = logging.getLogger("bio_automation.ort_providers")

CUDA = "CUDAExecutionProvider"
CPU = "CPUExecutionProvider"
DEVICE_ENV = "BIO_ORT_CUDA_DEVICE"
DEFAULT_REQUEST = (CUDA, CPU)
WARN_INTERVAL_S = 60.0

_COUNTS: Counter = Counter()
_LAST_WARN: dict = {}


def degraded_counts() -> dict:
    """How many times each degraded outcome happened in this process, keyed ``"<label>:<reason>"``."""
    return dict(_COUNTS)


def _report(label: str, reason: str, message: str, clock=time.monotonic) -> None:
    """Count every degraded outcome; log at WARNING at most once per WARN_INTERVAL_S per label+reason."""
    key = f"{label}:{reason}"
    _COUNTS[key] += 1
    now = clock()
    last = _LAST_WARN.get(key)
    if last is None or now - last >= WARN_INTERVAL_S:
        _LAST_WARN[key] = now
        log.warning("%s (occurrence %d of this kind in this process)", message, _COUNTS[key])


def configured_device_id(config_value=None, environ=None) -> int:
    """CUDA device index: ``BIO_ORT_CUDA_DEVICE`` wins, then *config_value*, then 0 (ORT's own default)."""
    environ = os.environ if environ is None else environ
    raw = environ.get(DEVICE_ENV)
    source = DEVICE_ENV
    if raw in (None, ""):
        raw, source = config_value, "config cuda_device_id"
    if raw in (None, ""):
        return 0
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{source} must be an integer CUDA device index, got {raw!r}") from None
    if value < 0:
        raise ValueError(f"{source} must be >= 0, got {value}")
    return value


def open_session(ort, model_path, *, sess_options=None, requested=None, device_id=None, require_cuda=False,
                 label="onnx", retries=2, backoff_s=0.5, sleep=time.sleep, clock=time.monotonic):
    """Create an ``ort.InferenceSession`` for *model_path* and verify which provider it really got.

    requested     provider names in priority order (default CUDA then CPU).  Names ORT does not offer are dropped
                  with a WARNING; CPU is always kept last so a session can still be created.
    device_id     CUDA device index (default :func:`configured_device_id`).
    require_cuda  raise RuntimeError instead of running on CPU when CUDA was requested but not obtained.
    retries       extra attempts (exponential backoff from *backoff_s*) when session creation raises.  A raised error
                  is treated as possibly transient; only after the retries are spent does it propagate.
    """
    wanted = list(requested) if requested else list(DEFAULT_REQUEST)
    if device_id is None:
        device_id = configured_device_id()
    available = list(ort.get_available_providers())

    usable = [p for p in wanted if p in available]
    for missing in (p for p in wanted if p not in available):
        _report(label, f"{missing}-unavailable",
                f"{label}: requested {missing} but this onnxruntime offers only {available}; "
                f"install onnxruntime-gpu to use the GPU", clock)
    if CPU not in usable:
        usable.append(CPU)
    wants_cuda = CUDA in wanted

    if CUDA in usable and hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()
        except Exception as e:  # the provider check below decides whether this mattered
            _report(label, "preload-dlls-failed",
                    f"{label}: ort.preload_dlls() raised {type(e).__name__}: {str(e)[:200]}", clock)

    providers = [(p, {"device_id": device_id}) if p == CUDA else p for p in usable]
    session = None
    for attempt in range(retries + 1):
        try:
            session = ort.InferenceSession(str(model_path), sess_options=sess_options, providers=providers)
            break
        except Exception as e:
            if attempt >= retries:
                raise
            _report(label, "session-create-retry",
                    f"{label}: InferenceSession creation failed ({type(e).__name__}: {str(e)[:200]}); "
                    f"retry {attempt + 1} of {retries}", clock)
            sleep(backoff_s * (2 ** attempt))

    got = list(session.get_providers())
    if wants_cuda and (not got or got[0] != CUDA):
        msg = (f"{label}: CUDA was requested (device_id={device_id}) but the session runs on {got}; "
               f"expect ~10-50x slower inference (cuDNN/CUDA DLLs missing or CPU-only onnxruntime)")
        _report(label, "cuda-not-obtained", msg, clock)
        if require_cuda:
            raise RuntimeError(msg)
    elif wants_cuda:
        log.info("%s: session runs on %s (device_id=%d)", label, got, device_id)
    return session
