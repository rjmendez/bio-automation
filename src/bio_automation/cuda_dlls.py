"""Put the CUDA/cuDNN DLL folders (settings.CUDA_DLL_DIRS) on the DLL search path. Call before `import onnxruntime`.

Without them onnxruntime's CUDA provider cannot load (cublas64_13 / cudart64_13) and a session silently runs on the CPU.
"""
from __future__ import annotations

import os
from pathlib import Path

from . import settings

_added: set[str] = set()


def add_cuda_dll_dirs(dirs=settings.CUDA_DLL_DIRS, *, add_dll_directory=getattr(os, "add_dll_directory", None), environ=os.environ) -> list[Path]:
    """Add each existing folder to os.add_dll_directory and to PATH (cuDNN loads its sub-libraries through PATH). Idempotent.
    Returns the folders that were missing, so a caller can report them; requiring CUDA is left to ort_providers.open_session."""
    missing = []
    for d in map(Path, dirs):
        if not d.is_dir():
            missing.append(d)
            continue
        key = os.path.normcase(str(d))
        if key in _added:
            continue
        if add_dll_directory is not None:
            add_dll_directory(str(d))
        environ["PATH"] = str(d) + os.pathsep + environ.get("PATH", "")
        _added.add(key)
    return missing
