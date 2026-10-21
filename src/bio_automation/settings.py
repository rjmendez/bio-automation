"""Every machine-specific location, tool and tuning value the package uses: one definition, each overridable by the environment variable
named in its env() call. Modules import these instead of carrying their own literals, so a path or threshold cannot drift between the
producer and the consumer of a handoff. The ops scripts' bio_paths.py re-exports this module.
"""
from __future__ import annotations

import os
from pathlib import Path
from zoneinfo import ZoneInfo

HOME = Path.home()


def env(name: str, default) -> str:
    v = os.environ.get(name)
    return v if v not in (None, "") else str(default)


def env_path(name: str, default) -> Path:
    return Path(env(name, default))


def env_float(name: str, default: float) -> float:
    return float(env(name, default))


# --- photo storage
LIBRARY = env_path("BIO_LIBRARY", r"F:\Bio")                            # the raw library: <year>/<day>/<file>
EXPORT_DIR = env_path("BIO_EXPORT_DIR", r"F:\Bio-export")
STOCK_DIR = env_path("BIO_STOCK_DIR", EXPORT_DIR / "adobe-stock")
GALLERY_DIR = env_path("BIO_GALLERY_DIR", STOCK_DIR / "gallery")       # the PiGallery2 media folder (<year>/<year-month>/)
CACHE_DIR = env_path("BIO_CACHE_DIR", r"D:\bio-cache")
PHOTOS_CACHE = env_path("BIO_PHOTOS_CACHE", CACHE_DIR / "photos")
DOWNLOAD_DIR_ENV = "BIO_DOWNLOAD_DIR"                                    # read at call time by bio export-zip
DOWNLOAD_DIR = CACHE_DIR / "downloads"
NAS = env_path("BIO_NAS", r"Y:\rjmendez-admin\DESKTOP-BVRDK4J\Data\F\Bio")   # the Bio mirror on the NAS: <year>/<day>/<file>
SYNC_DIR = env_path("BIO_SYNC_DIR", r"D:\nikon_stack_sync")             # where the camera sync lands stacks

# --- state
QUEUE_DB = env_path("BIO_QUEUE_DB", CACHE_DIR / "state" / "queue.sqlite3")
STOCK_FLOOR = env_path("BIO_STOCK_FLOOR", PHOTOS_CACHE / "stock_floor.json")
INAT_PHOTO_CACHE = env_path("BIO_INAT_PHOTO_CACHE", PHOTOS_CACHE / "inat_photos")

# --- external tools: <NAME>_ENV is read at call time by the module that runs the tool (then PATH, then <NAME>_DEFAULT)
PROGRAM_FILES = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
LIGHTCRAFT_CLI_ENV = "BIO_LIGHTCRAFT_CLI"
LIGHTCRAFT_CLI_DEFAULT = PROGRAM_FILES / "LightCraft" / "lightcraft-cli.exe"
LIGHTCRAFT_CLI = env_path(LIGHTCRAFT_CLI_ENV, LIGHTCRAFT_CLI_DEFAULT)
LIGHTCRAFT_LIBRARY = env_path("BIO_LC_LIBRARY", HOME / "Pictures" / "LightCraft Library")   # the persistent catalog every render uses
LIGHTCRAFT_GUI = env("BIO_LC_GUI", "127.0.0.1:7980")                    # the desktop app's control channel
LIGHTCRAFT_GPU = env("BIO_LC_GPU", "1") not in ("0", "false", "no")     # headless develop/export on the GPU (LightCraft falls back to CPU per render)
EXIFTOOL_ENV = "BIO_EXIFTOOL"
EXIFTOOL_DEFAULT = PROGRAM_FILES / "ZereneStacker" / "exiftool" / "exiftool.exe"
FOCUS_STACK_ENV = "BIO_FOCUS_STACK"
FOCUS_STACK_DEFAULT = HOME / "AppData" / "Local" / "Programs" / "focus-stack" / "focus-stack" / "focus-stack.exe"
TOPAZ_MODELS_DIR = env_path("BIO_TOPAZ_MODELS_DIR", r"C:\ProgramData\Topaz Labs LLC\Topaz Photo AI\models")
MODEL_CACHE_DIR = env_path("BIO_MODEL_CACHE_DIR", r"C:\TopazModels")   # extracted ONNX models (DA-CLIP, Topaz)
# onnxruntime's CUDA provider here is a CUDA 13 build: these DLL folders must be on the DLL path before onnxruntime is imported
# (os.pathsep-separated list; the CUDA 12 pip `nvidia` packages do not satisfy it)
CUDA_DLL_DIRS = tuple(Path(p) for p in env("BIO_CUDA_DLL_DIRS", os.pathsep.join([
    str(PROGRAM_FILES / "NVIDIA GPU Computing Toolkit" / "CUDA" / "v13.0" / "bin" / "x64"),
    str(PROGRAM_FILES / "NVIDIA" / "CUDNN" / "v9.13" / "bin" / "13.0")])).split(os.pathsep) if p)

# --- local time: iNat observed_on days and camera clocks are local
LOCAL_TZ = ZoneInfo(env("BIO_LOCAL_TZ", "America/New_York"))

# --- DA-CLIP photo matching (cosine similarity, measured by eye on real pairs)
EXACT_FRAME_SIMILARITY = env_float("BIO_EXACT_FRAME_SIMILARITY", 0.97)   # the very frame that was uploaded
SOURCE_OR_CROP_SIMILARITY = env_float("BIO_SOURCE_SIMILARITY", 0.93)     # the source raw of an uploaded crop / edit
PHOTO_MATCH_WINDOW_S = env_float("BIO_PHOTO_MATCH_WINDOW_S", 15 * 60)    # time bracket when the frame's capture time is known
DAY_BRACKET_S = env_float("BIO_DAY_BRACKET_S", 12 * 3600)                # bracket when only the day is known (frame stamped at local noon)

RAW_EXTENSIONS = (".nef", ".dng", ".cr2", ".cr3", ".arw", ".raf", ".orf", ".rw2")
MIN_FREE_GB = env_float("BIO_MIN_FREE_GB", 15)                            # stop writing renders below this much free space
