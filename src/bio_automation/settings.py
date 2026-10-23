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
CACHE_DIR = env_path("BIO_CACHE_DIR", r"D:\bio-cache")
# new results (renders, published Stock files, pipeline outputs) go here; what is already under EXPORT_DIR stays where it is
RESULTS_DIR = env_path("BIO_RESULTS_DIR", r"D:\bio-results")
GALLERY_DIR = env_path("BIO_GALLERY_DIR", RESULTS_DIR / "gallery")     # the PiGallery2 media folder: ONE <year>/<year-month>/ tree, on D:
PHOTOS_CACHE = env_path("BIO_PHOTOS_CACHE", CACHE_DIR / "photos")
DOWNLOAD_DIR_ENV = "BIO_DOWNLOAD_DIR"                                    # read at call time by bio export-zip
DOWNLOAD_DIR = CACHE_DIR / "downloads"
NAS = env_path("BIO_NAS", r"Y:\rjmendez-admin\DESKTOP-BVRDK4J\Data\F\Bio")   # the Bio mirror on the NAS: <year>/<day>/<file>
SYNC_DIR = env_path("BIO_SYNC_DIR", r"D:\nikon_stack_sync")             # where the camera sync lands stacks

# --- state
QUEUE_DB = env_path("BIO_QUEUE_DB", CACHE_DIR / "state" / "queue.sqlite3")
STOCK_FLOOR = env_path("BIO_STOCK_FLOOR", PHOTOS_CACHE / "stock_floor.json")
INAT_PHOTO_CACHE = env_path("BIO_INAT_PHOTO_CACHE", PHOTOS_CACHE / "inat_photos")

# --- raw working cache (raw_cache.py): a NAS raw is read over the network once, then served from a bounded copy on D: ("0"/"false"/"no" = off: NAS paths)
RAW_CACHE = env("BIO_RAW_CACHE", "1") not in ("0", "false", "no")
RAW_CACHE_DIR = env_path("BIO_RAW_CACHE_DIR", CACHE_DIR / "raw-work")
RAW_CACHE_GB = env_float("BIO_RAW_CACHE_GB", 40)                          # least recently used entries are evicted past this
RAW_CACHE_LOCK_STALE_S = env_float("BIO_RAW_CACHE_LOCK_STALE_S", 600)     # a fill lock older than this belongs to a dead process

# --- NAS read gate (nas_io.py): at most NAS_IO_SLOTS NAS reads at once across all pipeline processes, tag before lightcraft before render before match
NAS_IO_GATE = env("BIO_NAS_IO_GATE", "1")                                  # "0" = no gate: every read goes straight to the NAS
NAS_IO_DB = env_path("BIO_NAS_IO_DB", CACHE_DIR / "state" / "nas_io.sqlite3")
NAS_IO_SLOTS = int(env("BIO_NAS_IO_SLOTS", 2))
NAS_IO_HOLD_TTL_S = env_float("BIO_NAS_IO_HOLD_TTL_S", 120)               # a slot whose holder stops renewing (it died) is free after this
NAS_IO_HOLD_MAX_S = env_float("BIO_NAS_IO_HOLD_MAX_S", 1800)              # a holder renews for at most this long: a hung read gives its slot back
NAS_IMPORT_CHUNK = int(env("BIO_NAS_IMPORT_CHUNK", 8))                     # raws per LightCraft import call, each call inside one slot (~14 s at 1.7 s a new raw)

# --- external tools: <NAME>_ENV is read at call time by the module that runs the tool (then PATH, then <NAME>_DEFAULT)
PROGRAM_FILES = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
LIGHTCRAFT_CLI_ENV = "BIO_LIGHTCRAFT_CLI"
LIGHTCRAFT_CLI_DEFAULT = PROGRAM_FILES / "LightCraft" / "lightcraft-cli.exe"
LIGHTCRAFT_CLI = env_path(LIGHTCRAFT_CLI_ENV, LIGHTCRAFT_CLI_DEFAULT)
LIGHTCRAFT_LIBRARY = env_path("BIO_LC_LIBRARY", HOME / "Pictures" / "LightCraft Library")   # the persistent catalog every render uses
LIGHTCRAFT_GUI = env("BIO_LC_GUI", "127.0.0.1:7980")                    # the desktop app's control channel
LIGHTCRAFT_GPU = env("BIO_LC_GPU", "1") not in ("0", "false", "no")     # headless develop/export on the GPU (LightCraft falls back to CPU per render)
LIGHTCRAFT_RENDER_DIR = env_path("BIO_LC_RENDER_DIR", CACHE_DIR / "lightcraft-render")   # one scratch library per parallel render engine
LIGHTCRAFT_LOG_DIR = env_path("BIO_LC_LOG_DIR", CACHE_DIR / "logs" / "lightcraft")   # each engine's stderr (<library name>.stderr.log)
LIGHTCRAFT_EVENTS = env_path("BIO_LC_EVENTS", LIGHTCRAFT_LOG_DIR / "events.jsonl")   # failed resets, CPU fallbacks: one JSON line each
RENDER_ENGINES = int(env("BIO_RENDER_ENGINES", 4))       # measured 2026-10-21: 1/4/6/8 CPU engines = 12/19/22/23 renders/min; GPU engines contend
RENDER_ENGINE_GPU = env("BIO_RENDER_ENGINE_GPU", "0") not in ("0", "false", "no")   # several engines on the shared GPUs were slower than one
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

# --- iNaturalist API: a 429 (rate limit), a 5xx or a network error is retried with exponential backoff (Retry-After honoured, capped)
INAT_RETRIES = int(env("BIO_INAT_RETRIES", 5))                            # retries after the first attempt
INAT_BACKOFF_S = env_float("BIO_INAT_BACKOFF_S", 2.0)                     # first wait; doubles each retry
INAT_BACKOFF_MAX_S = env_float("BIO_INAT_BACKOFF_MAX_S", 120.0)           # no single wait is longer than this
# one pace for the whole machine: every process that calls the API reserves its slot in this file, so N workers together stay at one request
# per INAT_MIN_INTERVAL_S (a 429 pushes the next slot of every process back, not only the one that was throttled)
INAT_MIN_INTERVAL_S = env_float("BIO_INAT_MIN_INTERVAL_S", 1.1)
INAT_RATE_DB = env_path("BIO_INAT_RATE_DB", CACHE_DIR / "state" / "inat_rate.sqlite3")
INAT_TAXON_CACHE = env_path("BIO_INAT_TAXON_CACHE", CACHE_DIR / "inat" / "taxa.sqlite3")   # taxon lineages (observations are always fetched live)
INAT_TAXON_TTL_S = env_float("BIO_INAT_TAXON_TTL_S", 30 * 86400)

# --- social posts (Bluesky): drafts the owner approves, one posted a day, from the current month of every year in the gallery
SOCIAL_DIR = env_path("BIO_SOCIAL_DIR", RESULTS_DIR / "social")                     # drafts/<draft id>/: draft.json + the post's images
SOCIAL_LEDGER = env_path("BIO_SOCIAL_LEDGER", CACHE_DIR / "state" / "social_ledger.jsonl")   # append-only draft/approve/post events
GALLERY_TITLE_CACHE = env_path("BIO_GALLERY_TITLE_CACHE", CACHE_DIR / "state" / "gallery_titles_cache.json")   # social-index: embedded titles by (path, size, mtime)
SOCIAL_PREP_MIN_FREE_C_GB = float(env("BIO_SOCIAL_PREP_MIN_FREE_C_GB", 15))    # social-prep pauses below this (the NAS pipeline pauses there too)
SOCIAL_PREP_MIN_FREE_D_GB = float(env("BIO_SOCIAL_PREP_MIN_FREE_D_GB", 20))
SOCIAL_PREP_MAX_WAIT_S = env_float("BIO_SOCIAL_PREP_MAX_WAIT_S", 3600)         # social-prep pauses this long for disk space (then stops with the reason)
SOCIAL_POST_MIN_FREE_GB = env_float("BIO_SOCIAL_POST_MIN_FREE_GB", 2)          # a post run writes only the ledger, audit log and a draft's JSON: a far lower floor than a render
SOCIAL_POST_MAX_WAIT_S = env_float("BIO_SOCIAL_POST_MAX_WAIT_S", 600)         # a missed post is worse than a late one, but a full disk must not be written to forever
SOCIAL_INDEX_AUDIT = env("BIO_SOCIAL_INDEX_AUDIT", "D:/bio-cache/photos/adobe_ready_audit.csv")       # social-prep's social-index inputs, ';' separated (globs ok)
SOCIAL_INDEX_CANDIDATES = env("BIO_SOCIAL_INDEX_CANDIDATES", ";".join([
    "D:/bio-cache/photos/stock_candidates_*.csv", "F:/bio-automation-data/nas_pipeline/candidates/*.csv",
    "D:/bio-results/nas_pipeline/candidates/*.csv", "D:/bio-results/nas_pipeline/candidates-gaps/*.csv"]))
GALLERY_OBS_INDEX = env_path("BIO_GALLERY_OBS_INDEX", CACHE_DIR / "state" / "gallery_obs_index.json")   # gallery file -> iNat observation
SOCIAL_KEEP_PENDING = int(env("BIO_SOCIAL_KEEP_PENDING", 7))      # social-draft tops the current month's unposted drafts up to this many
SOCIAL_POSTS_PER_DAY = int(env("BIO_SOCIAL_POSTS_PER_DAY", 3))    # scheduled posts a day (08:00, 12:00, 20:00); manual posts are not counted
SOCIAL_AUTO_APPROVE = env("BIO_SOCIAL_AUTO_APPROVE", "1").strip().lower() not in ("", "0", "false", "no")   # owner 2026-10-22: no per-draft review
# themed posts ("Mushroom Monday": up to 4 photos of one kind of organism in one post, one theme per weekday; see themes.py)
SOCIAL_THEMES = env("BIO_SOCIAL_THEMES", "1").strip().lower() not in ("", "0", "false", "no")
SOCIAL_THEMES_FILE = env_path("BIO_SOCIAL_THEMES_FILE", SOCIAL_DIR / "themes.toml")             # the owner's calendar; built-in defaults without it
SOCIAL_THEME_DAYS_AHEAD = int(env("BIO_SOCIAL_THEME_DAYS_AHEAD", 7))                              # social-draft prepares the themed posts of the next N days
OBS_TAXONOMY_FILE = env_path("BIO_OBS_TAXONOMY_FILE", CACHE_DIR / "state" / "obs_taxonomy.json")   # observation id -> taxon + lineage (iNat, cached)
INAT_BEST_PHOTO_DIR = env_path("BIO_INAT_BEST_PHOTO_DIR", PHOTOS_CACHE / "inat_best")                 # the best size of an iNat photo, fetched when a draft needs it
INAT_SNAPSHOT = env_path("BIO_INAT_SNAPSHOT", Path(r"F:\bio-automation-data\inat_observations.json"))   # all of the owner's iNat observations (photo ids included)
# every posting rule is optional and every decision audited (social_rules.py): the owner's switches, and the append-only decision log
SOCIAL_RULES_FILE = env_path("BIO_SOCIAL_RULES_FILE", SOCIAL_DIR / "rules.toml")
SOCIAL_AUDIT_LOG = env_path("BIO_SOCIAL_AUDIT_LOG", CACHE_DIR / "state" / "social_audit.jsonl")
SOCIAL_BREAKER_COOLDOWN_H = float(env("BIO_SOCIAL_BREAKER_COOLDOWN_H", 6))                  # an open circuit breaker allows one probe post after this many hours
SOCIAL_PAUSE_FILE = env_path("BIO_SOCIAL_PAUSE_FILE", SOCIAL_DIR / "PAUSED")                    # while it exists no scheduled post goes out (pause_switch rule)
SOCIAL_ALERTS_LOG = env_path("BIO_SOCIAL_ALERTS_LOG", CACHE_DIR / "state" / "social_alerts.jsonl")
SOCIAL_ALERT_COMMAND = env("BIO_SOCIAL_ALERT_COMMAND", "")                                         # optional: run with each alert's text; empty = log only
SOCIAL_DIGEST_DIR = env_path("BIO_SOCIAL_DIGEST_DIR", SOCIAL_DIR / "digest")
SOCIAL_INTEGRITY_ACK_FILE = env_path("BIO_SOCIAL_INTEGRITY_ACK_FILE", SOCIAL_DIR / "integrity_ack.json")   # duplicate posts the owner accepted (social-integrity --accept-duplicate)
SOCIAL_BACKUP_DIR = env_path("BIO_SOCIAL_BACKUP_DIR", SOCIAL_DIR / "backups")
SOCIAL_ENGAGEMENT_LOG = env_path("BIO_SOCIAL_ENGAGEMENT_LOG", CACHE_DIR / "state" / "social_engagement.jsonl")
SOCIAL_SLOTS = env("BIO_SOCIAL_SLOTS", "08:00,12:00,20:00")                                        # the scheduled run times (local), for slot-aware rules
# app.bsky.embed.images allows 2,000,000 B per image (formerly 1 MB); the official app compresses post images to the same pair
# (social-app src/lib/constants.ts IMAGE_SIZE_CONFIG_POSTS: maxDimension 4000, maxSize 2000000)
BSKY_IMAGE_MAX_BYTES = int(env("BIO_BSKY_IMAGE_MAX_BYTES", 2_000_000))
BSKY_IMAGE_MAX_EDGE = int(env("BIO_BSKY_IMAGE_MAX_EDGE", 4000))
# observation id -> where to buy a print (e.g. its Darkroom product page; Darkroom has no API, so the owner keeps this csv: obs_id,url)
SOCIAL_PRINTS_FILE = env_path("BIO_SOCIAL_PRINTS_FILE", RESULTS_DIR / "social" / "prints.csv")
SOCIAL_PRINTS_MANUAL_FILE = env_path("BIO_SOCIAL_PRINTS_MANUAL_FILE", SOCIAL_DIR / "prints_manual.csv")   # owner's obs_id,url rows: they win over darkroom-sync's
DARKROOM_URL = env("BIO_DARKROOM_URL", "https://rjmendez.darkroom.com")             # the owner's storefront
DARKROOM_DIR = env_path("BIO_DARKROOM_DIR", CACHE_DIR / "darkroom")                 # product images, vectors, the sync state, the review csv
DARKROOM_MARGIN = env_float("BIO_DARKROOM_MARGIN", 0.005)                           # a print matches only if no other observation is this close
DARKROOM_PAUSE_S = env_float("BIO_DARKROOM_PAUSE_S", 1.0)                           # between storefront page requests
QDRANT_COLLECTION = env("BIO_QDRANT_COLLECTION", "bio_observations")                # gallery images with iNat + Stock data (url/key: BIO_QDRANT_URL/_API_KEY or Loci's backends.toml)
QDRANT_INAT_CACHE = env_path("BIO_QDRANT_INAT_CACHE", CACHE_DIR / "state" / "obs_inat.json")   # observation id -> taxon, lineage, grade, place, license, tags
# every iNat observation of the owner (no photos) in Qdrant: `<prefix>observations` and `<prefix>taxa` (qdrant_inat.py)
INAT_LOGIN = env("BIO_INAT_LOGIN", "rjmendez")                                       # whose public observations are read (unauthenticated API)
QDRANT_INAT_PREFIX = env("BIO_QDRANT_INAT_PREFIX", "bio_inat_")
QDRANT_INAT_FULL = env_path("BIO_QDRANT_INAT_FULL", CACHE_DIR / "state" / "inat_full.json")          # the shaped observations + places, topped up incrementally
QDRANT_INAT_STATE = env_path("BIO_QDRANT_INAT_STATE", CACHE_DIR / "state" / "qdrant_inat_state.json")   # point id -> content hash: only changed points are written
INAT_COVERAGE_CSV = env_path("BIO_INAT_COVERAGE_CSV", "F:/bio-automation-data/inat_coverage.csv")     # which iNat photos have a local source file
# Bluesky sign-in: atproto OAuth through the bsky-oauth library (github.com/rjmendez/bsky-oauth, glue around Bluesky's official
# @atproto/oauth-client-node). This is the "poster" profile: a confidential client whose metadata + PUBLIC key are published by repo
# rjmendez/bio-social-oauth (GitHub Pages); its private key and session stay in BSKY_OAUTH_DIR (owner + SYSTEM only). No password.
BSKY_HANDLE = env("BIO_BSKY_HANDLE", "trunkfullofdeadhookers.com")
BSKY_OAUTH_DIR = env_path("BIO_BSKY_OAUTH_DIR", r"F:\services\bio-social\oauth")
SOCIAL_SCRIPTS_DIR = env_path("BIO_SOCIAL_SCRIPTS_DIR", r"F:\services\bio-social")      # where scripts\deploy_prod.ps1 copies scripts\windows\*.ps1; social-selfcheck compares them to the repo copies
BSKY_CLIENT_ID = env("BIO_BSKY_CLIENT_ID", "https://rjmendez.github.io/bio-social-oauth/poster/client-metadata.json")
BSKY_REDIRECT_URI = env("BIO_BSKY_REDIRECT_URI", "https://rjmendez.github.io/bio-social-oauth/callback.html")
BSKY_OAUTH_SCOPE = env("BIO_BSKY_OAUTH_SCOPE", "atproto repo:app.bsky.feed.post?action=create blob:image/*")   # create posts + upload images only
BSKY_CLIENT_NAME = env("BIO_BSKY_CLIENT_NAME", "bio-automation photo posts")
# "cleanup" profile: a second confidential client that can ONLY delete posts (removing the old inat_skeet bot posts); its own key,
# session and sign-in, so the poster can never delete and the cleanup client can never post
BSKY_CLEANUP_OAUTH_DIR = env_path("BIO_BSKY_CLEANUP_OAUTH_DIR", r"F:\services\bio-social\oauth-cleanup")
BSKY_CLEANUP_CLIENT_ID = env("BIO_BSKY_CLEANUP_CLIENT_ID", "https://rjmendez.github.io/bio-social-oauth/cleanup/client-metadata.json")
BSKY_CLEANUP_SCOPE = env("BIO_BSKY_CLEANUP_SCOPE", "atproto repo:app.bsky.feed.post?action=delete")
BSKY_ARCHIVE_DIR = env_path("BIO_BSKY_ARCHIVE_DIR", CACHE_DIR / "bsky-archive")   # repo exports, cleanup plans and their ledgers

# --- local vision model (Ollama, GPU only). No model is named here: env BIO_VISION_MODEL, else the best-ranked installed "vision" model of
# Loci's pool file
OLLAMA_URL = env("BIO_OLLAMA", "http://127.0.0.1:11440")         # the server itself: the resident proxy on :11435 rewrites model names
VISION_MODEL_ENV = "BIO_VISION_MODEL"
EMBED_MODEL_ENV = "BIO_EMBED_MODEL"                                 # else the pool's "embed" role (the owner's alt-text style examples)
SOCIAL_STYLE_PROFILE = env_path("BIO_SOCIAL_STYLE_PROFILE", CACHE_DIR / "state" / "bsky_style_profile.json")
BSKY_APPVIEW = env("BIO_BSKY_APPVIEW", "https://public.api.bsky.app")   # public, unauthenticated reads
PLC_URL = env("BIO_PLC_URL", "https://plc.directory")
MODEL_POOL_FILE = env_path("BIO_MODEL_POOL", HOME / ".loci" / "backends.toml")
VISION_IMAGE_EDGE = int(env("BIO_VISION_IMAGE_EDGE", 1024))       # long edge of the copy the vision model sees

# --- disk guard (disk_guard.py): workers PAUSE (poll, audit, resume) instead of failing mid-write when a drive drops under its floor
DISK_GUARD_SYSTEM_MIN_GB = env_float("BIO_DISK_GUARD_SYSTEM_MIN_GB", 15)    # the system drive (C:)
DISK_GUARD_DATA_MIN_GB = env_float("BIO_DISK_GUARD_DATA_MIN_GB", 20)        # the data drive: the one holding CACHE_DIR
DISK_GUARD_POLL_S = env_float("BIO_DISK_GUARD_POLL_S", 60)
DISK_GUARD_RESUME_MARGIN_GB = env_float("BIO_DISK_GUARD_RESUME_MARGIN_GB", 1)   # resume only at floor + margin, so a drive at the edge does not flap
DISK_GUARD_MAX_WAIT_S = env_float("BIO_DISK_GUARD_MAX_WAIT_S", 0)           # 0 = a daemon worker waits as long as it takes; a scheduled run sets its own limit
DISK_GUARD_LOG = env_path("BIO_DISK_GUARD_LOG", CACHE_DIR / "state" / "disk_guard.jsonl")   # pause/resume events of workers that have no audit log of their own
SCRATCH_DIR = env_path("BIO_SCRATCH_DIR", CACHE_DIR / "tmp")               # every temp file the package makes goes here (D:), never %TEMP% on C:


def scratch_dir() -> Path:
    """SCRATCH_DIR, created. Pass it as dir= to tempfile calls."""
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    return SCRATCH_DIR


RAW_EXTENSIONS = (".nef", ".dng", ".cr2", ".cr3", ".arw", ".raf", ".orf", ".rw2")
MIN_FREE_GB = env_float("BIO_MIN_FREE_GB", 15)                            # stop writing renders below this much free space
RAW_CACHE_MIN_FREE_GB = env_float("BIO_RAW_CACHE_MIN_FREE_GB", MIN_FREE_GB + 10)   # a fill first evicts old entries to keep this free, else reads the NAS:
                                                                                   # above MIN_FREE_GB, so the cache gives its space back before a stage pauses

