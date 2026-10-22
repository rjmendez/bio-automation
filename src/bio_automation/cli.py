from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import settings
from .auto_ingest import NotACameraVolumeError, auto_ingest_volume
from .bundle import DEFAULT_DOWNLOAD_DIR, BundleError, build_bundle
from .inat import fetch_inat_observation
from .models import Asset, Observation
from .paths import make_observation_path
from .pipeline import (
    build_and_save_keywords,
    copy_into_raw,
    delete_rejected_raws,
    ensure_layout,
    export_inat,
    export_stock,
    load_observation,
    merge_inat_fields,
    process_all,
    process_observation,
    render_review,
    save_assets,
    save_observation,
    set_asset_status,
    clean_and_stack,
    stack_all,
    stack_focus,
    write_sidecars,
)
from .quality import PRESORT_FILENAME, suggest_culling
from .stack_sync import process_synced_stacks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bio")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest")
    p_ingest.add_argument("source")
    p_ingest.add_argument("--base-dir", default=str(settings.LIBRARY))
    p_ingest.add_argument("--obsdate", required=True)
    p_ingest.add_argument("--slug", required=True)

    p_auto_ingest = sub.add_parser(
        "auto-ingest",
        help="Detect whether volume_path looks like a camera/card source, group its raw "
             "files by real EXIF (capture date, camera model), and ingest each group into "
             "its own observation folder under --base-dir -- idempotently: re-running "
             "against the same card never resets an existing asset's kept/rejected status "
             "or an already-linked iNat observation. See bio_automation.auto_ingest.",
    )
    p_auto_ingest.add_argument("volume_path")
    p_auto_ingest.add_argument("--base-dir", default=str(settings.LIBRARY))

    p_sync_stacks = sub.add_parser(
        "sync-stacks",
        help="Ingest focus-stacks synced in from the NikonStacker Pi (nikon-z7ii-tools/"
             "stack_buffer.py) and auto-process each one (stack/sidecars/review/export) "
             "on arrival. Idempotent: a stack already ingested is skipped on rerun. "
             "See bio_automation.stack_sync.",
    )
    p_sync_stacks.add_argument("--sync-dir", default=str(settings.SYNC_DIR))
    p_sync_stacks.add_argument("--base-dir", default=str(settings.LIBRARY))
    p_sync_stacks.add_argument("--no-cleanup", action="store_true",
                                help="Keep the D:\\ landing copy after a successful "
                                     "ingest+process instead of deleting it.")

    p_cull = sub.add_parser("cull")
    p_cull.add_argument("folder")

    p_presort = sub.add_parser(
        "presort",
        help="Score ingested raws for blur/exposure/near-duplicates and write a "
             "keep/cull proposal -- never changes asset status itself; review the "
             "written file, then apply with `bio reject`/`bio keep`.",
    )
    p_presort.add_argument("folder")

    p_reject = sub.add_parser("reject")
    p_reject.add_argument("folder")
    p_reject.add_argument("files", nargs="+")

    p_keep = sub.add_parser("keep")
    p_keep.add_argument("folder")
    p_keep.add_argument("files", nargs="+")

    p_fetch_inat = sub.add_parser("fetch-inat")
    p_fetch_inat.add_argument("folder")
    p_fetch_inat.add_argument("--observation-id", required=True, type=int)
    p_fetch_inat.add_argument("--force", action="store_true")

    p_stack = sub.add_parser("stack")
    p_stack.add_argument("folder")
    p_stack.add_argument("--method", default="focus-stack", choices=["focus-stack", "zerene"])
    p_stack.add_argument("--timeout", type=float, default=1800.0)
    p_stack.add_argument("--denoise", action="store_true",
                          help="Denoise+levels every raw (fixed params shared across "
                               "the stack) before handing off to the stacking engine")
    p_stack.add_argument("--noise", type=float, default=0.0,
                          help="Fixed slb t_param1 (a SHARPENING amplitude, not a denoiser; measured 2026-10-20) 0..1, default 0, only with --denoise")
    p_stack.add_argument("--exposure-ev", type=float, default=0.0,
                          help="Fixed EV pull-down applied after denoise, only with --denoise "
                               "(negative leaves headroom under clipping for specular highlights)")
    p_stack.add_argument("--subject-boost-ev", type=float, default=0.0,
                          help="Extra EV lift inside the subject only (on top of --exposure-ev), "
                               "only with --denoise. Uses Topaz Photo AI's s_mask_l_flexible "
                               "subject-mask model (not Meta's SAM); 0.0 (default) skips the mask "
                               "model entirely.")
    p_stack.add_argument("--min-free-gb", type=float, default=5.0,
                          help="Abort before any work if free disk space at the "
                               "destination/source folder is below this many GB")
    p_stack.add_argument("--skip-subject-check", action="store_true",
                          help="Skip the DA-CLIP subject-consistency safety gate before "
                               "handing off to the stacking engine. Runs by default and "
                               "raises SubjectMismatchError if any frame looks like a "
                               "different subject (see bio_automation.grouping); pass this "
                               "flag only for a deliberate override.")

    p_stack_all = sub.add_parser("stack-all")
    p_stack_all.add_argument("base_dir")
    p_stack_all.add_argument("--method", default="focus-stack", choices=["focus-stack", "zerene"])
    p_stack_all.add_argument("--timeout", type=float, default=1800.0)
    p_stack_all.add_argument("--noise", type=float, default=0.0,
                              help="Fixed slb t_param1 (a SHARPENING amplitude, not a denoiser; measured 2026-10-20) 0..1, default 0, applied to every folder's stack")
    p_stack_all.add_argument("--exposure-ev", type=float, default=0.0,
                              help="Fixed EV pull-down applied after denoise to every folder's "
                                   "stack (negative leaves headroom under clipping for specular "
                                   "highlights)")
    p_stack_all.add_argument("--subject-boost-ev", type=float, default=0.0,
                              help="Extra EV lift inside the subject only (on top of "
                                   "--exposure-ev), for every folder's stack. Uses Topaz Photo "
                                   "AI's s_mask_l_flexible subject-mask model (not Meta's SAM); "
                                   "0.0 (default) skips the mask model entirely.")
    p_stack_all.add_argument("--min-free-gb", type=float, default=5.0,
                              help="Abort before any work if free disk space at the "
                                   "destination/source folder is below this many GB")
    p_stack_all.add_argument("--skip-subject-check", action="store_true",
                              help="Skip the DA-CLIP subject-consistency safety gate before "
                                   "handing off to the stacking engine, for every folder. Runs "
                                   "by default and raises SubjectMismatchError if any frame "
                                   "looks like a different subject (see "
                                   "bio_automation.grouping); pass this flag only for a "
                                   "deliberate override.")

    p_write_sidecars = sub.add_parser("write-sidecars")
    p_write_sidecars.add_argument("folder")

    p_render_review = sub.add_parser("render-review")
    p_render_review.add_argument("folder")

    p_export_inat = sub.add_parser("export-inat")
    p_export_inat.add_argument("folder")

    p_export_stock = sub.add_parser("export-stock")
    p_export_stock.add_argument("folder")

    p_export_zip = sub.add_parser(
        "export-zip",
        help="Validate export-ready JPEGs against the iNat or Stock standard and write ONE flat zip (+ .sha256) "
             "into the download folder, to fetch remotely with scp/sftp.")
    p_export_zip.add_argument("--kind", required=True, choices=["inat", "stock"])
    p_export_zip.add_argument("sources", nargs="+",
                              help="folders (JPEGs + any .csv, not recursive) or files; names must be unique")
    p_export_zip.add_argument("--out-dir", default=None,
                              help="download folder (default: env BIO_DOWNLOAD_DIR, else D:/bio-cache/downloads)")
    p_export_zip.add_argument("--name", default=None, help="zip name (default: <kind>-<n>photos-<timestamp>.zip)")

    p_flag = sub.add_parser(
        "flag-upload-set",
        help="Append `flags`/`flag_notes` columns to an iNat upload set's observations.csv: black frames (measured) plus manual per-group notes. "
             "Nothing is removed.")
    p_flag.add_argument("folder", help="upload set folder holding the JPEGs and observations.csv")
    p_flag.add_argument("--notes", default=None, help='json {"<group>": {"flags": [...], "note": "..."}}')

    p_stock_cal = sub.add_parser(
        "stock-calibrate",
        help="Measure a folder of already-approved Stock exports and save the sharpness floor candidates are judged against.")
    p_stock_cal.add_argument("folder")
    p_stock_cal.add_argument("--out", default=None, help="floor file (default: env BIO_STOCK_FLOOR)")
    p_stock_cal.add_argument("--sample", type=int, default=300)
    p_stock_cal.add_argument("--exclude-csv", default=None,
                             help="an Adobe Stock csv (Filename column) whose files are to be left out, e.g. the ones stock-publish already linked into the folder")

    p_stock_queue = sub.add_parser(
        "stock-tag-queue",
        help="Score every frame of a LightCraft iNat queue json from its raw, tag the Stock candidates, write a csv (resumable).")
    p_stock_queue.add_argument("queue_json")
    p_stock_queue.add_argument("--raw-root", required=True)
    p_stock_queue.add_argument("--out", required=True, help="csv with one row per frame, reasons included")
    p_stock_queue.add_argument("--floor", default=None)
    p_stock_queue.add_argument("--workers", type=int, default=3)
    p_stock_queue.add_argument("--created-since", default=None, metavar="YYYY-MM-DD",
                               help="only consider observations created on or after this date (the newly identified ones)")
    p_stock_queue.add_argument("--min-similarity", type=float, default=None,
                               help="photo-to-frame similarity needed (default 0.97, for frames that are the uploaded picture; ~0.93 matches "
                                    "Lightroom-edited or cropped versions of a frame)")
    p_stock_queue.add_argument("--top-per-group", type=int, default=None,
                               help="keep at most this many sharpest frames per observation (default: every uploaded photo that passes)")
    p_stock_queue.add_argument("--observations", required=True,
                               help="the owner's live iNat observations json (must be at most 3 days old); each iNat photo picks the frame that looks like it, "
                                    "each observation is fetched live and labelled -- a frame that is not an uploaded photo is never a candidate")
    p_stock_queue.add_argument("--notes", default=None,
                               help="the upload set's manual flag notes json (see flag-upload-set): every group named in it is excluded from the candidates")

    p_stock_pub = sub.add_parser(
        "stock-publish",
        help="Hardlink the csv's Stock candidates' RETOUCHED full-size renders (render-dir/<year>/<day>/<day>_<stem>.jpg) into the gallery "
             "as <year>/<year-month>/, after validating each against the Stock standard. Candidates with no render are reported, not shipped.")
    p_stock_pub.add_argument("csv")
    p_stock_pub.add_argument("--raw-root", required=True)
    p_stock_pub.add_argument("--render-dir", required=True)
    p_stock_pub.add_argument("--gallery-dir", required=True)
    p_stock_pub.add_argument("--floor", default=None, help="floor file (default: env BIO_STOCK_FLOOR)")
    p_stock_pub.add_argument("--adobe-csv", default=None, help="where to write the Adobe Stock csv (default: <render-dir>/adobe_stock_metadata.csv)")
    p_stock_pub.add_argument("--plain-decode", action="store_true",
                             help="Also produce a PLAIN raw decode (flat, no retouching) for candidates with no render. Off by default: not Stock-grade.")
    p_stock_pub.add_argument("--no-queue", action="store_true",
                             help="Don't enqueue a gallery.reindex message after linking (default: on, so the gallery worker picks up the new "
                                  "files on its own -- no manual folder-browse or scheduled rescan needed).")
    p_stock_pub.add_argument("--queue-db", default=None, help="queue database file (default: env BIO_QUEUE_DB)")

    p_social_index = sub.add_parser(
        "social-index",
        help="Map every gallery file to its iNat observation (from the audit and Stock-candidate csvs, each row checked against the title "
             "embedded in the file) and write the index social-draft reads. Unresolved files are listed with the reason.")
    p_social_index.add_argument("--gallery-dir", default=str(settings.GALLERY_DIR))
    p_social_index.add_argument("--audit", nargs="*", default=[], help="audit csv(s) with file,obs_id columns (glob patterns allowed)")
    p_social_index.add_argument("--candidates", nargs="*", default=[], help="Stock candidate csv(s) (glob patterns allowed)")
    p_social_index.add_argument("--out", default=str(settings.GALLERY_OBS_INDEX))

    p_social_draft = sub.add_parser(
        "social-draft",
        help="Top up this month's unposted Bluesky drafts: observations with gallery photos in this month of any year, fetched live, "
             "alt text from the local vision model. Writes drafts for the owner to approve; posts nothing.")
    p_social_draft.add_argument("--keep-pending", type=int, default=None, help="default: env BIO_SOCIAL_KEEP_PENDING")
    p_social_draft.add_argument("--date", default=None, help="draft for this day's month (YYYY-MM-DD; default today)")
    p_social_draft.add_argument("--index", default=str(settings.GALLERY_OBS_INDEX))
    p_social_draft.add_argument("--gallery-dir", default=str(settings.GALLERY_DIR))

    p_social_list = sub.add_parser("social-list", help="List drafts and their state (drafted/approved/rejected/posting/posted/failed).")
    p_social_list.add_argument("--state", default=None)
    p_social_list.add_argument("--full", action="store_true", help="also print each draft's whole post text, images and alt text")

    p_social_approve = sub.add_parser("social-approve", help="Approve drafts for posting (bound to their current files).")
    p_social_approve.add_argument("draft_ids", nargs="+")

    p_social_reject = sub.add_parser("social-reject", help="Reject drafts; their observations are never drafted again.")
    p_social_reject.add_argument("draft_ids", nargs="+")
    p_social_reject.add_argument("--reason", default="")

    p_social_resolve = sub.add_parser("social-resolve", help="Settle a post whose outcome is unknown, after checking the profile.")
    p_social_resolve.add_argument("draft_id")
    g = p_social_resolve.add_mutually_exclusive_group(required=True)
    g.add_argument("--posted-uri", default=None)
    g.add_argument("--not-posted", action="store_true")

    p_social_post = sub.add_parser(
        "social-post",
        help="Post the oldest approved draft of this month (at most BIO_SOCIAL_POSTS_PER_DAY a day). Dry-run unless --live; --live needs "
             "the OAuth session from `bio social-login`.")
    p_social_post.add_argument("--live", action="store_true")

    p_social_login = sub.add_parser(
        "social-login",
        help="One-time Bluesky sign-in (atproto OAuth, Bluesky's official client): open the printed address, approve, paste the address "
             "of the page you land on. Stores the session in BIO_BSKY_OAUTH_DIR; no password is stored.")
    p_social_login.add_argument("--handle", default=settings.BSKY_HANDLE)

    sub.add_parser("social-whoami", help="Refresh the stored Bluesky session and show its account, scope and expiry.")

    p_cl_plan = sub.add_parser(
        "social-cleanup-plan",
        help="Plan the removal of the old inat_skeet bot posts from live public data (those with replies from people are kept); "
             "writes the plan and prints its id. Deletes nothing.")
    p_cl_plan.add_argument("--handle", default=settings.BSKY_HANDLE)

    p_cl_login = sub.add_parser("social-cleanup-login", help="One-time sign-in for the delete-only cleanup client.")
    p_cl_login.add_argument("--handle", default=settings.BSKY_HANDLE)

    p_cl_run = sub.add_parser(
        "social-cleanup-run",
        help="Delete exactly the posts in a cleanup plan. Requires --confirm <plan id>; resumable (ledger next to the plan).")
    p_cl_run.add_argument("plan")
    p_cl_run.add_argument("--confirm", required=True, help="the plan id printed by social-cleanup-plan: your approval of that list")

    p_social_style = sub.add_parser(
        "social-style",
        help="Learn the owner's alt-text style from their own public Bluesky posts (inat_skeet bot posts left out) and save the profile "
             "social-draft uses; also writes <profile>.loci.json, the structured record for Loci memory.")
    p_social_style.add_argument("--handle", default=settings.BSKY_HANDLE)
    p_social_style.add_argument("--out", default=str(settings.SOCIAL_STYLE_PROFILE))

    p_social_metadata = sub.add_parser(
        "social-metadata", help="Write the poster profile's client-metadata.json, jwks.json (public key) and callback.html for publishing.")
    p_social_metadata.add_argument("out_dir")

    p_build_keywords = sub.add_parser("build-keywords")
    p_build_keywords.add_argument("folder")

    p_process = sub.add_parser(
        "process",
        help="Run the full mechanical post-curation chain for one observation "
             "folder with zero further manual calls: stack -> write-sidecars -> "
             "render-review -> export-inat (if iNat-linked) -> export-stock -> "
             "build-keywords. Assumes ingest/cull/reject/keep (and fetch-inat, "
             "if this observation is iNat-linked) already ran -- those stay "
             "separate, deliberate steps.",
    )
    p_process.add_argument("folder")
    p_process.add_argument("--method", default="focus-stack", choices=["focus-stack", "zerene"])
    p_process.add_argument("--timeout", type=float, default=1800.0)
    p_process.add_argument("--denoise", action="store_true",
                            help="Denoise+levels every raw before handing off to the "
                                 "stacking engine, same as `bio stack --denoise`")
    p_process.add_argument("--noise", type=float, default=0.0,
                            help="Fixed slb t_param1 (a SHARPENING amplitude, not a denoiser; measured 2026-10-20) 0..1, default 0, only with --denoise")
    p_process.add_argument("--exposure-ev", type=float, default=0.0,
                            help="Fixed EV pull-down applied after denoise, only with --denoise")
    p_process.add_argument("--subject-boost-ev", type=float, default=0.0,
                            help="Extra EV lift inside the subject only, only with --denoise")
    p_process.add_argument("--min-free-gb", type=float, default=5.0)
    p_process.add_argument("--skip-subject-check", action="store_true",
                            help="Skip the DA-CLIP subject-consistency safety gate before "
                                 "handing off to the stacking engine")

    p_process_all = sub.add_parser(
        "process-all",
        help="Run `process` over every observation folder under base_dir "
             "(see stack-all's own folder discovery).",
    )
    p_process_all.add_argument("base_dir")
    p_process_all.add_argument("--method", default="focus-stack", choices=["focus-stack", "zerene"])
    p_process_all.add_argument("--timeout", type=float, default=1800.0)
    p_process_all.add_argument("--denoise", action="store_true",
                                help="Denoise+levels every raw for every folder before "
                                     "handing off to the stacking engine")
    p_process_all.add_argument("--noise", type=float, default=0.0,
                                help="Fixed slb t_param1 (a SHARPENING amplitude, not a denoiser; measured 2026-10-20) 0..1, default 0, only with --denoise")
    p_process_all.add_argument("--exposure-ev", type=float, default=0.0,
                                help="Fixed EV pull-down applied after denoise, only with --denoise")
    p_process_all.add_argument("--subject-boost-ev", type=float, default=0.0,
                                help="Extra EV lift inside the subject only, only with --denoise")
    p_process_all.add_argument("--min-free-gb", type=float, default=5.0)
    p_process_all.add_argument("--skip-subject-check", action="store_true",
                                help="Skip the DA-CLIP subject-consistency safety gate before "
                                     "handing off to the stacking engine, for every folder")

    p_serve = sub.add_parser("serve", help="Start Topaz enhancement HTTP server")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8091)
    p_serve.add_argument("--device", default="AUTO",
                         help="OpenVINO device: CPU, GPU, AUTO (default: AUTO)")

    p_review_console = sub.add_parser(
        "review-console",
        help="Start the local sort/crop/cull web console over every observation "
             "folder under --base-dir. Loopback only by default -- reach it "
             "through a reverse proxy that restricts by source IP, not by "
             "binding this process to a public interface. See review_console.py.",
    )
    p_review_console.add_argument("--base-dir", default=str(settings.LIBRARY))
    p_review_console.add_argument("--host", default="127.0.0.1")
    p_review_console.add_argument("--port", type=int, default=8780)

    return parser


def cmd_ingest(args: argparse.Namespace) -> int:
    base_dir = Path(args.base_dir)
    source = Path(args.source)
    observation_dir = make_observation_path(base_dir, args.obsdate, args.slug)
    ensure_layout(observation_dir)
    copied = copy_into_raw(observation_dir, source)
    save_observation(
        observation_dir,
        Observation(
            obsdate=args.obsdate,
            source="local-ingest",
        ),
    )
    save_assets(
        observation_dir,
        [
            Asset(
                file=str(path.relative_to(observation_dir)).replace("\\", "/"),
                role="original",
                status="ingested",
                selected=False,
            )
            for path in copied
        ],
    )
    print(observation_dir)
    return 0


def cmd_auto_ingest(args: argparse.Namespace) -> int:
    try:
        result = auto_ingest_volume(Path(args.volume_path), Path(args.base_dir))
    except NotACameraVolumeError as exc:
        raise SystemExit(str(exc))
    for group in result["groups"]:
        print(group["folder"])
    print(
        f"{result['observation_folders_created']} observation folder(s) created, "
        f"{result['observation_folders_updated']} updated, "
        f"{result['files_ingested']} new file(s) ingested, "
        f"{result['files_already_present']} already present (skipped)"
    )
    return 0


def cmd_sync_stacks(args: argparse.Namespace) -> int:
    results = process_synced_stacks(
        Path(args.sync_dir), Path(args.base_dir), cleanup=not args.no_cleanup,
    )
    ok = [r for r in results if r["status"] == "ok"]
    errors = [r for r in results if r["status"] == "error"]
    for r in ok:
        print(r["observation_dir"])
    for r in errors:
        print(f"FAILED {r['stack_dir']}: {r['error']}")
    print(f"{len(ok)} stack(s) ingested and processed, {len(errors)} failed")
    return 1 if errors else 0


def cmd_cull(args: argparse.Namespace) -> int:
    folder = Path(args.folder)
    deleted = delete_rejected_raws(folder)
    for path in deleted:
        print(path)
    return 0


def cmd_presort(args: argparse.Namespace) -> int:
    """Run suggest_culling() and print a short human-readable summary.

    Deliberately does not call reject/keep/cull -- see quality.py's module
    docstring. Only reports what suggest_culling() found; applying the
    suggestions is a separate, human-reviewed step via `bio reject`/`bio keep`.
    """
    folder = Path(args.folder)
    result = suggest_culling(folder)
    summary = result["summary"]
    print(f"{summary['total']} ingested asset(s) scored: "
          f"{summary['keep_candidates']} keep candidate(s), "
          f"{summary['cull_candidates']} cull candidate(s)")
    print(folder / PRESORT_FILENAME)
    return 0


def cmd_reject(args: argparse.Namespace) -> int:
    folder = Path(args.folder)
    set_asset_status(folder, args.files, "rejected", False)
    return 0


def cmd_keep(args: argparse.Namespace) -> int:
    folder = Path(args.folder)
    set_asset_status(folder, args.files, "kept", True)
    return 0


def cmd_fetch_inat(args: argparse.Namespace) -> int:
    folder = Path(args.folder)
    existing = load_observation(folder)
    if (
        existing.observation_id is not None
        and existing.observation_id != args.observation_id
        and not args.force
    ):
        raise SystemExit(
            f"{folder} is already linked to iNat observation {existing.observation_id}; "
            f"pass --force to relink to {args.observation_id}"
        )
    fetched = fetch_inat_observation(args.observation_id, with_lineage=True)
    save_observation(folder, merge_inat_fields(existing, fetched))
    print(folder / "metadata" / "observation.json")
    return 0


def cmd_stack(args: argparse.Namespace) -> int:
    from .focus_stack import FocusStackError
    from .grouping import SubjectMismatchError
    from .zerene import ZereneError

    check_subjects = not args.skip_subject_check
    try:
        if args.denoise:
            asset = clean_and_stack(Path(args.folder), noise=args.noise,
                                     exposure_ev=args.exposure_ev,
                                     subject_boost_ev=args.subject_boost_ev,
                                     method=args.method, timeout=args.timeout,
                                     min_free_gb=args.min_free_gb,
                                     check_subjects=check_subjects)
        else:
            asset = stack_focus(Path(args.folder), method=args.method, timeout=args.timeout,
                                 min_free_gb=args.min_free_gb,
                                 check_subjects=check_subjects)
    except (FocusStackError, ZereneError, SubjectMismatchError) as exc:
        raise SystemExit(str(exc))
    print(Path(args.folder) / asset.file)
    return 0


def cmd_stack_all(args: argparse.Namespace) -> int:
    base_dir = Path(args.base_dir)
    check_subjects = not args.skip_subject_check
    results = stack_all(
        base_dir, noise=args.noise, exposure_ev=args.exposure_ev,
        subject_boost_ev=args.subject_boost_ev, method=args.method,
        timeout=args.timeout, min_free_gb=args.min_free_gb, check_subjects=check_subjects,
    )
    ok = sum(1 for row in results if row["status"] == "ok")
    failed = len(results) - ok
    print(base_dir / "stack_batch_manifest.json")
    print(f"{ok} ok, {failed} failed")
    return 0


def cmd_write_sidecars(args: argparse.Namespace) -> int:
    for path in write_sidecars(Path(args.folder)):
        print(path)
    return 0


def cmd_render_review(args: argparse.Namespace) -> int:
    for path in render_review(Path(args.folder)):
        print(path)
    return 0


def cmd_export_inat(args: argparse.Namespace) -> int:
    for path in export_inat(Path(args.folder)):
        print(path)
    return 0


def cmd_export_stock(args: argparse.Namespace) -> int:
    for path in export_stock(Path(args.folder)):
        print(path)
    return 0


def cmd_export_zip(args: argparse.Namespace) -> int:
    import getpass
    import os
    import socket
    out_dir = Path(args.out_dir or settings.env(settings.DOWNLOAD_DIR_ENV, DEFAULT_DOWNLOAD_DIR))
    try:
        result = build_bundle(args.kind, [Path(s) for s in args.sources], out_dir, name=args.name)
    except BundleError as e:
        print(f"export-zip failed: {e}", file=sys.stderr)
        return 1
    print(f"{result.path}")
    print(f"  {result.photos} photos + {len(result.extras)} extra file(s) {list(result.extras)}, "
          f"{result.size_bytes / 1e6:.1f} MB, sha256 {result.sha256}")
    print("  fetch over ssh (key auth via the existing sshd; nothing is served over HTTP from here):")
    print(f"    scp {getpass.getuser()}@{socket.gethostname()}:{result.path.as_posix()} .")
    return 0


def cmd_build_keywords(args: argparse.Namespace) -> int:
    import json

    print(json.dumps(build_and_save_keywords(Path(args.folder)), indent=2))
    return 0


def cmd_flag_upload_set(args: argparse.Namespace) -> int:
    import json

    from .upload_flags import flag_observations_csv, load_manual_notes

    try:
        manual = load_manual_notes(Path(args.notes) if args.notes else None)
        result = flag_observations_csv(Path(args.folder), manual)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc))
    print(json.dumps(result))
    return 0


def cmd_stock_calibrate(args: argparse.Namespace) -> int:
    import json

    from .stock_select import calibrate_from_folder

    exclude = None
    if args.exclude_csv:
        import csv

        with open(args.exclude_csv, newline="", encoding="utf-8-sig") as f:
            exclude = {row["Filename"] for row in csv.DictReader(f)}
    print(json.dumps(calibrate_from_folder(Path(args.folder), Path(args.out) if args.out else None, args.sample, exclude), indent=2))
    return 0


def cmd_stock_tag_queue(args: argparse.Namespace) -> int:
    from .stock_select import StockFloorMissing, tag_queue

    from .inat_stock import SnapshotStale, load_snapshot
    from .upload_flags import load_manual_notes

    try:
        notes = load_manual_notes(Path(args.notes) if args.notes else None)
        observations = load_snapshot(Path(args.observations))
        if args.created_since:
            observations = [o for o in observations if (o.get("created_at") or "")[:10] >= args.created_since]
        rows = tag_queue(Path(args.queue_json), Path(args.raw_root), Path(args.out), observations=observations,
                         floor_path=Path(args.floor) if args.floor else None, workers=args.workers, top_per_group=args.top_per_group,
                         min_similarity=args.min_similarity,
                         excluded_groups={g: "flagged " + ", ".join(v["flags"]) for g, v in notes.items()})
    except (StockFloorMissing, SnapshotStale, ValueError) as exc:
        raise SystemExit(str(exc))
    picked = [r for r in rows if r["stock_candidate"]]
    print(f"{len(picked)} Stock candidates of {len(rows)} frames in {len({r['group'] for r in picked})} of "
          f"{len({r['group'] for r in rows})} groups -> {args.out}")
    return 0


def cmd_stock_publish(args: argparse.Namespace) -> int:
    import json

    from .stock_select import publish_candidates

    queue = None
    if not args.no_queue:
        from .queue import Queue

        queue = Queue(Path(args.queue_db)) if args.queue_db else Queue()
    try:
        result = publish_candidates(Path(args.csv), Path(args.raw_root), Path(args.render_dir), Path(args.gallery_dir),
                                    plain_decode=args.plain_decode, floor_path=Path(args.floor) if args.floor else None,
                                    adobe_csv=Path(args.adobe_csv) if args.adobe_csv else None, queue=queue)
    finally:
        if queue is not None:
            queue.close()
    print(json.dumps(result, indent=2))
    return 1 if result["failed"] or result["missing"] or result.get("refused") else 0      # skipped NEF/Enhanced-NR twins are not a failure


def _expand(patterns: list[str]) -> list[Path]:
    import glob

    out = []
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if not hits:
            raise SystemExit(f"no file matches {pat}")
        out += [Path(h) for h in hits]
    return out


def cmd_social_index(args: argparse.Namespace) -> int:
    from collections import Counter

    from .social import build_obs_index, write_index

    index = build_obs_index(Path(args.gallery_dir), _expand(args.audit), _expand(args.candidates))
    write_index(index, Path(args.out))
    reasons = Counter(r.split(":")[0] for r in index["unresolved"].values())
    print(f"{len(index['files'])} files mapped to {len(set(index['files'].values()))} observations, {len(index['unresolved'])} unresolved "
          f"{dict(reasons)} -> {args.out}")
    return 0


def cmd_social_draft(args: argparse.Namespace) -> int:
    import datetime as dt
    import json

    from .social import Ledger, draft_month, load_index

    today = dt.date.fromisoformat(args.date) if args.date else dt.datetime.now(settings.LOCAL_TZ).date()
    report = draft_month(load_index(Path(args.index)), Ledger(), today=today, gallery_dir=Path(args.gallery_dir),
                         drafts_dir=settings.SOCIAL_DIR / "drafts", keep_pending=args.keep_pending)
    print(json.dumps(report, indent=2))
    return 0


def cmd_social_list(args: argparse.Namespace) -> int:
    from .social import Ledger, load_draft, segments_text

    for draft_id, d in Ledger().drafts().items():
        if args.state and d["state"] != args.state:
            continue
        line = f"{d['state']:9} {draft_id}"
        ddir = settings.SOCIAL_DIR / "drafts" / draft_id
        if not (ddir / "draft.json").is_file():
            print(line + "  (draft folder missing)")
            continue
        draft = load_draft(ddir)
        print(line + "  " + segments_text(draft["text"]).splitlines()[0])
        if args.full:
            print("    " + segments_text(draft["text"]).replace("\n", "\n    "))
            for img in draft["images"]:
                print(f"    [{ddir / img['file']}] from {img['source']}\n      alt: {img['alt']}")
            print()
    return 0


def cmd_social_approve(args: argparse.Namespace) -> int:
    from .social import Ledger, approve

    ledger = Ledger()
    for draft_id in args.draft_ids:
        approve(ledger, settings.SOCIAL_DIR / "drafts", draft_id)
        print(f"approved {draft_id}")
    return 0


def cmd_social_reject(args: argparse.Namespace) -> int:
    from .social import Ledger, reject

    ledger = Ledger()
    for draft_id in args.draft_ids:
        reject(ledger, draft_id, args.reason)
        print(f"rejected {draft_id}")
    return 0


def cmd_social_resolve(args: argparse.Namespace) -> int:
    from .social import Ledger, resolve

    rec = resolve(Ledger(), args.draft_id, posted_uri=None if args.not_posted else args.posted_uri)
    print(f"{args.draft_id}: {rec['event']}")
    return 0


def cmd_social_post(args: argparse.Namespace) -> int:
    import datetime as dt
    import json

    from .social import Ledger, post_next

    result = post_next(Ledger(), settings.SOCIAL_DIR / "drafts", now=dt.datetime.now(settings.LOCAL_TZ), live=args.live)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def _bsky(call):
    """Run a bsky-oauth call; its errors become SocialError so main() prints them as one line."""
    from bsky_oauth import BskyError

    from .social import SocialError

    try:
        return call()
    except BskyError as e:
        raise SocialError(str(e)) from e


def cmd_social_login(args: argparse.Namespace) -> int:
    from .social import poster_account

    return _bsky(lambda: poster_account().login(args.handle))


def cmd_social_whoami(args: argparse.Namespace) -> int:
    import json

    from .social import poster_account

    print(json.dumps(_bsky(lambda: poster_account().whoami()), indent=2))
    return 0


def cmd_social_cleanup_plan(args: argparse.Namespace) -> int:
    import json

    from .social_cleanup import build_plan, save_plan, summary

    plan = build_plan(args.handle)
    path = save_plan(plan)
    print(json.dumps({"plan": str(path), **summary(plan)}, indent=2, ensure_ascii=False))
    return 0


def cmd_social_cleanup_login(args: argparse.Namespace) -> int:
    from .social_cleanup import cleanup_account

    return _bsky(lambda: cleanup_account().login(args.handle))


def cmd_social_cleanup_run(args: argparse.Namespace) -> int:
    import json

    from .social_cleanup import run_plan

    report = _bsky(lambda: run_plan(Path(args.plan), args.confirm))
    print(json.dumps(report, indent=2))
    return 0 if not report["stopped"] and report["remaining"] == 0 else 1


def cmd_social_style(args: argparse.Namespace) -> int:
    import json

    from .social import embedder
    from .social_style import build_profile, loci_record, save_profile

    model, embed = embedder()
    profile = build_profile(args.handle, embed=embed, embed_model=model)
    path = save_profile(profile, Path(args.out))
    loci = path.with_suffix(".loci.json")
    loci.write_text(json.dumps(loci_record(profile), indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"profile": str(path), "loci_record": str(loci), "source": profile["source"], "stats": profile["stats"],
                      "rules": profile["rules"]}, indent=2, ensure_ascii=False))
    return 0


def cmd_social_metadata(args: argparse.Namespace) -> int:
    import json

    from .social import poster_account

    print(json.dumps(_bsky(lambda: poster_account().metadata(Path(args.out_dir))), indent=2))
    return 0


SOCIAL_COMMANDS = {"social-index": cmd_social_index, "social-draft": cmd_social_draft, "social-list": cmd_social_list,
                   "social-approve": cmd_social_approve, "social-reject": cmd_social_reject, "social-resolve": cmd_social_resolve,
                   "social-post": cmd_social_post, "social-login": cmd_social_login, "social-whoami": cmd_social_whoami,
                   "social-metadata": cmd_social_metadata, "social-style": cmd_social_style,
                   "social-cleanup-plan": cmd_social_cleanup_plan, "social-cleanup-login": cmd_social_cleanup_login,
                   "social-cleanup-run": cmd_social_cleanup_run}


def cmd_process(args: argparse.Namespace) -> int:
    import json

    from .focus_stack import FocusStackError
    from .grouping import SubjectMismatchError
    from .zerene import ZereneError

    check_subjects = not args.skip_subject_check
    try:
        result = process_observation(
            Path(args.folder), method=args.method, timeout=args.timeout,
            min_free_gb=args.min_free_gb, check_subjects=check_subjects,
            denoise=args.denoise, noise=args.noise, exposure_ev=args.exposure_ev,
            subject_boost_ev=args.subject_boost_ev,
        )
    except (FocusStackError, ZereneError, SubjectMismatchError) as exc:
        raise SystemExit(str(exc))
    print(json.dumps(result, indent=2))
    return 0


def cmd_process_all(args: argparse.Namespace) -> int:
    base_dir = Path(args.base_dir)
    check_subjects = not args.skip_subject_check
    results = process_all(
        base_dir, method=args.method, timeout=args.timeout, min_free_gb=args.min_free_gb,
        check_subjects=check_subjects, denoise=args.denoise, noise=args.noise,
        exposure_ev=args.exposure_ev, subject_boost_ev=args.subject_boost_ev,
    )
    ok = sum(1 for row in results if row["status"] == "ok")
    failed = len(results) - ok
    print(base_dir / "process_batch_manifest.json")
    print(f"{ok} ok, {failed} failed")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import subprocess

    serve_script = Path(__file__).parent.parent.parent / "tools" / "topaz_serve.py"
    cmd = [
        sys.executable, str(serve_script),
        "--host", args.host,
        "--port", str(args.port),
        "--device", args.device,
    ]
    raise SystemExit(subprocess.call(cmd))


def cmd_review_console(args: argparse.Namespace) -> int:
    from .review_console import serve

    serve(Path(args.base_dir), host=args.host, port=args.port)
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "ingest":
        return cmd_ingest(args)
    if args.command == "auto-ingest":
        return cmd_auto_ingest(args)
    if args.command == "sync-stacks":
        return cmd_sync_stacks(args)
    if args.command == "cull":
        return cmd_cull(args)
    if args.command == "presort":
        return cmd_presort(args)
    if args.command == "reject":
        return cmd_reject(args)
    if args.command == "keep":
        return cmd_keep(args)
    if args.command == "fetch-inat":
        return cmd_fetch_inat(args)
    if args.command == "stack":
        return cmd_stack(args)
    if args.command == "stack-all":
        return cmd_stack_all(args)
    if args.command == "write-sidecars":
        return cmd_write_sidecars(args)
    if args.command == "render-review":
        return cmd_render_review(args)
    if args.command == "export-inat":
        return cmd_export_inat(args)
    if args.command == "export-stock":
        return cmd_export_stock(args)
    if args.command == "export-zip":
        return cmd_export_zip(args)
    if args.command == "flag-upload-set":
        return cmd_flag_upload_set(args)
    if args.command == "stock-calibrate":
        return cmd_stock_calibrate(args)
    if args.command == "stock-tag-queue":
        return cmd_stock_tag_queue(args)
    if args.command == "stock-publish":
        return cmd_stock_publish(args)
    if args.command.startswith("social-"):
        from .social import SocialError

        try:
            return SOCIAL_COMMANDS[args.command](args)
        except SocialError as e:          # the message says what to do; a traceback would only bury it
            print(f"bio {args.command}: {e}", file=sys.stderr)
            return 1
    if args.command == "build-keywords":
        return cmd_build_keywords(args)
    if args.command == "process":
        return cmd_process(args)
    if args.command == "process-all":
        return cmd_process_all(args)
    if args.command == "serve":
        return cmd_serve(args)
    if args.command == "review-console":
        return cmd_review_console(args)
    raise SystemExit(2)


if __name__ == "__main__":
    raise SystemExit(main())
