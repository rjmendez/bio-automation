from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .auto_ingest import NotACameraVolumeError, auto_ingest_volume
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
    p_ingest.add_argument("--base-dir", default=r"F:\Bio")
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
    p_auto_ingest.add_argument("--base-dir", default=r"F:\Bio")

    p_sync_stacks = sub.add_parser(
        "sync-stacks",
        help="Ingest focus-stacks synced in from the NikonStacker Pi (nikon-z7ii-tools/"
             "stack_buffer.py) and auto-process each one (stack/sidecars/review/export) "
             "on arrival. Idempotent: a stack already ingested is skipped on rerun. "
             "See bio_automation.stack_sync.",
    )
    p_sync_stacks.add_argument("--sync-dir", default=r"D:\nikon_stack_sync")
    p_sync_stacks.add_argument("--base-dir", default=r"F:\Bio")
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
    p_review_console.add_argument("--base-dir", default=r"F:\Bio")
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
    fetched = fetch_inat_observation(args.observation_id)
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


def cmd_build_keywords(args: argparse.Namespace) -> int:
    import json

    print(json.dumps(build_and_save_keywords(Path(args.folder)), indent=2))
    return 0


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
