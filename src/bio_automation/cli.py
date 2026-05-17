from __future__ import annotations

import argparse
from pathlib import Path

from .paths import make_observation_path
from .pipeline import (
    copy_into_raw,
    delete_rejected_raws,
    ensure_layout,
    save_assets,
    save_observation,
    set_asset_status,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bio")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest")
    p_ingest.add_argument("source")
    p_ingest.add_argument("--base-dir", default=r"F:\Bio")
    p_ingest.add_argument("--obsdate", required=True)
    p_ingest.add_argument("--slug", required=True)

    p_cull = sub.add_parser("cull")
    p_cull.add_argument("folder")

    p_reject = sub.add_parser("reject")
    p_reject.add_argument("folder")
    p_reject.add_argument("files", nargs="+")

    p_keep = sub.add_parser("keep")
    p_keep.add_argument("folder")
    p_keep.add_argument("files", nargs="+")

    return parser


def cmd_ingest(args: argparse.Namespace) -> int:
    base_dir = Path(args.base_dir)
    source = Path(args.source)
    observation_dir = make_observation_path(base_dir, args.obsdate, args.slug)
    ensure_layout(observation_dir)
    copied = copy_into_raw(observation_dir, source)
    save_observation(
        observation_dir,
        {
            "observation_id": None,
            "obsdate": args.obsdate,
            "source": "local-ingest",
            "place_guess": str(observation_dir.parent.parent),
            "taxon": {},
            "locality": None,
            "license": None,
            "quality_grade": None,
            "keywords_seed": [],
        },
    )
    save_assets(
        observation_dir,
        [
            {
                "file": str(path.relative_to(observation_dir)).replace("\\", "/"),
                "role": "original",
                "status": "ingested",
                "selected": False,
            }
            for path in copied
        ],
    )
    print(observation_dir)
    return 0


def cmd_cull(args: argparse.Namespace) -> int:
    folder = Path(args.folder)
    deleted = delete_rejected_raws(folder)
    for path in deleted:
        print(path)
    return 0


def cmd_reject(args: argparse.Namespace) -> int:
    folder = Path(args.folder)
    set_asset_status(folder, args.files, "rejected", False)
    return 0


def cmd_keep(args: argparse.Namespace) -> int:
    folder = Path(args.folder)
    set_asset_status(folder, args.files, "kept", True)
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "ingest":
        return cmd_ingest(args)
    if args.command == "cull":
        return cmd_cull(args)
    if args.command == "reject":
        return cmd_reject(args)
    if args.command == "keep":
        return cmd_keep(args)
    raise SystemExit(2)


if __name__ == "__main__":
    raise SystemExit(main())
