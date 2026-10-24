"""Run the ported Topaz subject_mask() over weak_train_pool.json and extract each mask's
bounding box (not just area fraction) to seed SAM3 box-prompting. Non-degenerate masks
give a real bbox; degenerate ones (>85% of frame) give bbox=None (falls back to a
centered-box heuristic downstream). Per-image timeout guard (90s) since one photo took
7.5+ minutes in an earlier run on this same pipeline (content-dependent slow path, not a
hang -- see batch_topaz_general.py's docstring). Run with F:\\venvs\\bio-sr-train\\Scripts\\python.exe.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from topaz_subject_mask_port import ModelCache, subject_mask  # noqa: E402

import numpy as np
from PIL import Image

ITEMS = Path(r"D:\dev\bio-sr-train\tools\sr_train\sam3_eval\weak_train_pool.json")
OUT_DIR = Path(r"D:\dev\bio-sr-train\tools\sr_train\sam3_eval")
PER_IMAGE_TIMEOUT_S = 90


def _process_one(item):
    img = Image.open(item["path"]).convert("RGB")
    w, h = img.size
    img_chw_f32 = np.transpose(np.array(img, dtype=np.float32) / 255.0, (2, 0, 1))
    prob = subject_mask(img_chw_f32, _CACHE)
    mask_area_frac = float((prob > 0.5).mean())
    degenerate = mask_area_frac > 0.85
    bbox = None
    if not degenerate and mask_area_frac > 0.001:
        ys, xs = np.where(prob > 0.5)
        if len(ys) > 0:
            x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
            bbox = [x0 / w, y0 / h, x1 / w, y1 / h]  # normalized xyxy
    return mask_area_frac, degenerate, bbox, w, h


def main():
    global _CACHE
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    all_items = json.loads(ITEMS.read_text(encoding="utf-8"))
    items = all_items[args.shard_index::args.num_shards]
    suffix = f".shard{args.shard_index}of{args.num_shards}" if args.num_shards > 1 else ""
    out = OUT_DIR / f"weak_train_pool_topaz_bbox{suffix}.json"
    partial = OUT_DIR / f"weak_train_pool_topaz_bbox{suffix}.partial.json"

    _CACHE = ModelCache()
    results = []
    executor = ThreadPoolExecutor(max_workers=1)
    for i, item in enumerate(items):
        t0 = time.time()
        try:
            future = executor.submit(_process_one, item)
            mask_area_frac, degenerate, bbox, w, h = future.result(timeout=PER_IMAGE_TIMEOUT_S)
            results.append({**item, "topaz_mask_area_frac": mask_area_frac,
                             "topaz_degenerate": degenerate, "topaz_bbox_norm_xyxy": bbox,
                             "width": w, "height": h, "error": None})
        except FutureTimeoutError:
            results.append({**item, "error": f"timeout>{PER_IMAGE_TIMEOUT_S}s"})
            executor.shutdown(wait=False, cancel_futures=False)
            executor = ThreadPoolExecutor(max_workers=1)
        except Exception as e:
            results.append({**item, "error": str(e)})
        print(f"shard{args.shard_index} [{i+1}/{len(items)}] {item['iconic']:10s} obs_id={item['obs_id']} "
              f"-> degenerate={results[-1].get('topaz_degenerate')} "
              f"bbox={results[-1].get('topaz_bbox_norm_xyxy')} err={results[-1].get('error')} "
              f"({time.time()-t0:.1f}s)")
        partial.write_text(json.dumps(results, indent=2))

    out.write_text(json.dumps(results, indent=2))
    print(f"\nshard{args.shard_index}: wrote {len(results)} results to {out}")


if __name__ == "__main__":
    main()
