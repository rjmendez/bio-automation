"""Batch-run the ported Topaz subject_mask() over the cross-taxon general_set.json.
Run with F:\\venvs\\bio-sr-train\\Scripts\\python.exe.

Per-image timeout (90s) via a worker thread: one real image (a normal 45MP file, nothing
pathological about its dimensions) took 7.5+ minutes and climbing with no end in sight
while every other image in this and the prior spider batch finished in seconds to ~2
minutes even at 95MP -- almost certainly a content-dependent slow path inside
_local_background_hue/refine_mask_by_color, not investigated further here. A timeout
keeps one such image from blocking the whole batch; timed-out images are recorded with
error="timeout>90s", not silently dropped. Results are saved incrementally so a kill
doesn't lose completed work.
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from topaz_subject_mask_port import ModelCache, subject_mask  # noqa: E402

import numpy as np
from PIL import Image

ITEMS = Path(r"D:\dev\bio-sr-train\tools\sr_train\sam3_eval\general_set.json")
OUT_DIR = Path(r"D:\dev\bio-sr-train\tools\sr_train\sam3_eval\results_general\topaz")
OUT_DIR.mkdir(parents=True, exist_ok=True)
PER_IMAGE_TIMEOUT_S = 90


def _process_one(item):
    img = Image.open(item["path"]).convert("RGB")
    img_chw_f32 = np.transpose(np.array(img, dtype=np.float32) / 255.0, (2, 0, 1))
    prob = subject_mask(img_chw_f32, _CACHE)
    mask_area_frac = float((prob > 0.5).mean())
    thumb = Image.fromarray((prob * 255).clip(0, 255).astype(np.uint8), mode="L")
    thumb.thumbnail((400, 400))
    thumb.save(OUT_DIR / f"{item['obs_id']}_mask.png")
    return mask_area_frac, float(prob.mean())


def main():
    global _CACHE
    items = json.loads(ITEMS.read_text(encoding="utf-8"))
    _CACHE = ModelCache()
    results = []
    executor = ThreadPoolExecutor(max_workers=1)
    for i, item in enumerate(items):
        t0 = time.time()
        try:
            future = executor.submit(_process_one, item)
            mask_area_frac, mask_mean = future.result(timeout=PER_IMAGE_TIMEOUT_S)
            results.append({**item, "mask_area_frac": mask_area_frac, "mask_mean": mask_mean,
                             "elapsed_s": time.time() - t0, "error": None})
        except FutureTimeoutError:
            results.append({**item, "mask_area_frac": None,
                             "error": f"timeout>{PER_IMAGE_TIMEOUT_S}s"})
            executor.shutdown(wait=False, cancel_futures=False)
            executor = ThreadPoolExecutor(max_workers=1)  # abandon the stuck thread, start fresh
        except Exception as e:
            results.append({**item, "mask_area_frac": None, "error": str(e)})
        print(f"[{i+1}/{len(items)}] {item['iconic']:12s} obs_id={item['obs_id']} "
              f"taxon={item['taxon']!r} -> {results[-1].get('mask_area_frac')} "
              f"err={results[-1].get('error')} ({time.time()-t0:.1f}s)")
        (OUT_DIR / "_summary.partial.json").write_text(json.dumps(results, indent=2))

    (OUT_DIR / "_summary.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {len(results)} results to {OUT_DIR / '_summary.json'}")


if __name__ == "__main__":
    main()
