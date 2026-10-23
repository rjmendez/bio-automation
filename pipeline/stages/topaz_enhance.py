"""Topaz-enhance the triage-selected stock renders.

Reads the rendered JPEGs from triage_selection.json and runs lens-blur
sharpening (and optional 2× upscale) using extracted Topaz OpenVINO models.

Writes results back to the same path (in-place overwrite), preserving the
existing JPEG filename. Resumable: a sidecar flag file <stem>.enhanced is
written on success; files that already have a sidecar are skipped.

Env vars:
  BIO_TOPAZ_MODE     stock|inat|upscale  (default: stock)
  BIO_TOPAZ_BLUR     0.0–1.0             (default: 0.5)
  BIO_TOPAZ_NOISE    0.0–1.0             (default: 0.0)
  BIO_RENDER_OUT     base path of rendered JPEGs
  BIO_MIN_FREE_GB    stop threshold in GB (default: 10)
"""
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))

from bio_automation import disk_guard, topaz

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUT = os.environ.get('BIO_RENDER_OUT', r'F:\Bio-export\adobe-stock\rendered_new')
MODE = os.environ.get('BIO_TOPAZ_MODE', 'stock')
BLUR = float(os.environ.get('BIO_TOPAZ_BLUR', '0.5'))
NOISE = float(os.environ.get('BIO_TOPAZ_NOISE', '0.0'))
MIN_FREE_GB = float(os.environ.get('BIO_MIN_FREE_GB', '10'))
LOG = os.path.join(ROOT, 'topaz_enhance.log')


def log(msg):
    line = time.strftime('%H:%M:%S ') + msg
    print(line, flush=True)
    open(LOG, 'a', encoding='utf-8').write(line + '\n')


def main():
    sel_path = os.path.join(ROOT, 'triage_selection.json')
    todo = json.load(open(sel_path, encoding='utf-8'))

    done = skipped = failed = 0
    t0 = time.time()

    with topaz.ModelCache() as cache:
        for i, r in enumerate(todo, 1):
            fid = r['id']
            safe = ''.join(c if c not in '<>:"/\\|?*' else '_' for c in r['stem'])
            jpg = os.path.join(OUT, r['day'][:4], r['day'], f"{safe}_{fid}.jpg")
            sidecar = jpg + '.enhanced'

            if os.path.exists(sidecar):
                skipped += 1
                continue

            if not os.path.exists(jpg):
                log(f'MISSING {fid} {r["stem"]}: {jpg}')
                failed += 1
                continue

            if not disk_guard.wait_for_room(disk_guard.output_floors(OUT, MIN_FREE_GB), who='topaz_enhance', audit=disk_guard.log_event, say=log):
                free = shutil.disk_usage(OUT).free / 1e9
                log(f'STOP: only {free:.0f} GB free (limit {MIN_FREE_GB:.0f}); '
                    f'{len(todo) - i + 1} frames left')
                break

            try:
                from pathlib import Path
                topaz.enhance(
                    Path(jpg), Path(jpg),
                    mode=MODE,
                    cache=cache,
                    min_free_gb=MIN_FREE_GB,
                )
                open(sidecar, 'w').close()
                done += 1
            except RuntimeError as e:
                # Model not yet fully downloaded (truncated ZIP)
                if "not yet fully downloaded" in str(e):
                    log(f'SKIP (sharpen model not yet downloaded): {fid} {r["stem"]}')
                    skipped += 1
                else:
                    failed += 1
                    log(f'ERROR {fid} {r["stem"]}: {type(e).__name__} {str(e)[:120]}')
            except Exception as e:
                failed += 1
                log(f'ERROR {fid} {r["stem"]}: {type(e).__name__} {str(e)[:120]}')

            if i % 10 == 0:
                el = time.time() - t0
                eta_min = (len(todo) - i) * el / i / 60
                log(f'{i}/{len(todo)} done={done} skipped={skipped} failed={failed} '
                    f'{el / i:.1f}s/img eta {eta_min:.0f} min')

    log(f'finished: enhanced={done} skipped={skipped} failed={failed}')


if __name__ == '__main__':
    main()
