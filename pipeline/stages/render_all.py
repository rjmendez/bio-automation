"""Render the TARGET raws (triage_targets.json from plan_backlog.py: untouched days only) to an iNaturalist-ready JPEG: LightCraft Auto, subtle denoise at
ISO >= 800, 2048 px long edge (iNaturalist's own maximum), sRGB, q90, camera EXIF kept. No culling, no selection.

Keywords/title are written afterwards by tag_raws.py (lossless segment surgery on these files). Newest days first. Resumable: existing
outputs are skipped. Each photo is reset to pristine in the catalog afterwards (non-destructive). Free-space guard: BIO_MIN_FREE_GB.
Headless engine: BIO_LC_HEADLESS=1 (LightCraft GUI closed).
"""
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))
from pilot_style import Mcp, text, ROOT
from bio_automation import disk_guard

WHO = 'render_all'

OUT = os.environ.get('BIO_RENDER_OUT', r'F:\Bio-export\inat-raws')
MIN_FREE_GB = float(os.environ.get('BIO_MIN_FREE_GB', '20'))
ISO_DENOISE = 800
LOG = os.path.join(ROOT, 'render_all.log')


def log(msg):
    line = time.strftime('%H:%M:%S ') + msg
    print(line, flush=True)
    open(LOG, 'a', encoding='utf-8').write(line + '\n')


def main():
    targets = json.load(open(os.path.join(ROOT, 'triage_targets.json'), encoding='utf-8'))
    todo = [dict(r, kind=targets[str(r['id'])]['kind']) for r in json.load(open(os.path.join(ROOT, 'triage_index.json'), encoding='utf-8'))
            if str(r['id']) in targets]
    todo.sort(key=lambda r: (r['kind'] != 'backlog', r['day'], r['id']))     # resume after the last worked day, oldest first; gap days last
    os.makedirs(OUT, exist_ok=True)
    m = Mcp()
    done = skipped = failed = 0
    t0 = time.time()
    for i, r in enumerate(todo, 1):
        fid = r['id']
        safe = ''.join(c if c not in '<>:"/\\|?*' else '_' for c in r['stem'])
        base = OUT if r['kind'] == 'backlog' else os.path.join(OUT, '_gap-days')
        out = os.path.join(base, r['day'][:4], r['day'], f"{r['day']}_{safe}_{fid}.jpg")
        if os.path.exists(out) and os.path.getsize(out) > 0:
            done += 1
            continue
        if not disk_guard.wait_for_output(OUT, MIN_FREE_GB, who=WHO, say=log):    # pauses, resumes by itself
            free = shutil.disk_usage(OUT).free / 1e9
            log(f'STOP: only {free:.0f} GB free (limit {MIN_FREE_GB:.0f}); {len(todo) - i + 1} frames left')
            break
        os.makedirs(os.path.dirname(out), exist_ok=True)
        pristine = False
        try:
            ins = text(m.call('cmd_photo_inspect', {'id': fid})) or {}
            pristine = not ins.get('edited')
            iso = ((ins.get('meta') or {}).get('iso') if isinstance(ins, dict) else None) or 0
            m.call('select_photos', {'ids': [fid], 'active': fid})
            if pristine:
                m.call('command_run', {'id': 'develop.auto', 'params': {}})      # photos you edited are exported as they are
            if pristine and iso >= ISO_DENOISE and r['kind'] == 'backlog':       # gap days: CPU denoise costs 6-10 min per frame, skipped
                m.call('command_run', {'id': 'denoise.toggle', 'params': {'id': fid, 'enabled': True}})
                m.call('set_develop', {'id': fid, 'values': {'enhance.denoise': 25.0}, 'label': 'render: subtle AI denoise'})
                m.call('command_run', {'id': 'denoise.queue', 'params': {'ids': [fid]}})
                calm, t2 = 0, time.time()
                while time.time() - t2 < 900 and calm < 2:
                    st = text(m.call('command_run', {'id': 'denoise.status', 'params': {}})) or {}
                    busy = isinstance(st, dict) and ((st.get('queued', 0) or 0) > 0 or st.get('running') not in (None, False))
                    calm = 0 if busy else calm + 1
                    time.sleep(4)
            res = m.call('export', {'id': fid, 'path': out, 'format': 'jpeg', 'quality': 90, 'longEdge': 2048, 'colorSpace': 'srgb',
                                    'metadata': 'all', 'conflict': 'overwrite'})
            if not os.path.exists(out):
                t1, last = time.time(), -1
                while time.time() - t1 < 420:
                    if os.path.exists(out):
                        sz = os.path.getsize(out)
                        if sz > 0 and sz == last:
                            break
                        last = sz
                    time.sleep(8)
            if os.path.exists(out) and os.path.getsize(out) > 0:
                done += 1
            else:
                failed += 1
                log(f'FAILED {fid} {r["stem"]}: {str(res)[:120]}')
        except Exception as e:
            failed += 1
            log(f'ERROR {fid} {r["stem"]}: {type(e).__name__} {str(e)[:100]}')
        finally:
            try:
                if pristine:
                    m.call('command_run', {'id': 'denoise.toggle', 'params': {'id': fid, 'enabled': False}})
                    m.call('command_run', {'id': 'develop.reset', 'params': {'ids': [fid]}})
            except Exception:
                pass
        if i % 25 == 0:
            el = time.time() - t0
            log(f'{i}/{len(todo)} done {done} failed {failed} {el / i:.1f}s per frame eta {(len(todo) - i) * el / i / 60:.0f} min')
    log(f'finished: rendered/present {done}, failed {failed}')


if __name__ == '__main__':
    main()
