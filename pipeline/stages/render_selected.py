"""Render the triage-selected raws: LightCraft Auto -> full-size sRGB JPEG (location removed), subtle denoise at ISO >= 800, then reset.

Same logic as render_missing.py. Differences: input is triage_selection.json; output F:\\Bio-export\\adobe-stock\\rendered_new\\YYYY\\YYYY-MM-DD\\;
a disk guard stops cleanly when free space on the output drive falls under MIN_FREE_GB; resumable (existing outputs are skipped).
Headless engine: BIO_LC_HEADLESS=1 (GUI closed).
"""
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilot_style import Mcp, text, ROOT

OUT = os.environ.get('BIO_RENDER_OUT', r'F:\Bio-export\adobe-stock\rendered_new')
MIN_FREE_GB = float(os.environ.get('BIO_MIN_FREE_GB', '35'))
ISO_DENOISE = 800
LOG = os.path.join(ROOT, 'render_selected.log')


def log(msg):
    line = time.strftime('%H:%M:%S ') + msg
    print(line, flush=True)
    open(LOG, 'a', encoding='utf-8').write(line + '\n')


def main():
    todo = json.load(open(os.path.join(ROOT, 'triage_selection.json'), encoding='utf-8'))
    os.makedirs(OUT, exist_ok=True)
    m = Mcp()
    done = skipped = failed = 0
    t0 = time.time()
    for i, r in enumerate(todo, 1):
        fid = r['id']
        safe = ''.join(c if c not in '<>:"/\\|?*' else '_' for c in r['stem'])
        out = os.path.join(OUT, r['day'][:4], r['day'], f"{safe}_{fid}.jpg")
        if os.path.exists(out) and os.path.getsize(out) > 0:
            done += 1
            continue
        free = shutil.disk_usage(OUT).free / 1e9
        if free < MIN_FREE_GB:
            log(f'STOP: only {free:.0f} GB free on the output drive (limit {MIN_FREE_GB:.0f}); {len(todo) - i + 1} frames left')
            break
        os.makedirs(os.path.dirname(out), exist_ok=True)
        try:
            ins = text(m.call('cmd_photo_inspect', {'id': fid})) or {}
            if ins.get('edited'):
                skipped += 1
                log(f'skip {fid} {r["stem"]}: already has edits, not touching it')
                continue
            iso = ((ins.get('meta') or {}).get('iso') if isinstance(ins, dict) else None) or 0
            m.call('select_photos', {'ids': [fid], 'active': fid})
            m.call('command_run', {'id': 'develop.auto', 'params': {}})
            if iso >= ISO_DENOISE:
                m.call('command_run', {'id': 'denoise.toggle', 'params': {'id': fid, 'enabled': True}})
                m.call('set_develop', {'id': fid, 'values': {'enhance.denoise': 25.0}, 'label': 'render: subtle AI denoise'})
                m.call('command_run', {'id': 'denoise.queue', 'params': {'ids': [fid]}})
                calm, t2 = 0, time.time()
                while time.time() - t2 < 900 and calm < 2:
                    st = text(m.call('command_run', {'id': 'denoise.status', 'params': {}})) or {}
                    busy = isinstance(st, dict) and ((st.get('queued', 0) or 0) > 0 or st.get('running') not in (None, False))
                    calm = 0 if busy else calm + 1
                    time.sleep(4)
            res = m.call('export', {'id': fid, 'path': out, 'format': 'jpeg', 'quality': 93, 'longEdge': 0, 'colorSpace': 'srgb',
                                    'metadata': 'all', 'removeLocation': True, 'conflict': 'overwrite'})
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
                m.call('command_run', {'id': 'denoise.toggle', 'params': {'id': fid, 'enabled': False}})
                m.call('command_run', {'id': 'develop.reset', 'params': {'ids': [fid]}})
            except Exception:
                pass
        if i % 10 == 0:
            el = time.time() - t0
            log(f'{i}/{len(todo)} done {done} skipped {skipped} failed {failed} {el / i:.0f}s per frame eta {(len(todo) - i) * el / i / 60:.0f} min')
    log(f'finished: rendered/present {done}, skipped {skipped}, failed {failed}')


if __name__ == '__main__':
    main()
