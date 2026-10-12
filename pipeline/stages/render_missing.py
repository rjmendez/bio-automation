"""Render the matched NEFs that have no existing JPEG export: LightCraft Auto -> full-size sRGB JPEG, then reset the photo.

Default look = LightCraft 'Auto Settings' (what the owner uses on most images in Lightroom). CPU rendering, location removed.
Non-destructive: only photos that were pristine are touched, and each is reset to pristine afterwards.
Resumable: skips outputs that already exist. Output tree mirrors YYYY/YYYY-MM-DD so embed_meta.py can use it as a source.
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lightcraft_mcp import Mcp, text, ROOT

OUT = r'F:\Bio-export\adobe-stock\rendered'
ISO_DENOISE = 800          # measured denoise pilot: below this there is little noise to remove and the detail cost is the same
todo =json.load(open(os.path.join(ROOT, 'missing_exports.json'), encoding='utf-8'))
LOG = os.path.join(ROOT, 'render_missing.log')


def log(msg):
    line = time.strftime('%H:%M:%S ') + msg
    print(line, flush=True)
    open(LOG, 'a', encoding='utf-8').write(line + '\n')


def main():
    m = Mcp()
    ed = text(m.call('query_photos', {'filter': {'edited': True}, 'limit': 500})) or {}
    edited = {p['id'] for p in (ed.get('photos') if isinstance(ed, dict) else [])}
    done = skipped = failed = 0
    t0 = time.time()
    for i, r in enumerate(todo, 1):
        fid = r['frame_id']
        stem = os.path.splitext(r['file'])[0]
        safe = ''.join(c if c not in '<>:"/\\|?*' else '_' for c in stem)
        out = os.path.join(OUT, *r['day'].split('/'), safe + '.jpg')
        if os.path.exists(out) and os.path.getsize(out) > 0:
            done += 1
            continue
        if fid in edited:
            skipped += 1
            log(f'skip {fid} {r["file"]}: already has edits, not touching it')
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        try:
            m.call('select_photos', {'ids': [fid], 'active': fid})
            ins = text(m.call('cmd_photo_inspect', {'id': fid})) or {}
            iso = ((ins.get('meta') or {}).get('iso') if isinstance(ins, dict) else None) or 0
            m.call('command_run', {'id': 'develop.auto', 'params': {}})
            denoised = iso >= ISO_DENOISE
            if denoised:                                   # measured pilot: amount 25 = ~25% less noise for 8-17% less fine detail
                m.call('command_run', {'id': 'denoise.toggle', 'params': {'id': fid, 'enabled': True}})
                m.call('set_develop', {'id': fid, 'values': {'enhance.denoise': 25.0}, 'label': 'render: subtle AI denoise'})
                m.call('command_run', {'id': 'denoise.queue', 'params': {'ids': [fid]}})
                calm = 0
                t2 = time.time()
                while time.time() - t2 < 600 and calm < 2:
                    st = text(m.call('command_run', {'id': 'denoise.status', 'params': {}})) or {}
                    busy = isinstance(st, dict) and ((st.get('queued', 0) or 0) > 0 or st.get('running') not in (None, False))
                    calm = 0 if busy else calm + 1
                    time.sleep(4)
            res = m.call('export', {'id': fid, 'path': out, 'format': 'jpeg', 'quality': 93, 'longEdge': 0, 'colorSpace': 'srgb',
                                    'metadata': 'allExceptCamera' if False else 'all', 'removeLocation': True, 'conflict': 'overwrite'})
            if not os.path.exists(out):                      # the control channel can time out while the app still renders
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
                log(f'FAILED {fid} {r["file"]}: {str(res)[:120]}')
        except Exception as e:
            failed += 1
            log(f'ERROR {fid} {r["file"]}: {type(e).__name__} {str(e)[:100]}')
        finally:
            try:
                m.call('command_run', {'id': 'denoise.toggle', 'params': {'id': fid, 'enabled': False}})
                m.call('command_run', {'id': 'develop.reset', 'params': {'ids': [fid]}})     # back to pristine, always
            except Exception:
                pass
        if i % 10 == 0:
            el = time.time() - t0
            log(f'{i}/{len(todo)}  done {done}  skipped {skipped}  failed {failed}  {el / max(i, 1):.0f}s per frame  eta {(len(todo) - i) * el / max(i, 1) / 60:.0f} min')
    log(f'finished: rendered/present {done}, skipped {skipped}, failed {failed}')


if __name__ == '__main__':
    main()
