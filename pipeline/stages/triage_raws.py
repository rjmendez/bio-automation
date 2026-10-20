"""Triage step 1: resolve the unprocessed raw frames (no JPEG anywhere, unprocessed_raws.json, source F) to LightCraft library ids and
write a 768 px preview of each (LightCraft render_preview, default develop, nothing saved to the catalog).

Output: triage_index.json  [{id, day, file, raw}]   and   triage_thumbs/<id>.jpg   (resumable: existing thumbs are skipped).
Headless engine: BIO_LC_HEADLESS=1 (GUI must be closed).
"""
import collections
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilot_style import Mcp, text, image

ROOT = os.environ.get('BIO_DATA', r'F:\bio-automation-data')
THUMBS = os.path.join(ROOT, 'triage_thumbs')
os.makedirs(THUMBS, exist_ok=True)


def main():
    un = [u for u in json.load(open(os.path.join(ROOT, 'unprocessed_raws.json'))) if u['src'] == 'F']
    by_day = collections.defaultdict(list)
    for u in un:
        by_day[u['day']].append(u)
    m = Mcp()
    m.call('command_run', {'id': 'library.source', 'params': {'kind': 'all'}})
    index, missing = [], 0
    for n, (day, items) in enumerate(sorted(by_day.items()), 1):
        want = {os.path.basename(u['raw']).lower(): u for u in items}
        folder = f"F:/Bio/{day[:4]}/{day}"
        off, got = 0, {}
        while True:
            r = text(m.call('query_photos', {'filter': {'libraryFolder': folder}, 'offset': off, 'limit': 500})) or {}
            ph = r.get('photos', [])
            for p in ph:
                got[p['fileName'].lower()] = p['id']
            off += len(ph)
            if not ph or off >= r.get('total', 0):
                break
        for fn, u in want.items():
            pid = got.get(fn)
            if pid is None:
                missing += 1
                continue
            index.append({'id': pid, 'day': day, 'file': u['raw'], 'stem': u['stem']})
        if n % 20 == 0:
            print(f'  resolved {n}/{len(by_day)} days, {len(index)} frames, {missing} not found', flush=True)
    json.dump(index, open(os.path.join(ROOT, 'triage_index.json'), 'w'))
    print(f'index: {len(index)} frames (not found in library: {missing})', flush=True)

    t0, done, fail = time.time(), 0, 0
    todo = [i for i in index if not os.path.exists(os.path.join(THUMBS, f"{i['id']}.jpg"))]
    print('thumbs to make:', len(todo), flush=True)
    for k, it in enumerate(todo, 1):
        try:
            im = image(m.call('render_preview', {'id': it['id'], 'max_side': 768}))
            im.save(os.path.join(THUMBS, f"{it['id']}.jpg"), 'JPEG', quality=85)
            done += 1
        except Exception as e:
            fail += 1
            print('  fail', it['id'], type(e).__name__, str(e)[:80], flush=True)
        if k % 100 == 0:
            el = time.time() - t0
            print(f'  thumbs {k}/{len(todo)}  {el / k:.2f}s each  eta {(len(todo) - k) * el / k / 60:.0f} min  fail {fail}', flush=True)
    print('thumbs done', done, 'failed', fail, flush=True)


if __name__ == '__main__':
    main()
