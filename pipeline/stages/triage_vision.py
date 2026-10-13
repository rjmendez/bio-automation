"""Triage step 2: vision + embedding over the triage thumbnails (same prompt and model choice as vision_files.py).

Runs alongside triage_raws.py: processes whichever thumbs exist, repeats until every indexed frame is done (or no progress for 15 min).
Output: triage_vision.json {id: {tags: {...}}}, triage_vectors.npz {ids, vectors} (SigLIP, 768-d, L2-normalised).
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vision_files as vf
from embed_index import Embedder

ROOT = vf.ROOT
THUMBS = os.path.join(ROOT, 'triage_thumbs')
OUT = os.path.join(ROOT, 'triage_vision.json')
VEC = os.path.join(ROOT, 'triage_vectors.npz')


def main():
    model = vf.pick_model()
    emb = Embedder()
    done = json.load(open(OUT, encoding='utf-8')) if os.path.exists(OUT) else {}
    vec = {}
    if os.path.exists(VEC):
        z = np.load(VEC)
        vec = {int(i): v for i, v in zip(z['ids'], z['vectors'])}
    gpu = {'main_gpu': 1}
    last_progress = time.time()
    t0, n = time.time(), 0
    print('vision model:', model, flush=True)
    while True:
        idx_path = os.path.join(ROOT, 'triage_index.json')
        total = len(json.load(open(idx_path))) if os.path.exists(idx_path) else None
        have = [int(f[:-4]) for f in os.listdir(THUMBS) if f.endswith('.jpg')]
        tp = os.path.join(ROOT, 'triage_targets.json')
        if os.path.exists(tp):                      # only the frames plan_backlog.py chose, oldest day first
            tg = json.load(open(tp))
            have = sorted((i for i in have if str(i) in tg), key=lambda i: (tg[str(i)]['kind'] != 'backlog', tg[str(i)]['day'], i))
            total = len(tg)
        todo = [i for i in have if str(i) not in done or 'error' in done[str(i)] or i not in vec]
        if not todo:
            if total is not None and len(done) >= total:
                break
            if time.time() - last_progress > 900:
                print('no new thumbs for 15 min, stopping', flush=True)
                break
            time.sleep(30)
            continue
        for i in todo:
            p = os.path.join(THUMBS, f'{i}.jpg')
            try:
                if str(i) not in done or 'error' in done[str(i)]:
                    done[str(i)] = {'tags': vf.ask(model, vf.small(p), gpu)}
                if i not in vec:
                    vec[i] = emb.embed([p])[0]
            except Exception as e:
                done[str(i)] = {'error': f'{type(e).__name__}: {str(e)[:100]}'}
                if gpu and n < 5:
                    gpu = {}
            n += 1
            last_progress = time.time()
            if n % 50 == 0:
                json.dump(done, open(OUT, 'w'))
                ids = sorted(vec)
                np.savez(VEC, ids=np.array(ids), vectors=np.stack([vec[k] for k in ids]))
                el = time.time() - t0
                print(f'  {len(done)}/{total}  {el / n:.1f}s each', flush=True)
    json.dump(done, open(OUT, 'w'))
    ids = sorted(vec)
    np.savez(VEC, ids=np.array(ids), vectors=np.stack([vec[k] for k in ids]))
    print('done', len(done), flush=True)


if __name__ == '__main__':
    main()
