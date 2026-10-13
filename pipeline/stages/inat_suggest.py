"""Advisory taxon suggestions for the proposed observations in inat_queue.json (never posted anywhere, never pre-selected).

Embeds each upload copy with the local SigLIP ONNX model and votes over the owner's own labelled iNaturalist photos (bio_photos,
source=inat_photo). Votes are pooled per proposed observation (group). Each candidate carries the accuracy MEASURED by leave-one-out
on this index (embed_loo.py: same-observation and same-day neighbours excluded) for its vote-share bin, so the console can show an
honest number instead of a raw score.
Output: inat_suggestions.json {group: {"photos": n, "species": [...], "iconic": [...]}}
"""
import collections
import json
import os
import sys

import numpy as np
from qdrant_client import QdrantClient

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embed_index import Embedder

ROOT = os.environ.get('BIO_DATA', r'F:\bio-automation-data')
K = 15
TAG = os.environ.get('BIO_QUEUE_TAG', '')
# measured by embed_loo.py on 2026-10-13 (species level / iconic level, vote share bins)
BINS_SPECIES = [(0.0, 0.3, 0.154), (0.3, 0.5, 0.273), (0.5, 0.7, 0.486), (0.7, 0.9, 0.782), (0.9, 1.01, 0.877)]


def measured(share):
    for lo, hi, acc in BINS_SPECIES:
        if lo <= share < hi:
            return acc
    return None


def main():
    q = json.load(open(os.path.join(ROOT, f'inat_queue{TAG}.json')))
    obs = json.load(open(os.path.join(ROOT, 'inat_observations.json'), encoding='utf-8'))['observations']
    tid = {}
    for o in obs:
        t = o.get('taxon') or {}
        if t.get('name') and t.get('id'):
            tid[t['name']] = t['id']
    c = QdrantClient(path=os.path.join(ROOT, 'qdrant_local'))
    vecs, pay, off = [], [], None
    while True:
        pts, off = c.scroll('bio_photos', limit=2000, with_payload=True, with_vectors=True, offset=off)
        for p in pts:
            if p.payload.get('source') == 'inat_photo' and p.payload.get('taxon'):
                vecs.append(p.vector); pay.append(p.payload)
        if off is None:
            break
    X = np.asarray(vecs, dtype=np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    print('reference photos', len(X), flush=True)

    items = [i for i in q if i.get('copy') and os.path.exists(i['copy'])]
    emb = Embedder()
    by_group = collections.defaultdict(list)
    B = 16
    for s in range(0, len(items), B):
        chunk = items[s:s + B]
        V = emb.embed([i['copy'] for i in chunk])
        S = V @ X.T
        for it, row in zip(chunk, S):
            top = np.argpartition(-row, K)[:K]
            by_group[it['group']].append([pay[j] for j in top])
        if (s // B) % 10 == 0:
            print(f'  embedded {min(s + B, len(items))}/{len(items)}', flush=True)

    out = {}
    for g, lists in by_group.items():
        sp, ic, info = collections.Counter(), collections.Counter(), {}
        for neigh in lists:
            n = len(neigh)
            for p in neigh:
                if p.get('rank') == 'species':
                    sp[p['taxon']] += 1.0 / n
                    info[p['taxon']] = p
                ic[p.get('iconic') or '?'] += 1.0 / n
        m = len(lists)
        species = []
        for name, v in sp.most_common(3):
            share = v / m
            p = info[name]
            species.append({'taxon': name, 'taxon_id': tid.get(name), 'common': p.get('common'), 'iconic': p.get('iconic'),
                            'vote_share': round(share, 3), 'measured_accuracy': measured(share)})
        iconic = [{'iconic': k, 'vote_share': round(v / m, 3)} for k, v in ic.most_common(2)]
        out[str(g)] = {'photos': m, 'species': species, 'iconic': iconic}
    json.dump(out, open(os.path.join(ROOT, f'inat_suggestions{TAG}.json'), 'w'))
    strong = sum(1 for v in out.values() if v['species'] and v['species'][0]['vote_share'] >= 0.7)
    print(f'groups with suggestions {len(out)} | species vote share >= 0.7 (measured ~78-88%): {strong}', flush=True)


if __name__ == '__main__':
    main()
