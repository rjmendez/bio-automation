"""Advisory taxon suggestions for the proposed observations in inat_queue.json (never posted anywhere, never pre-selected).

Embeds each upload copy with the configured backend (default BioCLIP 2, GPU only; BIO_SUGGEST_BACKEND=siglip for the old ONNX index) and votes
over the owner's own labelled iNaturalist photos in that backend's Qdrant collection. Votes are pooled per proposed observation (group).
Each candidate carries the accuracy MEASURED by leave-one-out on that index (embed_loo.py: same-observation and same-day neighbours
excluded) for its vote-share bin, read from <BIO_DATA>/suggest_bins.json, so the console shows an honest number instead of a raw score.
Output: inat_suggestions.json {group: {"photos": n, "species": [...], "iconic": [...]}} plus inat_suggestions<TAG>.meta.json (backend, collection,
reference photos). Nothing is written when no upload copy exists on disk (an empty result must never replace the previous suggestions).
"""
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from taxon_zeroshot import iconic_of, measured as zs_measured, pool_group  # noqa: E402

ROOT =os.environ.get('BIO_DATA', r'F:\bio-automation-data')
K = 15
TAG = os.environ.get('BIO_QUEUE_TAG', '')
BACKEND = os.environ.get('BIO_SUGGEST_BACKEND', 'bioclip2')


def load_bins(path, backend):
    """[(lo, hi, accuracy)] measured for *backend*. SystemExit when embed_loo.py has not been run for it: no number is better than an invented one."""
    if not os.path.exists(path):
        raise SystemExit(f'inat_suggest: {path} does not exist; run embed_loo.py --backend {backend} first')
    entry = json.load(open(path, encoding='utf-8')).get(backend)
    if not entry:
        raise SystemExit(f'inat_suggest: {path} has no bins for backend {backend!r}; run embed_loo.py --backend {backend} first')
    return [(b['lo'], b['hi'], b['accuracy']) for b in entry['species_bins']]


def measured(share, bins):
    for lo, hi, acc in bins:
        if lo <= share < hi:
            return acc
    return None


def suggest_groups(by_group, tid, bins, zs_by_group=None, zs_bins=None):
    """by_group: {group: [neighbour payload list per photo]} -> {str(group): {"photos", "species" (top 3), "iconic" (top 2)}}.
    With zs_by_group ({group: [per-photo [(species row, prob)]]}) and zs_bins, each group also gets "zero_shot": the top 3 species of the WHOLE taxonomy
    by pooled probability, each with the accuracy measured per observation for its probability bin ([] for a group without predictions)."""
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
                            'vote_share': round(share, 3), 'measured_accuracy': measured(share, bins)})
        iconic = [{'iconic': k, 'vote_share': round(v / m, 3)} for k, v in ic.most_common(2)]
        out[str(g)] = {'photos': m, 'species': species, 'iconic': iconic}
        if zs_by_group is not None:
            pooled = pool_group(zs_by_group[g]) if zs_by_group.get(g) else []
            out[str(g)]['zero_shot'] = [{'taxon': row[1], 'taxon_id': row[0], 'iconic': iconic_of(row), 'family': row[6], 'prob': round(p, 3),
                                         'measured_accuracy': zs_measured(p, zs_bins)} for row, p in pooled]
    return out


def load_zeroshot(emb, cfg, torch_mod=None):
    """(ZeroShot, logit scale, observation-level bins) when embed_config.json's bioclip.zeroshot_index is set and the backend is BioCLIP 2; else None.
    SystemExit when it is set but taxon_zeroshot.py calibrate has not been run: a probability without its measured accuracy is not shown."""
    index = cfg.get('zeroshot_index')
    if not index or BACKEND != 'bioclip2':
        return None
    path = os.path.join(ROOT, 'zeroshot_bins.json')
    if not os.path.exists(path):
        raise SystemExit(f'inat_suggest: {path} does not exist; run taxon_zeroshot.py calibrate first')
    if torch_mod is None:
        import torch as torch_mod
    from taxon_zeroshot import ZeroShot
    return ZeroShot(index, torch_mod, emb.device), emb.logit_scale(), json.load(open(path, encoding='utf-8'))['bins']


def main():
    from qdrant_client import QdrantClient
    from embed_index import CFG, backend_setup
    bins = load_bins(os.path.join(ROOT, 'suggest_bins.json'), BACKEND)
    q = json.load(open(os.path.join(ROOT, f'inat_queue{TAG}.json')))
    items = [i for i in q if i.get('copy') and os.path.exists(i['copy'])]
    if not items:
        raise SystemExit(f'inat_suggest: none of the {len(q)} queue entries has an upload copy on disk; nothing written '
                         f'(inat_suggestions{TAG}.json is untouched)')
    obs = json.load(open(os.path.join(ROOT, 'inat_observations.json'), encoding='utf-8'))['observations']
    tid = {}
    for o in obs:
        t = o.get('taxon') or {}
        if t.get('name') and t.get('id'):
            tid[t['name']] = t['id']
    emb, collection, _, _ = backend_setup(BACKEND)
    c = QdrantClient(path=CFG['qdrant_path'])
    vecs, pay, off = [], [], None
    while True:
        pts, off = c.scroll(collection, limit=2000, with_payload=True, with_vectors=True, offset=off)
        for p in pts:
            if p.payload.get('source') == 'inat_photo' and p.payload.get('taxon'):
                vecs.append(p.vector); pay.append(p.payload)
        if off is None:
            break
    X = np.asarray(vecs, dtype=np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    print(f'backend {BACKEND}: reference photos {len(X)} in {collection}; {len(items)} of {len(q)} queue entries have a copy', flush=True)

    zsl = load_zeroshot(emb, CFG['bioclip'])
    by_group, zs_by_group = collections.defaultdict(list), collections.defaultdict(list)
    B = 16
    for s in range(0, len(items), B):
        chunk = items[s:s + B]
        V = emb.embed([i['copy'] for i in chunk])
        if zsl:
            for it, preds in zip(chunk, zsl[0].predict(V, zsl[1])):
                zs_by_group[it['group']].append(preds)
        S = V @ X.T
        for it, row in zip(chunk, S):
            k = min(K, len(row))
            top = np.argpartition(-row, k - 1)[:k]
            by_group[it['group']].append([pay[j] for j in top])
        if (s // B) % 10 == 0:
            print(f'  embedded {min(s + B, len(items))}/{len(items)}', flush=True)

    out = suggest_groups(by_group, tid, bins, zs_by_group if zsl else None, zsl[2] if zsl else None)
    json.dump(out, open(os.path.join(ROOT, f'inat_suggestions{TAG}.json'), 'w'))
    json.dump({'backend': BACKEND, 'collection': collection, 'reference_photos': int(len(X)), 'queue_entries': len(q),
               'entries_with_copy': len(items), 'groups': len(out), 'zero_shot_index': CFG['bioclip'].get('zeroshot_index') if zsl else None},
              open(os.path.join(ROOT, f'inat_suggestions{TAG}.meta.json'), 'w'))
    strong = sum(1 for v in out.values() if v['species'] and v['species'][0]['vote_share'] >= 0.7)
    print(f'groups with suggestions {len(out)} | species vote share >= 0.7: {strong}', flush=True)


if __name__ == '__main__':
    main()
