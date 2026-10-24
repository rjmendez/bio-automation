"""Leave-one-out accuracy of an embedding index as an ID starting point (honest version), and the vote-share calibration bins inat_suggest.py shows.

Queries are the owner's labelled iNat photos (source=inat_photo). A query's neighbours exclude every photo of the SAME
observation and the same observed_on day (burst siblings would otherwise leak the label). Reports top-1 at species, genus and
iconic group, a kNN vote-share calibration table, and a majority-class baseline. Species-rank labels only for species/genus scoring.

usage: embed_loo.py [--backend siglip|bioclip2]   (default bioclip2; the collection comes from embed_config.json)
Writes the species vote-share bins to <BIO_DATA>/suggest_bins.json under the backend's name, which inat_suggest.py reads.
"""
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

K = 15
EDGES = ((0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01))


def bins_from_votes(conf, hit, edges=EDGES):
    """[{lo, hi, n, accuracy}] per vote-share bin: accuracy = share of correct top votes whose confidence is in [lo, hi); None when the bin is empty."""
    conf, hit = np.asarray(conf), np.asarray(hit, dtype=bool)
    out = []
    for lo, hi in edges:
        m = (conf >= lo) & (conf < hi)
        out.append({'lo': lo, 'hi': hi, 'n': int(m.sum()), 'accuracy': round(float(hit[m].mean()), 3) if m.any() else None})
    return out


def leave_one_out(X, obs, day, labels, k=K):
    """Top-1 kNN vote per query over its k nearest neighbours, never using the same observation or the same non-empty day.
    Returns (confidence, correct) arrays over the queries whose label is non-empty and that have at least one labelled neighbour."""
    X = np.asarray(X, dtype=np.float32)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    n = len(X)
    S = X @ X.T
    same = (obs[:, None] == obs[None, :]) | ((day[:, None] == day[None, :]) & (day[:, None] != ''))
    S[same] = -2
    kk = min(k, n - 1)
    top = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
    top = np.take_along_axis(top, np.take_along_axis(S, top, axis=1).argsort(axis=1)[:, ::-1], axis=1)
    conf, hit = [], []
    for i in range(n):
        if not labels[i]:
            continue
        neigh = [labels[j] for j in top[i] if labels[j] and S[i, j] > -2]
        if not neigh:
            continue
        best, cnt = collections.Counter(neigh).most_common(1)[0]
        conf.append(cnt / len(neigh))
        hit.append(best == labels[i])
    return np.array(conf), np.array(hit, dtype=bool)


def write_bins(path, backend, collection, bins, n):
    """Merge this backend's bins into the JSON file (other backends' entries are kept)."""
    data = json.load(open(path, encoding='utf-8')) if os.path.exists(path) else {}
    data[backend] = {'collection': collection, 'k': K, 'queries': n, 'species_bins': bins}
    tmp = path + '.tmp'
    json.dump(data, open(tmp, 'w', encoding='utf-8'), indent=1)
    os.replace(tmp, path)


def main():
    from qdrant_client import QdrantClient
    from embed_index import CFG
    backend = sys.argv[sys.argv.index('--backend') + 1] if '--backend' in sys.argv else 'bioclip2'
    collection = CFG['bioclip']['collection'] if backend == 'bioclip2' else CFG['collection']
    c = QdrantClient(path=CFG['qdrant_path'])
    vecs, pay, off = [], [], None
    while True:
        pts, off = c.scroll(collection, limit=2000, with_payload=True, with_vectors=True, offset=off)
        for p in pts:
            if p.payload.get('source') == 'inat_photo' and p.payload.get('taxon'):
                vecs.append(p.vector); pay.append(p.payload)
        if off is None:
            break
    print(f'{backend} ({collection}): labelled iNat photo vectors: {len(vecs)}')
    obs = np.array([p['obs_id'] for p in pay])
    day = np.array([p.get('observed_on') or '' for p in pay])
    sp = np.array([p['taxon'] if p.get('rank') == 'species' else '' for p in pay])
    gen = np.array([p['taxon'].split()[0] if p.get('rank') in ('species', 'genus') else '' for p in pay])
    ico = np.array([p.get('iconic') or '' for p in pay])
    bins, n_species = None, 0
    for label, labels in (('species', sp), ('genus', gen), ('iconic', ico)):
        cf, ht = leave_one_out(vecs, obs, day, labels)
        maj = collections.Counter(l for l in labels if l).most_common(1)[0][1] / sum(1 for l in labels if l)
        print(f'{label:8s} n={len(cf):5d}  top-1 {ht.mean():5.1%}  majority-class baseline {maj:5.1%}')
        if label == 'species':
            bins, n_species = bins_from_votes(cf, ht), len(cf)
            for b in bins:
                print(f'   vote share {b["lo"]:.1f}-{min(b["hi"], 1):.1f}: n={b["n"]:4d}  correct {b["accuracy"]}')
    root = os.environ.get('BIO_DATA', r'F:\bio-automation-data')
    write_bins(os.path.join(root, 'suggest_bins.json'), backend, collection, bins, n_species)


if __name__ == '__main__':
    main()
