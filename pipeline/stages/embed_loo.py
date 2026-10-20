"""Leave-one-out accuracy of the SigLIP embedding index as an ID starting point (honest version).

Queries are the owner's labelled iNat photos (bio_photos, source=inat_photo). A query's neighbours exclude every photo of the SAME
observation (burst siblings would otherwise leak the label) and the same observed_on day. Reports top-1/top-5 at species, genus and
iconic group, a kNN vote-share calibration table, and a majority-class baseline. Species-rank labels only for species/genus scoring.
"""
import collections
import sys

import numpy as np
from qdrant_client import QdrantClient

c = QdrantClient(path=r'F:\bio-automation-data\qdrant_local')
ids, vecs, pay = [], [], []
off = None
while True:
    pts, off = c.scroll('bio_photos', limit=2000, with_payload=True, with_vectors=True, offset=off)
    for p in pts:
        if p.payload.get('source') == 'inat_photo' and p.payload.get('taxon'):
            ids.append(p.id); vecs.append(p.vector); pay.append(p.payload)
    if off is None:
        break
X = np.asarray(vecs, dtype=np.float32)
X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
n = len(X)
print('labelled iNat photo vectors:', n)
obs = np.array([p['obs_id'] for p in pay])
day = np.array([p.get('observed_on') or '' for p in pay])
sp = np.array([p['taxon'] if p.get('rank') == 'species' else '' for p in pay])
gen = np.array([p['taxon'].split()[0] if p.get('rank') in ('species', 'genus') else '' for p in pay])
ico = np.array([p.get('iconic') or '' for p in pay])
print('species-rank:', int((sp != '').sum()), '| distinct species:', len(set(sp) - {''}), '| distinct genera:', len(set(gen) - {''}),
      '| iconic groups:', dict(collections.Counter(ico).most_common(6)))

S = X @ X.T
same = (obs[:, None] == obs[None, :]) | ((day[:, None] == day[None, :]) & (day[:, None] != ''))
S[same] = -2
K = 15
top = np.argpartition(-S, K, axis=1)[:, :K]
order = np.take_along_axis(S, top, axis=1).argsort(axis=1)[:, ::-1]
top = np.take_along_axis(top, order, axis=1)


def evaluate(label, labels, only_nonempty=True):
    ok1 = ok5 = tot = 0
    conf, hit = [], []
    maj = collections.Counter(l for l in labels if l).most_common(1)[0]
    for i in range(n):
        if only_nonempty and not labels[i]:
            continue
        neigh = [labels[j] for j in top[i] if labels[j]][:K]
        if not neigh:
            continue
        tot += 1
        vote = collections.Counter(neigh)
        best, cnt = vote.most_common(1)[0]
        ok1 += best == labels[i]
        ok5 += labels[i] in [labels[j] for j in top[i][:5]]
        conf.append(cnt / len(neigh)); hit.append(best == labels[i])
    print(f'{label:8s} n={tot:5d}  top-1 {ok1 / tot:5.1%}  top-5(any of 5 nearest) {ok5 / tot:5.1%}  majority-class baseline {maj[1] / sum(1 for l in labels if l):5.1%} ({maj[0]})')
    return np.array(conf), np.array(hit)


evaluate('species', sp)
evaluate('genus', gen)
cf, ht = evaluate('iconic', ico)
cf, ht = evaluate('species', sp)
print('species kNN vote-share vs accuracy:')
for lo, hi in ((0, .3), (.3, .5), (.5, .7), (.7, .9), (.9, 1.01)):
    m = (cf >= lo) & (cf < hi)
    if m.sum():
        print(f'   vote share {lo:.1f}-{min(hi, 1):.1f}: n={int(m.sum()):4d}  correct {ht[m].mean():5.1%}')
# species with >=5 photos only: what a classifier would actually have examples for
cnt = collections.Counter(s for s in sp if s)
rich = np.array([bool(s) and cnt[s] >= 5 for s in sp])
ok = tot = 0
for i in np.where(rich)[0]:
    neigh = [sp[j] for j in top[i] if sp[j]]
    if neigh:
        tot += 1
        ok += collections.Counter(neigh).most_common(1)[0][0] == sp[i]
print(f'species with >=5 photos: n={tot}  top-1 {ok / max(tot, 1):5.1%}  (species covered: {sum(1 for v in cnt.values() if v >= 5)})')
