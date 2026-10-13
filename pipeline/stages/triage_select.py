"""Triage step 3: decide which unprocessed raws are worth a full-size render. Nothing is deleted or hidden; the rest simply stay raw.

Per frame (from triage_vision.json): release flags (visionlib rules, owner-hands rule), subject class (nature / not nature / unclear).
Similarity clusters (SigLIP cosine >= SIM to the running cluster mean, same day, gap <= GAP_S between neighbours). Within a cluster the
sharpest frames among the nature + unflagged ones are selected: 1 if the cluster has <= 3 frames, 2 if <= 10, else 3.
Output: triage_selection.json (selected frames) and triage_all.json (every frame with its category, cluster and score).
"""
import collections
import json
import os
import re
import sys
from datetime import datetime

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import visionlib

ROOT = os.environ.get('BIO_DATA', r'F:\bio-automation-data')
THUMBS = os.path.join(ROOT, 'triage_thumbs')
SIM, GAP_S = 0.90, 600
EST_MB = 20

NATURE = re.compile(r'flower|plant|leaf|leaves|fungus|fungi|mushroom|fruiting|insect|animal|bird|spider|butterfly|moth|bee|fly|beetle|tree|bark|'
                    r'wing|petal|bud|berry|seed|cap|gill|lichen|moss|reptile|frog|snake|turtle|mammal|caterpillar|larva|cocoon|egg|web|nest|fruit|'
                    r'stem|branch|waterfall|water|landscape|sky|cloud|rock', re.I)
NOT_NATURE = re.compile(r'vehicle|car|truck|motorcycle|building|house|sign|person|people|human|food|plate|dish|drink|bottle|room|furniture|'
                        r'screen|phone|text|poster|crowd|street|road|boat|machine|tool|toy', re.I)


def sharpness(path):
    try:
        g = np.asarray(Image.open(path).convert('L'), dtype=np.float32)
        lap = g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
        return float(lap.var())
    except Exception:
        return 0.0


def main():
    index = {i['id']: i for i in json.load(open(os.path.join(ROOT, 'triage_index.json')))}
    vis = json.load(open(os.path.join(ROOT, 'triage_vision.json'), encoding='utf-8'))
    z = np.load(os.path.join(ROOT, 'triage_vectors.npz'))
    vec = {int(i): v for i, v in zip(z['ids'], z['vectors'])}
    cap = {r['id']: r.get('captured') for r in json.load(open(os.path.join(ROOT, 'lc_raws.json')))}

    rows = []
    for pid, it in index.items():
        e = vis.get(str(pid))
        if not e or 'tags' not in e or pid not in vec:
            rows.append({'id': pid, 'day': it['day'], 'category': 'pending'})
            continue
        t = visionlib.flat(e['tags'])
        flags = visionlib.release_flags(e['tags'])
        s = str(t.get('subject_part') or '').lower()
        if flags:
            cat = 'flagged'
        elif NOT_NATURE.search(s) and not NATURE.search(s):
            cat = 'not_nature'
        elif NATURE.search(s):
            cat = 'nature'
        else:
            cat = 'unclear'
        ts = None
        try:
            ts = datetime.fromisoformat(cap.get(pid) or '').timestamp()
        except ValueError:
            pass
        rows.append({'id': pid, 'day': it['day'], 'stem': it['stem'], 'file': it['file'], 'category': cat, 'flags': flags, 'subject': s,
                     'habitat': str(t.get('habitat') or ''), 't': ts, 'sharp': sharpness(os.path.join(THUMBS, f'{pid}.jpg'))})

    byday = collections.defaultdict(list)
    for r in rows:
        if r['category'] != 'pending':
            byday[r['day']].append(r)
    cid = 0
    for day, rs in byday.items():
        rs.sort(key=lambda r: (r['t'] is None, r['t'] or 0, r['id']))
        cur, mean, last_t = [], None, None
        for r in rs:
            v = vec[r['id']]
            ok = cur and mean is not None and float(v @ (mean / (np.linalg.norm(mean) + 1e-9))) >= SIM and \
                (r['t'] is None or last_t is None or r['t'] - last_t <= GAP_S)
            if not ok:
                cid += 1
                cur, mean = [], None
            cur.append(r)
            mean = v.copy() if mean is None else mean + v
            r['cluster'] = cid
            last_t = r['t'] if r['t'] is not None else last_t
    size = collections.Counter(r.get('cluster') for r in rows if r.get('cluster') is not None)
    groups = collections.defaultdict(list)
    for r in rows:
        if r.get('cluster') is not None:
            r['cluster_size'] = size[r['cluster']]
            groups[r['cluster']].append(r)
    sel = []
    for c, rs in groups.items():
        cand = sorted([r for r in rs if r['category'] in ('nature', 'unclear')], key=lambda r: -r['sharp'])
        k = 1 if len(rs) <= 3 else (2 if len(rs) <= 10 else 3)
        for r in cand[:k]:
            r['selected'] = True
            sel.append(r)
    json.dump(rows, open(os.path.join(ROOT, 'triage_all.json'), 'w'))
    sel.sort(key=lambda r: (r['day'], r['t'] or 0))
    json.dump(sel, open(os.path.join(ROOT, 'triage_selection.json'), 'w'))
    cat = collections.Counter(r['category'] for r in rows)
    print('frames', len(rows), '| categories', dict(cat))
    print('clusters', len(groups), '| singletons', sum(1 for g in groups.values() if len(g) == 1), '| largest', max((len(g) for g in groups.values()), default=0))
    print(f'SELECTED for a full render: {len(sel)} (~{len(sel) * EST_MB / 1024:.0f} GB at ~{EST_MB} MB each)')
    yr = collections.Counter(r['day'][:7] for r in sel)
    print('selected by month:', dict(sorted(yr.items())))


if __name__ == '__main__':
    main()
