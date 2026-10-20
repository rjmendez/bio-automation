"""Put the rendered backlog frames (F:\\Bio-export\\inat-raws) into the LightCraft queue of the review console (BIO_QUEUE_TAG=_lc).

Everything is built from data we already have, nothing is recomputed on the GPU:
  * frame facts: lc_raws.json (capture time WITH utc offset, camera), triage_index.json (day, stem), triage_targets.json (backlog / gap)
  * location: phone track (stability gated, real utc offset) via lc_inat.gpx_position, else none (the owner enters it in the console)
  * release/people flags + subject: triage_vision.json  -> vision_inat_lc.json (same shape the console reads)
  * taxon suggestions: nearest labelled iNat photos (triage_vectors.npz vs the Qdrant index), pooled per proposed observation
Writes inat_queue_lc.json (the previous 2-photo test queue is kept as inat_queue_lc.json.bak-test), inat_suggestions_lc.json,
vision_inat_lc.json. The console reads these at start-up: restart task Inat-Console-LC afterwards.
"""
import collections
import json
import os
import re
import shutil
import sys
from datetime import datetime

import numpy as np
from qdrant_client import QdrantClient

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lc_inat
from inat_suggest import BINS_SPECIES, measured

ROOT = r'F:\bio-automation-data'
OUT = r'F:\Bio-export\inat-raws'
K, BURST_S = 15, 120


def main():
    idx = {i['id']: i for i in json.load(open(os.path.join(ROOT, 'triage_index.json')))}
    tg = json.load(open(os.path.join(ROOT, 'triage_targets.json')))
    raws = {r['id']: r for r in json.load(open(os.path.join(ROOT, 'lc_raws.json')))}
    vis = json.load(open(os.path.join(ROOT, 'triage_vision.json'), encoding='utf-8'))
    z = np.load(os.path.join(ROOT, 'triage_vectors.npz'))
    vec = {int(i): v for i, v in zip(z['ids'], z['vectors'])}
    track = lc_inat.load_track()

    files = {}
    for root, _, fs in os.walk(OUT):
        for f in fs:
            m = re.search(r'_(\d+)\.jpg$', f)
            if m:
                files[int(m.group(1))] = os.path.join(root, f)
    items = []
    for fid, path in files.items():
        if str(fid) not in tg or fid not in idx:
            continue
        r = raws.get(fid, {})
        cap = r.get('captured') or ''
        try:
            dt = datetime.fromisoformat(cap)
            epoch, off, tstr = dt.timestamp(), (dt.utcoffset().total_seconds() if dt.utcoffset() is not None else None), dt.strftime('%Y:%m:%d %H:%M:%S')
        except ValueError:
            epoch, off, tstr = None, None, None
        it = {'src': 'LC', 'lc_id': fid, 'rel': os.path.basename(path), 'day': idx[fid]['day'], 'raw': idx[fid]['file'], 'w': r.get('w'),
              'h': r.get('h'), 'time': tstr, 't': epoch, 'tz_offset': off, 'camera': r.get('camera'), 'copy': path, 'copy_error': None,
              'release': 'unscreened', 'kind': tg[str(fid)]['kind'], 'keywords': [], 'title': ''}
        g = lc_inat.gpx_position(track, epoch) if epoch is not None else None
        if g:
            it.update(lat=g[0], lon=g[1], loc_source='gpx_myasshurts', loc_note=f'phone track, sigma {g[2]:.1f} m, spread {g[3]:.0f} m',
                      positional_accuracy=int(max(g[2], g[3], 5)))
        else:
            it.update(loc_source='none', loc_note='no GPS and no stable phone track; enter the place in the console')
        items.append(it)
    items.sort(key=lambda x: (x['kind'] != 'backlog', x['day'], x['camera'] or '', x['t'] if x['t'] is not None else -1))
    gid, last = 0, None
    for it in items:
        key = (it['day'], it['camera'])
        same = bool(last and last['key'] == key and it['t'] is not None and last['t'] is not None and it['t'] - last['t'] <= BURST_S)
        if not same:
            gid += 1
        it['group'] = gid
        last = {'key': key, 't': it['t']}

    # vision file in the shape the console reads: {copy path: {'tags': ...}}
    vision = {it['copy']: {'tags': (vis.get(str(it['lc_id'])) or {}).get('tags') or {}} for it in items}

    # suggestions
    obs = json.load(open(os.path.join(ROOT, 'inat_observations.json'), encoding='utf-8'))['observations']
    tid = {(o['taxon'] or {}).get('name'): o['taxon']['id'] for o in obs if o.get('taxon') and o['taxon'].get('id')}
    c = QdrantClient(path=os.path.join(ROOT, 'qdrant_local'))
    refs, pay, off = [], [], None
    while True:
        pts, off = c.scroll('bio_photos', limit=2000, with_payload=True, with_vectors=True, offset=off)
        for p in pts:
            if p.payload.get('source') == 'inat_photo' and p.payload.get('taxon'):
                refs.append(p.vector); pay.append(p.payload)
        if off is None:
            break
    R = np.asarray(refs, dtype=np.float32)
    R /= np.linalg.norm(R, axis=1, keepdims=True) + 1e-9
    by_group = collections.defaultdict(list)
    for it in items:
        v = vec.get(it['lc_id'])
        if v is None:
            continue
        top = np.argpartition(-(R @ v), K)[:K]
        by_group[it['group']].append([pay[j] for j in top])
    sugg = {}
    for g, lists in by_group.items():
        sp, ic, info = collections.Counter(), collections.Counter(), {}
        for neigh in lists:
            for p in neigh:
                if p.get('rank') == 'species':
                    sp[p['taxon']] += 1.0 / len(neigh)
                    info[p['taxon']] = p
                ic[p.get('iconic') or '?'] += 1.0 / len(neigh)
        m = len(lists)
        species = [{'taxon': n, 'taxon_id': tid.get(n), 'common': info[n].get('common'), 'iconic': info[n].get('iconic'),
                    'vote_share': round(v / m, 3), 'measured_accuracy': measured(v / m)} for n, v in sp.most_common(3)]
        sugg[str(g)] = {'photos': m, 'species': species, 'iconic': [{'iconic': k, 'vote_share': round(v / m, 3)} for k, v in ic.most_common(2)]}

    q = os.path.join(ROOT, 'inat_queue_lc.json')
    if os.path.exists(q) and not os.path.exists(q + '.bak-test'):
        shutil.copy(q, q + '.bak-test')
    json.dump(items, open(q, 'w'))
    json.dump(sugg, open(os.path.join(ROOT, 'inat_suggestions_lc.json'), 'w'))
    json.dump(vision, open(os.path.join(ROOT, 'vision_inat_lc.json'), 'w'))
    loc = collections.Counter(i['loc_source'] for i in items)
    strong = sum(1 for s in sugg.values() if s['species'] and s['species'][0]['vote_share'] >= 0.7)
    print(f'queue: {len(items)} photos in {gid} proposed observations | location {dict(loc)} | kind {dict(collections.Counter(i["kind"] for i in items))}')
    print(f'suggestions for {len(sugg)} groups, {strong} with a species vote >= 0.7')


if __name__ == '__main__':
    main()
