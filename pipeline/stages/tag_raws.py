"""Write title + keywords into the rendered raw JPEGs (F:\\Bio-export\\inat-raws): IPTC-IIM + XMP, lossless (pixel data untouched).

Inputs per frame (id is the suffix of the file name): triage_vision.json (descriptors, release flags), triage_vectors.npz (SigLIP).
Identification is ADVISORY and honest: nearest labelled photos from the owner's own iNaturalist library vote; a species name is written
only when the vote share is >= STRONG (measured on this index: 0.7-0.9 -> ~78%, >= 0.9 -> ~88% correct); otherwise only the group word
(fungus / plant / insect ...) at ~86% top-1 is used, and nothing the frame does not support. Keywords follow the Adobe order: ID, group,
subject, colours, habitat/substrate, viewpoint, light, adjectives, generic, season. <= 49, no commas, lowercase.
Idempotent: tag_raws_done.json maps frame id -> keyword hash; frames whose vision result is not there yet are left for the next run.
"""
import collections
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime

import numpy as np
from qdrant_client import QdrantClient

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import visionlib
from embed_meta import rewrite

ROOT = os.environ.get('BIO_DATA', r'F:\bio-automation-data')
OUT = os.environ.get('BIO_RENDER_OUT', r'F:\Bio-export\inat-raws')
STRONG, K = 0.7, 15
SUF = os.environ.get('BIO_TAG_SUFFIX', '')
DONE = os.path.join(ROOT, f'tag_raws_done{SUF}.json')
LOG = os.path.join(ROOT, f'tag_raws_log{SUF}.json')
ICONIC = {'Fungi': ['fungus'], 'Plantae': ['plant'], 'Insecta': ['insect', 'arthropod'], 'Arachnida': ['arachnid', 'arthropod', 'spider'],
          'Aves': ['bird'], 'Mammalia': ['mammal'], 'Reptilia': ['reptile'], 'Amphibia': ['amphibian'], 'Mollusca': ['mollusc'],
          'Animalia': ['animal'], 'Protozoa': ['slime mold'], 'Chromista': ['alga']}
LOW_VALUE = {'stem', 'leaf', 'stem and leaves', 'whole animal', 'whole insect', 'grass', ''}
NONVISUAL = {'natural', 'fresh', 'vibrant', 'lush', 'delicate', 'tender', 'small', 'bright', 'sparse', 'clustered', 'fuzzy', 'detailed',
             'sharp', 'soft', 'tactile', 'beautiful', 'colorful', 'colourful'}
SEASON = {12: 'winter', 1: 'winter', 2: 'winter', 3: 'spring', 4: 'spring', 5: 'spring', 6: 'summer', 7: 'summer', 8: 'summer',
          9: 'autumn', 10: 'autumn', 11: 'autumn'}


def lst(v):
    if v is None:
        return []
    if isinstance(v, str):
        v = re.split(r'[,;/]| and ', v.strip('[]'))
    return [str(x).strip().strip("'\"").lower() for x in v if str(x).strip().strip("'\"")]


def clean(k):
    return re.sub(r'\s+', ' ', k.replace(',', ' ')).strip().lower()


def main():
    idx = {i['id']: i for i in json.load(open(os.path.join(ROOT, 'triage_index.json')))}
    vis = json.load(open(os.path.join(ROOT, 'triage_vision.json'), encoding='utf-8'))
    z = np.load(os.path.join(ROOT, 'triage_vectors.npz'))
    vec = {int(i): v for i, v in zip(z['ids'], z['vectors'])}
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
    done = json.load(open(DONE)) if os.path.exists(DONE) else {}
    log = json.load(open(LOG)) if os.path.exists(LOG) else {}
    files = {}
    for root, _, fs in os.walk(OUT):
        for f in fs:
            m = re.search(r'_(\d+)\.jpg$', f)
            if m and f.lower().endswith('.jpg'):
                files[int(m.group(1))] = os.path.join(root, f)
    print(f'rendered files {len(files)} | vision results {len(vis)} | already tagged {len(done)}', flush=True)
    n = wait = fail = 0
    stats = collections.Counter()
    for fid, path in files.items():
        if str(fid) in done:
            continue
        if time.time() - os.path.getmtime(path) < 90:      # the renderer may still be writing it
            wait += 1
            continue
        e = vis.get(str(fid))
        if not e or 'tags' not in e or fid not in vec:
            wait += 1
            continue
        t = visionlib.flat(e['tags'])
        flags = visionlib.release_flags(e['tags'])
        sims = R @ vec[fid]
        top = np.argpartition(-sims, K)[:K]
        votes_sp, votes_ic, info = collections.Counter(), collections.Counter(), {}
        for j in top:
            p = pay[j]
            votes_ic[p.get('iconic') or '?'] += 1.0 / K
            if p.get('rank') == 'species':
                votes_sp[p['taxon']] += 1.0 / K
                info[p['taxon']] = p
        ic, ic_share = (votes_ic.most_common(1) or [('?', 0)])[0]
        sp, sp_share = (votes_sp.most_common(1) or [(None, 0)])[0]
        kws, why = [], 'group'
        if sp and sp_share >= STRONG:
            p = info[sp]
            kws += [p.get('common') or '', sp.lower()]
            why = f'species vote {sp_share:.2f}'
        kws += ICONIC.get(ic, []) if ic_share >= 0.4 else []
        subj = (lst(t.get('subject_part')) or [''])[0]
        if subj not in LOW_VALUE:
            kws.append(subj)
        kws += lst(t.get('colors'))[:3]
        kws += [x for x in lst(t.get('habitat'))[:1] + lst(t.get('substrate'))[:1] if x]
        view = (lst(t.get('viewpoint')) or [''])[0]
        if view:
            kws.append(view)
        if (lst(t.get('light')) or [''])[0] in ('backlit', 'low light', 'flash', 'overcast', 'shade'):
            kws.append(lst(t.get('light'))[0])
        kws += [a for a in lst(t.get('adjectives')) if a not in NONVISUAL][:3]
        kws += ['nature', 'outdoors']
        if not any(f in flags for f in ('person_visible', 'face_visible', 'identifiable_person')):
            kws.append('no people')
        try:
            kws.append(SEASON[datetime.strptime(idx[fid]['day'], '%Y-%m-%d').month])
        except (KeyError, ValueError):
            pass
        seen, final = set(), []
        for k in map(clean, kws):
            if k and len(k) <= 40 and k not in seen:
                seen.add(k); final.append(k)
        final = final[:49]
        name = (info[sp].get('common') if (sp and sp_share >= STRONG and info[sp].get('common')) else None)
        group_word = (ICONIC.get(ic) or [''])[0] if ic_share >= 0.4 else ''
        what = name or (f'{group_word} {subj}'.strip() if subj and subj not in LOW_VALUE else group_word) or 'nature subject'
        hab = (lst(t.get('habitat')) or [''])[0]
        prep = 'on' if re.search(r'floor|ground|bark|log|wood|rock|stone|wall|trunk|stump', hab) else 'in'
        hab = re.sub(r'^(a|the)\s+', '', hab)
        title = (f'{what.capitalize()} {prep} the {hab}' if hab else what.capitalize())[:69]
        try:
            data = open(path, 'rb').read()
            new = rewrite(data, title, final, title + '.')
            tmp = path + '.part'
            open(tmp, 'wb').write(new)
            os.replace(tmp, path)
            done[str(fid)] = hashlib.sha1('|'.join(final).encode()).hexdigest()[:12]
            log[str(fid)] = {'title': title, 'n_keywords': len(final), 'id_basis': why, 'group_vote': round(ic_share, 2),
                             'flags': flags, 'day': idx[fid]['day']}
            stats[why.split()[0]] += 1
            n += 1
        except Exception as ex:
            fail += 1
            print('fail', fid, type(ex).__name__, str(ex)[:80], flush=True)
        if n % 200 == 0 and n:
            json.dump(done, open(DONE, 'w')); json.dump(log, open(LOG, 'w'))
            print(f'  tagged {n}', flush=True)
    json.dump(done, open(DONE, 'w')); json.dump(log, open(LOG, 'w'))
    print(f'tagged now {n} | waiting for vision {wait} | failed {fail} | basis {dict(stats)} | total tagged {len(done)}', flush=True)


if __name__ == '__main__':
    main()
