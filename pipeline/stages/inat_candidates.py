"""Find prepared JPEG exports (local library + NAS) that are NOT on iNaturalist yet.

For every export: capture time, camera and EXIF GPS (header read only), a 32x32 gray feature. It is compared with every iNat photo
observed on the same day (+-1 day for timezone edges); correlation >= 0.95 with the same aspect ratio means "already uploaded".
Results: inat_candidates.json. Read-only against the library and the NAS. Resumable (feature cache).
"""
import collections
import concurrent.futures as cf
import csv
import json
import os
import pickle
import re
import sys
import urllib.request
from datetime import datetime, timedelta

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = os.environ.get('BIO_DATA', r'F:\bio-automation-data')
SRC = os.environ.get('BIO_SRC', r'F:\Bio')
NAS = os.environ.get('BIO_NAS', r'Y:\rjmendez-admin\DESKTOP-BVRDK4J\Data\F\Bio')
BS = chr(92)
TH = 0.95
PREPARED = ('inat-lr', 'inat', 'export-lr', 'export')
CACHE_INAT = os.path.join(ROOT, 'cache', 'inat')
FEAT_NAS = os.path.join(ROOT, 'nas_features.pkl')
FEAT_LOCAL = os.path.join(ROOT, 'local_features.pkl')
OUT = os.path.join(ROOT, 'inat_candidates.json')


def feat(im):
    g = np.asarray(im.convert('L').resize((32, 32), Image.LANCZOS), dtype=np.float32)
    g = (g - g.mean()) / (g.std() + 1e-6)
    return g.ravel().astype(np.float16)


def rational(x):
    try:
        return float(x)
    except Exception:
        return x[0] / x[1]


def read_one(args):
    root, rel = args
    p = os.path.join(root, rel.replace('/', os.sep))
    try:
        im = Image.open(p)
        w, h = im.size
        ex = im.getexif()
        sub = ex.get_ifd(0x8769)
        t = sub.get(0x9003) or ex.get(0x0132)
        gps = ex.get_ifd(0x8825)
        lat = lon = alt = None
        if gps and 2 in gps and 4 in gps:
            d, m_, s = (rational(v) for v in gps[2])
            lat = (d + m_ / 60 + s / 3600) * (-1 if gps.get(1) == 'S' else 1)
            d, m_, s = (rational(v) for v in gps[4])
            lon = (d + m_ / 60 + s / 3600) * (-1 if gps.get(3) == 'W' else 1)
            if 6 in gps:
                alt = rational(gps[6])
        im.draft('RGB', (128, 128))
        return rel, {'w': w, 'h': h, 'time': t, 'camera': ex.get(0x0110), 'lat': lat, 'lon': lon, 'alt': alt, 'f': feat(im), 'iso': sub.get(0x8827)}
    except Exception as e:
        return rel, {'err': type(e).__name__}


def inat_feature(photo):
    path = os.path.join(CACHE_INAT, f"{photo['id']}.jpg")
    if not os.path.exists(path):
        try:
            req = urllib.request.Request(photo['url'].replace('square', 'small'), headers={'User-Agent': 'bio-automation/0.1 (personal)'})
            open(path, 'wb').write(urllib.request.urlopen(req, timeout=30).read())
        except Exception:
            return photo['id'], None
    try:
        return photo['id'], feat(Image.open(path))
    except Exception:
        return photo['id'], None


def inventory(name):
    return [r['path'].replace(BS, '/') for r in csv.DictReader(open(os.path.join(ROOT, name), encoding='utf-8-sig'))]


def main():
    items = []                      # (root_tag, root_path, rel)
    raws = {}
    for tag, root, inv in (('F', SRC, 'f_bio_inventory.csv'), ('NAS', NAS, 'nas_inventory.csv')):
        for p in inventory(inv):
            parts = p.split('/')
            if len(parts) < 3 or not parts[0].isdigit() or len(parts[1]) != 10:
                continue
            stem, ext = os.path.splitext(parts[-1])
            if ext.lower() in ('.nef', '.dng'):
                raws[(tag, parts[1], stem.lower())] = p
            elif ext.lower() == '.jpg' and len(parts) >= 4 and parts[2].lower() in PREPARED:
                items.append((tag, root, p))
    print('prepared exports:', collections.Counter(t for t, _, _ in items), flush=True)

    caches = {}
    for tag, path in (('F', FEAT_LOCAL), ('NAS', FEAT_NAS)):
        caches[tag] = pickle.load(open(path, 'rb')) if os.path.exists(path) else {}
    info = {}
    todo = []
    for tag, root, rel in items:
        c = caches[tag].get(rel)
        # the NAS matcher cache holds (w, h, feature) tuples; header fields (time, GPS, camera) are still needed, so re-read headers
        todo.append((tag, root, rel))
    print('reading headers + features:', len(todo), flush=True)
    done = 0
    with cf.ThreadPoolExecutor(8) as ex:
        for (tag, _, rel), (_, d) in zip(todo, ex.map(read_one, [(r, p) for _, r, p in todo])):
            info[(tag, rel)] = d
            done += 1
            if done % 1000 == 0:
                print(f'  {done}/{len(todo)}', flush=True)
    pickle.dump({k: v for k, v in info.items()}, open(os.path.join(ROOT, 'candidate_headers.pkl'), 'wb'))

    inat = json.load(open(os.path.join(ROOT, 'inat_observations.json'), encoding='utf-8'))['observations']
    by_day = collections.defaultdict(list)
    for o in inat:
        if o.get('observed_on'):
            by_day[o['observed_on']].append(o)
    need_days = set()
    for tag, root, rel in items:
        d = rel.split('/')[1]
        dd = datetime.strptime(d, '%Y-%m-%d')
        for k in (-1, 0, 1):
            need_days.add((dd + timedelta(days=k)).strftime('%Y-%m-%d'))
    photos = [(ph, o) for dday in need_days for o in by_day.get(dday, []) for ph in o['photos']]
    print('iNat photos on candidate days:', len(photos), flush=True)
    ifeat = {}
    with cf.ThreadPoolExecutor(8) as ex:
        for i, (pid, f) in enumerate(ex.map(inat_feature, [p for p, _ in photos]), 1):
            if f is not None:
                ifeat[pid] = f
            if i % 2000 == 0:
                print(f'  inat features {i}/{len(photos)}', flush=True)
    obs_of = {ph['id']: (o, ph) for ph, o in photos}
    day_photos = collections.defaultdict(list)
    for pid, (o, ph) in obs_of.items():
        if pid in ifeat:
            day_photos[o['observed_on']].append(pid)

    out = []
    for tag, root, rel in items:
        d = info.get((tag, rel)) or {}
        if 'f' not in d:
            continue
        day = rel.split('/')[1]
        dd = datetime.strptime(day, '%Y-%m-%d')
        pool = [p for k in (-1, 0, 1) for p in day_photos.get((dd + timedelta(days=k)).strftime('%Y-%m-%d'), [])]
        best, best_obs = -1.0, None
        if pool:
            M = np.stack([ifeat[p].astype(np.float32) for p in pool])
            corr = (M @ d['f'].astype(np.float32)) / 1024.0
            ar = max(d['w'], d['h']) / max(min(d['w'], d['h']), 1)
            for j, p in enumerate(pool):
                dm = obs_of[p][1].get('dims') or {}
                if dm.get('width') and dm.get('height'):
                    if abs(max(dm['width'], dm['height']) / min(dm['width'], dm['height']) - ar) > 0.03:
                        corr[j] = -1
            j = int(np.argmax(corr))
            best, best_obs = float(corr[j]), obs_of[pool[j]][0]['id']
        stem = os.path.splitext(os.path.basename(rel))[0].lower()
        raw = raws.get((tag, day, stem)) or raws.get((tag, day, stem.split(' (')[0]))
        t = d.get('time')
        out.append({'src': tag, 'rel': rel, 'day': day, 'raw': raw, 'w': d['w'], 'h': d['h'], 'time': t, 'camera': d.get('camera'), 'iso': d.get('iso'),
                    'lat': d.get('lat'), 'lon': d.get('lon'), 'alt': d.get('alt'),
                    'best_corr': round(best, 4), 'best_obs': best_obs, 'already_on_inat': best >= TH})
    json.dump(out, open(OUT, 'w'))
    new = [x for x in out if not x['already_on_inat']]
    print(f'exports checked {len(out)} | already on iNat {len(out) - len(new)} | NOT on iNat {len(new)} '
          f'| of those with EXIF GPS {sum(1 for x in new if x["lat"] is not None)}, without {sum(1 for x in new if x["lat"] is None)}', flush=True)


if __name__ == '__main__':
    main()
