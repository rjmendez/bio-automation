"""Match iNaturalist photos to existing JPEG exports on the NAS (no NEF reading, no LightCraft).

Per day folder: decode each JPEG at 1/8 scale -> 32x32 gray feature (cached), compare to the iNat photo's feature
(correlation >= 0.95, same aspect ratio). Duplicate exports of one photo collapse to one. NEF/DNG is paired by name from the inventory.
Read-only against the NAS. Resumable: features cached in nas_features.pkl, results saved after every day.
"""
import collections, csv, concurrent.futures as cf, json, os, pickle, sys, time, urllib.request
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = os.environ.get('BIO_DATA', r'F:\bio-automation-data')
NAS = os.environ.get('BIO_NAS', r'Y:\Bio')
BS = chr(92)
TH, MARGIN = 0.95, 0.01
CACHE_INAT = os.path.join(ROOT, 'cache', 'inat')
FEAT = os.path.join(ROOT, 'nas_features.pkl')
OUT = os.path.join(ROOT, 'nas_matches.json')


def feat_from_image(im):
    g = np.asarray(im.convert('L').resize((32, 32), Image.LANCZOS), dtype=np.float32)
    g = (g - g.mean()) / (g.std() + 1e-6)
    return g.ravel().astype(np.float16)


def jpeg_feature(rel):
    p = os.path.join(NAS, rel.replace('/', os.sep))
    try:
        im = Image.open(p)
        w, h = im.size
        im.draft('RGB', (128, 128))
        return rel, (w, h, feat_from_image(im))
    except Exception as e:
        return rel, ('ERR', type(e).__name__)


def inat_feature(photo):
    path = os.path.join(CACHE_INAT, f"{photo['id']}.jpg")
    if not os.path.exists(path):
        url = photo['url'].replace('square', 'small')
        req = urllib.request.Request(url, headers={'User-Agent': 'bio-automation/0.1 (personal matching)'})
        open(path, 'wb').write(urllib.request.urlopen(req, timeout=30).read())
    return feat_from_image(Image.open(path))


def main():
    inv = [r['path'].replace(BS, '/') for r in csv.DictReader(open(os.path.join(ROOT, 'nas_inventory.csv'), encoding='utf-8-sig'))]
    jp_by_day, raw_by = collections.defaultdict(list), {}
    for p in inv:
        parts = p.split('/')
        if len(parts) < 3 or not parts[0].isdigit() or len(parts[1]) != 10:
            continue
        day = parts[1]
        stem, ext = os.path.splitext(parts[-1])
        if ext.lower() == '.jpg':
            jp_by_day[day].append(p)
        elif ext.lower() in ('.nef', '.dng'):
            raw_by[(day, stem.lower())] = p
    inat = json.load(open(os.path.join(ROOT, 'inat_observations.json'), encoding='utf-8'))['observations']
    done_local = set()
    try:
        for r in json.load(open(os.path.join(ROOT, 'matches_stage1d.json'), encoding='utf-8')):
            if any(p.get('best') and p['best']['corr'] >= TH for p in r['photos']):
                done_local.add(r['obs'])
    except Exception:
        pass
    obs_by_day = collections.defaultdict(list)
    for o in inat:
        d = o.get('observed_on')
        if d in jp_by_day and o['photos'] and o['id'] not in done_local:
            obs_by_day[d].append(o)
    days = sorted(obs_by_day, key=lambda d: -len(obs_by_day[d]))
    print(f'days with NAS JPEGs and unmatched observations: {len(days)} | observations: {sum(len(v) for v in obs_by_day.values())} | JPEGs to read: {sum(len(jp_by_day[d]) for d in days)}', flush=True)
    feats = pickle.load(open(FEAT, 'rb')) if os.path.exists(FEAT) else {}
    results = json.load(open(OUT, encoding='utf-8')) if os.path.exists(OUT) else {'days': {}, 'matches': []}
    t0 = time.time()
    nread = 0
    for di, day in enumerate(days, 1):
        if day in results['days']:
            continue
        todo = [p for p in jp_by_day[day] if p not in feats]
        with cf.ThreadPoolExecutor(6) as ex:
            for rel, f in ex.map(jpeg_feature, todo):
                feats[rel] = f
        nread += len(todo)
        J = [(p, feats[p]) for p in jp_by_day[day] if feats.get(p) and feats[p][0] != 'ERR']
        if J:
            M = np.stack([f[2].astype(np.float32) for _, f in J])           # n x 1024
            dims = [(f[0], f[1]) for _, f in J]
        day_m = 0
        for o in obs_by_day[day]:
            t = (o.get('taxon') or {}).get('name')
            for ph in o['photos']:
                if not J:
                    break
                try:
                    v = inat_feature(ph).astype(np.float32)
                except Exception:
                    continue
                d_ = ph.get('dims') or {}
                ar = (max(d_.get('width', 0), d_.get('height', 0)) / max(min(d_.get('width', 1), d_.get('height', 1)), 1)) if d_ else None
                corr = (M @ v) / 1024.0
                if ar:   # aspect gate: an uncropped export has the same aspect ratio as the iNat photo
                    for i, (w, h) in enumerate(dims):
                        if abs(max(w, h) / max(min(w, h), 1) - ar) > 0.03:
                            corr[i] = -1
                order = np.argsort(-corr)
                best = float(corr[order[0]])
                if best < TH:
                    continue
                same = [J[i][0] for i in order if corr[i] >= best - MARGIN]          # equivalent copies of the same export
                rest = [float(corr[i]) for i in order if corr[i] < best - MARGIN]
                second = rest[0] if rest else None
                if second is not None and best - second < MARGIN * 2 and len(same) == 1:
                    continue                                                        # ambiguous between different photos
                prefer = sorted(same, key=lambda p: (('inat-lr' not in p.lower()), len(p)))[0]
                stem = os.path.splitext(os.path.basename(prefer))[0].lower()
                raw = raw_by.get((day, stem)) or raw_by.get((day, stem.split(' (')[0]))
                results['matches'].append({'obs': o['id'], 'photo': ph['id'], 'day': day, 'jpeg': prefer, 'copies': same, 'raw': raw,
                                           'corr': round(best, 4), 'second': round(second, 4) if second is not None else None, 'taxon': t})
                day_m += 1
        results['days'][day] = {'obs': len(obs_by_day[day]), 'jpegs': len(J), 'matched_photos': day_m}
        json.dump(results, open(OUT, 'w'))
        pickle.dump(feats, open(FEAT, 'wb'))
        el = time.time() - t0
        print(f'day {day} ({di}/{len(days)}): jpegs {len(J)} obs {len(obs_by_day[day])} matched photos {day_m} | read {nread} files, {el:.0f}s', flush=True)
    m = results['matches']
    print('\nDONE. matched photos:', len(m), '| distinct observations:', len({x['obs'] for x in m}),
          '| with a paired NEF/DNG:', sum(1 for x in m if x['raw']), '| distinct JPEG exports:', len({x['jpeg'] for x in m}), flush=True)


if __name__ == '__main__':
    main()
