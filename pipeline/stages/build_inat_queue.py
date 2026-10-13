"""Build the iNaturalist upload queue from inat_candidates.json (exports not on iNat). Never uploads anything.

Location tiers (recorded per item, never mixed up):
  exif      GPS in the export's own EXIF
  neighbor  same-day frame with EXIF GPS within NEIGHBOR_S seconds (both sides must agree within AGREE_M when both exist)
  none      "location needed": the item stays in the queue but cannot be uploaded until the owner supplies a place
Items are grouped into proposed observations (same day, same camera, <= BURST_S seconds apart, same place when known).
Upload copies: <= 2048 px long edge, q90, EXIF + ICC kept, GPS embedded (neighbor-sourced GPS carries a 50 m error).
Outputs (PRIVATE, data dir): inat_queue.json / inat_queue.csv, copies under UPLOAD_DIR. Prints counts only, never coordinates.
"""
import collections
import csv
import json
import math
import os
import sys
from datetime import datetime

import piexif
import visionlib
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = os.environ.get('BIO_DATA', r'F:\bio-automation-data')
SRC = os.environ.get('BIO_SRC', r'F:\Bio')
NAS = os.environ.get('BIO_NAS', r'Y:\rjmendez-admin\DESKTOP-BVRDK4J\Data\F\Bio')
UPLOAD_DIR = os.environ.get('BIO_INAT_UPLOAD', r'F:\Bio-export\inat-upload')
NEIGHBOR_S = int(os.environ.get('BIO_NEIGHBOR_S', '900'))
AGREE_M = 100.0
BURST_S = 120
LONG_EDGE = 2048
BS = chr(92)


def secs(t):
    try:
        return datetime.strptime(t[:19], '%Y:%m:%d %H:%M:%S').timestamp()
    except Exception:
        return None


def hav(a, b):
    r = 6371000.0
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    x = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(b[1] - a[1]) / 2) ** 2
    return 2 * r * math.asin(math.sqrt(x))


def dms(v):
    v = abs(v)
    d = int(v)
    m = int((v - d) * 60)
    s = round((v - d - m / 60) * 3600 * 10000)
    return ((d, 1), (m, 1), (s, 10000))


def write_copy(item, dst):
    root = SRC if item['src'] == 'F' else NAS
    p = os.path.join(root, item['rel'].replace('/', os.sep))
    im = Image.open(p)
    exif = piexif.load(im.info['exif']) if im.info.get('exif') else {'0th': {}, 'Exif': {}, 'GPS': {}, '1st': {}, 'thumbnail': None}
    exif['thumbnail'] = None
    exif['1st'] = {}
    if item['loc_source'] == 'neighbor':
        g = exif.setdefault('GPS', {})
        g[piexif.GPSIFD.GPSVersionID] = (2, 3, 0, 0)
        g[piexif.GPSIFD.GPSLatitudeRef] = 'N' if item['lat'] >= 0 else 'S'
        g[piexif.GPSIFD.GPSLatitude] = dms(item['lat'])
        g[piexif.GPSIFD.GPSLongitudeRef] = 'E' if item['lon'] >= 0 else 'W'
        g[piexif.GPSIFD.GPSLongitude] = dms(item['lon'])
        g[piexif.GPSIFD.GPSHPositioningError] = (50, 1)
    # the pixel size recorded in EXIF would be wrong after resizing
    for k in (piexif.ExifIFD.PixelXDimension, piexif.ExifIFD.PixelYDimension):
        exif.get('Exif', {}).pop(k, None)
    icc = im.info.get('icc_profile')
    w, h = im.size
    if max(w, h) > LONG_EDGE:
        im.draft('RGB', (LONG_EDGE, LONG_EDGE))
        im = im.convert('RGB')
        s = LONG_EDGE / max(im.size)
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    else:
        im = im.convert('RGB')
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    kw = {'quality': 90, 'exif': piexif.dump(exif)}
    if icc:
        kw['icc_profile'] = icc
    im.save(dst, 'JPEG', **kw)


def main():
    cand = json.load(open(os.path.join(ROOT, 'inat_candidates.json')))
    new = [c for c in cand if not c['already_on_inat']]
    vision = json.load(open(os.path.join(ROOT, 'vision_files.json'), encoding='utf-8'))

    # GPS reference frames: every prepared export with EXIF GPS (on iNat or not), per (src, day)
    refs = collections.defaultdict(list)
    for c in cand:
        t = secs(c.get('time') or '')
        if c['lat'] is not None and t is not None:
            refs[(c['src'], c['day'])].append((t, c['lat'], c['lon']))
    for v in refs.values():
        v.sort()

    stat = collections.Counter()
    items = []
    for c in new:
        it = {k: c[k] for k in ('src', 'rel', 'day', 'raw', 'w', 'h', 'time', 'camera')}
        it['t'] = secs(c.get('time') or '')
        if c['lat'] is not None:
            it.update(lat=c['lat'], lon=c['lon'], loc_source='exif', loc_note='')
        else:
            it.update(loc_source='none', loc_note='no GPS in file')
            r = refs.get((c['src'], c['day']), [])
            if it['t'] is not None and r:
                before = [x for x in r if x[0] <= it['t'] and it['t'] - x[0] <= NEIGHBOR_S]
                after = [x for x in r if x[0] > it['t'] and x[0] - it['t'] <= NEIGHBOR_S]
                nb, na = (before[-1] if before else None), (after[0] if after else None)
                pick = None
                if nb and na:
                    if hav(nb[1:], na[1:]) <= AGREE_M:
                        pick = nb if it['t'] - nb[0] <= na[0] - it['t'] else na
                    else:
                        it['loc_note'] = 'neighbours disagree (moved)'
                else:
                    pick = nb or na
                if pick:
                    it.update(lat=pick[1], lon=pick[2], loc_source='neighbor', loc_note=f'GPS frame {abs(it["t"] - pick[0]):.0f}s away')
        # release screening carries over from the Adobe-set screening when the file was screened there
        stem = os.path.splitext(os.path.basename(c['rel']))[0]
        vk = [k for k in vision if k.endswith(stem + '.jpg') or k.endswith(stem + '-Enhanced-NR.jpg')]
        flag = None
        for k in vk:
            fl = visionlib.release_flags(vision[k].get('tags'), k)
            if any(f in fl for f in ('person_visible', 'face_visible', 'identifiable_person')):
                flag = True
        it['release'] = 'flagged' if flag else ('clear' if vk else 'unscreened')
        stat[it['loc_source']] += 1
        items.append(it)

    # proposed observations: bursts of one subject
    items.sort(key=lambda x: (x['src'], x['day'], x['camera'] or '', x['t'] if x['t'] is not None else -1, x['rel']))
    gid, last = 0, None
    for it in items:
        key = (it['src'], it['day'], it['camera'])
        same = False
        if last is not None and last['key'] == key and it['t'] is not None and last['t'] is not None and it['t'] - last['t'] <= BURST_S:
            if it.get('lat') is None or last.get('lat') is None or hav((it['lat'], it['lon']), (last['lat'], last['lon'])) <= 50:
                same = True
        if not same:
            gid += 1
        it['group'] = gid
        last = {'key': key, 't': it['t'], 'lat': it.get('lat'), 'lon': it.get('lon')}

    # upload copies for items that have a location
    done = fail = 0
    for it in items:
        it['copy'] = None
        if it['loc_source'] == 'none':
            continue
        dst = os.path.join(UPLOAD_DIR, it['day'][:4], it['day'], f"g{it['group']:05d}_" + os.path.basename(it['rel']))
        try:
            if not os.path.exists(dst):
                write_copy(it, dst)
            it['copy'] = dst
            done += 1
        except Exception as e:
            it['copy_error'] = f'{type(e).__name__}: {e}'[:120]
            fail += 1
        if (done + fail) % 200 == 0:
            print(f'  copies {done + fail}', flush=True)

    json.dump(items, open(os.path.join(ROOT, 'inat_queue.json'), 'w'))
    with open(os.path.join(ROOT, 'inat_queue.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['group', 'day', 'time', 'camera', 'src', 'rel', 'loc_source', 'loc_note', 'release', 'copy'])
        for it in items:
            w.writerow([it['group'], it['day'], it['time'], it['camera'], it['src'], it['rel'], it['loc_source'], it['loc_note'], it['release'], it['copy'] or ''])
    ng = len({i['group'] for i in items})
    print(f"queue items {len(items)} in {ng} proposed observations | location {dict(stat)}")
    print(f"upload copies written {done}, failed {fail} | release {dict(collections.Counter(i['release'] for i in items))}")
    nl = collections.Counter(i['day'][:4] for i in items if i['loc_source'] == 'none')
    print('location needed by year:', dict(sorted(nl.items())))
    notes = collections.Counter(i['loc_note'] for i in items if i['loc_source'] == 'none')
    print('why none:', dict(notes))


if __name__ == '__main__':
    main()
