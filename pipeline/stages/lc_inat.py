"""LightCraft <-> iNaturalist review console adapter (companion integration through LightCraft's own MCP/control surface).

  lc_inat.py pull --keyword "inat-queue"        photos carrying that keyword
  lc_inat.py pull --album "iNat"                photos in that album (name resolved by probing album ids)
  lc_inat.py pull --day 2026-09-27 [--day ...]  all photos captured that day
  lc_inat.py pull --ids 15581,15582
  lc_inat.py writeback [--dry-run]              mark photos of posted observations (keywords "iNat", "iNat obs <id>", green label)

pull: reads photo info from LightCraft, exports 2048 px sRGB JPEG upload copies with the photo's own develop settings (pristine raws get
develop.auto for the export only and are reset afterwards), finds a location (LightCraft GPS > phone track, stability gated, using the
photo's own UTC offset > GPS frame of the same session within 15 min > none), groups bursts into proposed observations and writes
inat_queue_lc.json for the console (BIO_QUEUE_TAG=_lc). Nothing is posted and no photo is edited besides the optional writeback.

Headless engine (BIO_LC_HEADLESS=1) needs the GUI closed (one client holds the library); without it the running app's control channel
(lightcraft --control 7980) is used. Coordinates never leave the local files.
"""
import argparse
import collections
import json
import math
import os
import struct
import sys
from datetime import datetime, timezone

import numpy as np
import piexif
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from pilot_style import Mcp, text
except ImportError:                      # repo layout
    from lightcraft_mcp import Mcp, text

Image.MAX_IMAGE_PIXELS = None
ROOT = os.environ.get('BIO_DATA', r'F:\bio-automation-data')
TAG = os.environ.get('BIO_QUEUE_TAG', '_lc')
UPLOAD = os.environ.get('BIO_LC_UPLOAD', r'F:\Bio-export\inat-upload-lc')
QUEUE = os.path.join(ROOT, f'inat_queue{TAG}.json')
GPX = os.path.join(ROOT, 'gpx_index', 'myasshurts.bin')
DT = np.dtype([('t', '<f8'), ('lat', '<f8'), ('lon', '<f8'), ('sg', '<f4')])
BURST_S, NEIGHBOR_S, AGREE_M, SPREAD_M, CLOCK_TOL = 120, 900, 100.0, 30.0, 120
KEYWORD_POSTED = 'iNat'


def run(m, cmd, params=None):
    return text(m.call('command_run', {'id': cmd, 'params': params or {}}))


def hav(a, b):
    r = 6371000.0
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    x = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(b[1] - a[1]) / 2) ** 2
    return 2 * r * math.asin(math.sqrt(x))


def parse_gps(g):
    try:
        if isinstance(g, (list, tuple)) and len(g) >= 2:
            return float(g[0]), float(g[1])
        if isinstance(g, dict):
            la = g.get('lat', g.get('latitude'))
            lo = g.get('lon', g.get('lng', g.get('longitude')))
            return (float(la), float(lo)) if la is not None and lo is not None else None
        if isinstance(g, str) and ',' in g:
            a, b = g.split(',')[:2]
            return float(a), float(b)
    except (TypeError, ValueError):
        return None
    return None


def dms(v):
    v = abs(v)
    d = int(v)
    mi = int((v - d) * 60)
    return ((d, 1), (mi, 1), (round((v - d - mi / 60) * 3600 * 10000), 10000))


def set_gps(path, lat, lon, err_m):
    exif = piexif.load(path)
    g = exif.setdefault('GPS', {})
    g[piexif.GPSIFD.GPSVersionID] = (2, 3, 0, 0)
    g[piexif.GPSIFD.GPSLatitudeRef] = 'N' if lat >= 0 else 'S'
    g[piexif.GPSIFD.GPSLatitude] = dms(lat)
    g[piexif.GPSIFD.GPSLongitudeRef] = 'E' if lon >= 0 else 'W'
    g[piexif.GPSIFD.GPSLongitude] = dms(lon)
    g[piexif.GPSIFD.GPSHPositioningError] = (int(err_m), 1)
    piexif.insert(piexif.dump(exif), path)


def album_id(m, name):
    """Album names are not listed by the control surface; probe ids until Show Source reports the name."""
    for i in range(1, 80):
        try:
            run(m, 'library.source', {'kind': 'album', 'id': i})
            lab = (run(m, 'library.state') or {}).get('sourceLabel')
        except Exception:
            continue
        if lab and lab.strip().lower() == name.strip().lower():
            return i
    return None


def select_ids(m, a):
    ids = []
    if a.ids:
        return [int(x) for x in a.ids.split(',') if x.strip()]
    if a.album:
        aid = album_id(m, a.album)
        if aid is None:
            sys.exit(f'no album named {a.album!r}')
        run(m, 'library.source', {'kind': 'album', 'id': aid})
        flt = {}
    elif a.keyword:
        run(m, 'library.source', {'kind': 'all'})
        flt = {'keyword': a.keyword}
    elif a.day:
        run(m, 'library.source', {'kind': 'all'})
        flt = None
    else:
        sys.exit('give --keyword, --album, --day or --ids')
    off = 0
    while True:
        q = text(m.call('query_photos', {'filter': flt, 'offset': off, 'limit': 500})) if flt is not None else \
            text(m.call('query_photos', {'offset': off, 'limit': 500}))
        ph = (q or {}).get('photos', [])
        for p in ph:
            if a.day and (p.get('captured') or '')[:10] not in a.day:
                continue
            if p.get('deleted') or p.get('kind') not in ('raw', 'jpeg', 'image', 'heif', 'tiff', None):
                if p.get('deleted'):
                    continue
            ids.append(p['id'])
        off += len(ph)
        if not ph or off >= (q or {}).get('total', 0):
            break
    return list(dict.fromkeys(ids))


def load_track():
    return np.fromfile(GPX, dtype=DT) if os.path.exists(GPX) else None


def gpx_position(tr, t):
    if tr is None or len(tr) == 0:
        return None
    ts = t + np.arange(-CLOCK_TOL, CLOCK_TOL + 1, 30)
    i = np.searchsorted(tr['t'], ts)
    lo, hi = np.clip(i - 1, 0, len(tr) - 1), np.clip(i, 0, len(tr) - 1)
    near = np.where(np.abs(ts - tr['t'][lo]) <= np.abs(tr['t'][hi] - ts), lo, hi)
    if not (np.abs(tr['t'][near] - ts) <= 30).all():
        return None
    pts = [(float(tr['lat'][j]), float(tr['lon'][j])) for j in near]
    med = (float(np.median([p[0] for p in pts])), float(np.median([p[1] for p in pts])))
    spread = max(hav(p, med) for p in pts)
    if spread > SPREAD_M:
        return None
    c = near[len(near) // 2]
    return float(tr['lat'][c]), float(tr['lon'][c]), float(tr['sg'][near].max()), spread


def pull(a):
    os.makedirs(UPLOAD, exist_ok=True)
    m = Mcp()
    ids = select_ids(m, a)
    print(f'photos selected in LightCraft: {len(ids)}', flush=True)
    tr = load_track()
    items = []
    for n, pid in enumerate(ids, 1):
        ins = run(m, 'photo.inspect', {'id': pid}) or {}
        meta = ins.get('meta') or {}
        cap = ins.get('captured') or ''
        try:
            dt = datetime.fromisoformat(cap)
            epoch = dt.timestamp()
            off = dt.utcoffset().total_seconds() if dt.utcoffset() is not None else None
            time_str = dt.strftime('%Y:%m:%d %H:%M:%S')
            day = dt.strftime('%Y-%m-%d')
        except ValueError:
            epoch, off, time_str, day = None, None, None, 'unknown'
        gps = parse_gps(meta.get('gps'))
        src = (ins.get('source') or {}).get('path') or ''
        stem = os.path.splitext(ins.get('file_name') or str(pid))[0]
        out = os.path.join(UPLOAD, day[:4], day, f'{day}_{stem}_{pid}.jpg')
        os.makedirs(os.path.dirname(out), exist_ok=True)
        err = None
        if not os.path.exists(out):
            pristine = not ins.get('edited')
            try:
                m.call('select_photos', {'ids': [pid], 'active': pid})
                if pristine:
                    run(m, 'develop.auto')
                m.call('export', {'id': pid, 'path': out, 'format': 'jpeg', 'quality': 90, 'longEdge': 2048, 'colorSpace': 'srgb',
                                  'metadata': 'all', 'conflict': 'overwrite'})
            except Exception as e:
                err = f'{type(e).__name__}: {str(e)[:100]}'
            finally:
                if pristine:
                    try:
                        run(m, 'develop.reset', {'ids': [pid]})
                    except Exception:
                        pass
        it = {'src': 'LC', 'lc_id': pid, 'rel': os.path.basename(out), 'day': day, 'raw': src, 'w': ins.get('width'), 'h': ins.get('height'),
              'time': time_str, 't': epoch, 'tz_offset': off, 'camera': meta.get('camera'), 'iso': meta.get('iso'),
              'copy': out if os.path.exists(out) else None, 'copy_error': err, 'release': 'unscreened', 'title': meta.get('title') or '',
              'keywords': meta.get('keywords') or []}
        if gps:
            it.update(lat=gps[0], lon=gps[1], loc_source='lightcraft_gps', loc_note='GPS in the LightCraft catalog')
        else:
            g = gpx_position(tr, epoch) if epoch is not None else None
            if g:
                it.update(lat=g[0], lon=g[1], loc_source='gpx_myasshurts', loc_note=f'phone track, sigma {g[2]:.1f} m, spread {g[3]:.0f} m',
                          positional_accuracy=int(max(g[2], g[3], 5)))
            else:
                it.update(loc_source='none', loc_note='no GPS in catalog and no stable phone track')
        items.append(it)
        if n % 10 == 0 or n == len(ids):
            print(f'  {n}/{len(ids)}', flush=True)

    # neighbour inheritance (same day, <= 15 min, both sides must agree)
    refs = collections.defaultdict(list)
    for it in items:
        if it.get('lat') is not None and it['t'] is not None and it['loc_source'] != 'neighbor':
            refs[it['day']].append((it['t'], it['lat'], it['lon']))
    for v in refs.values():
        v.sort()
    for it in items:
        if it.get('lat') is None and it['t'] is not None:
            r = refs.get(it['day'], [])
            b = [x for x in r if x[0] <= it['t'] and it['t'] - x[0] <= NEIGHBOR_S]
            af = [x for x in r if x[0] > it['t'] and x[0] - it['t'] <= NEIGHBOR_S]
            nb, na = (b[-1] if b else None), (af[0] if af else None)
            pick = None
            if nb and na:
                pick = (nb if it['t'] - nb[0] <= na[0] - it['t'] else na) if hav(nb[1:], na[1:]) <= AGREE_M else None
            else:
                pick = nb or na
            if pick:
                it.update(lat=pick[1], lon=pick[2], loc_source='neighbor', loc_note=f'GPS frame {abs(it["t"] - pick[0]):.0f}s away',
                          positional_accuracy=50)
    for it in items:                      # embed the found coordinates in the copy so the web uploader would also read them
        if it.get('copy') and it.get('lat') is not None and it['loc_source'] in ('gpx_myasshurts', 'neighbor'):
            try:
                set_gps(it['copy'], it['lat'], it['lon'], it.get('positional_accuracy') or 50)
            except Exception as e:
                it['loc_note'] += f' (GPS not written into the copy: {type(e).__name__})'

    items.sort(key=lambda x: (x['day'], x['camera'] or '', x['t'] if x['t'] is not None else -1))
    gid, last = 0, None
    for it in items:
        key = (it['day'], it['camera'])
        same = False
        if last and last['key'] == key and it['t'] is not None and last['t'] is not None and it['t'] - last['t'] <= BURST_S:
            same = it.get('lat') is None or last['lat'] is None or hav((it['lat'], it['lon']), (last['lat'], last['lon'])) <= 50
        if not same:
            gid += 1
        it['group'] = gid
        last = {'key': key, 't': it['t'], 'lat': it.get('lat'), 'lon': it.get('lon')}
    json.dump(items, open(QUEUE, 'w'))
    c = collections.Counter(i['loc_source'] for i in items)
    print(f'queue {QUEUE}: {len(items)} photos in {gid} proposed observations | location {dict(c)} | '
          f'copies {sum(1 for i in items if i["copy"])} | export errors {sum(1 for i in items if i["copy_error"])}')
    print('next: BIO_QUEUE_TAG=_lc inat_screen / inat_suggest / inat_console (see README)')


def writeback(a):
    posted = os.path.join(ROOT, f'inat_posted{TAG}.jsonl')
    if not os.path.exists(posted):
        sys.exit('nothing posted yet')
    q = json.load(open(QUEUE))
    by_group = collections.defaultdict(list)
    for it in q:
        by_group[it['group']].append(it)
    decs = {}
    dp = os.path.join(ROOT, f'inat_decisions{TAG}.jsonl')
    for line in open(dp, encoding='utf-8'):
        r = json.loads(line)
        decs[r['group']] = r
    todo = []
    for line in open(posted, encoding='utf-8'):
        r = json.loads(line)
        if r.get('status') == 'posted' and r.get('observation_id'):
            order = (decs.get(r['group']) or {}).get('photo_order') or []
            for idx in order:
                if 0 <= idx < len(q) and q[idx].get('lc_id'):
                    todo.append((q[idx]['lc_id'], r['observation_id']))
    print(f'photos to mark: {len(todo)}')
    if a.dry_run or not todo:
        return
    m = Mcp()
    for pid, obs in todo:
        run(m, 'photo.setMeta', {'ids': [pid], 'addKeywords': [KEYWORD_POSTED, f'iNat obs {obs}']})
        run(m, 'photo.label', {'ids': [pid], 'label': 'green'})
    print('done')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('pull')
    p.add_argument('--keyword')
    p.add_argument('--album')
    p.add_argument('--day', action='append')
    p.add_argument('--ids')
    w = sub.add_parser('writeback')
    w.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    pull(a) if a.cmd == 'pull' else writeback(a)


if __name__ == '__main__':
    main()
