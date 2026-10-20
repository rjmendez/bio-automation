"""Geotag the NEFs from the phone-track window (>= 2026-07-04) that have no EXIF GPS, from the myasshurts GPX index.

The camera clock cannot be calibrated here (only 3 iNat observations fall inside the track window and two sit in track gaps), so:
  * nominal offset: camera clock = local time of the observation tz (America/New_York, EDT = UTC-4 until 2026-11-01);
  * stability gate: sample the track at capture time +-120 s (30 s steps); accept only when every sample exists and all lie within
    SPREAD_M of their median. A clock error of up to ~2 min then cannot move the location by more than SPREAD_M.
Each frame gets status ok | unstable | no_coverage and the accuracy figures. Coordinates stay in the private output only.
"""
import collections
import csv
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = os.environ.get('BIO_DATA', r'F:\bio-automation-data')
SRC = os.environ.get('BIO_SRC', r'F:\Bio')
IDX = os.path.join(ROOT, 'gpx_index', 'myasshurts.bin')
OUT = os.path.join(ROOT, 'new_raw_geotags.json')
DT = np.dtype([('t', '<f8'), ('lat', '<f8'), ('lon', '<f8'), ('sg', '<f4')])
EDT = timedelta(hours=4)          # camera local (EDT) -> UTC
CLOCK_TOL = 120                   # seconds of camera clock error the gate must absorb
SPREAD_M = float(os.environ.get('BIO_SPREAD_M', '30'))
START = '2026-07-04'
BS = chr(92)


def hav(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def header(rel):
    try:
        im = Image.open(os.path.join(SRC, rel.replace('/', os.sep)))
        ex = im.getexif()
        sub = ex.get_ifd(0x8769)
        gps = ex.get_ifd(0x8825)
        return {'time': sub.get(0x9003) or ex.get(0x0132), 'iso': sub.get(0x8827), 'camera': ex.get(0x0110),
                'gps': bool(gps and 2 in gps and 4 in gps)}
    except Exception as e:
        return {'err': type(e).__name__}


def main():
    tr = np.fromfile(IDX, dtype=DT)
    rels = []
    for r in csv.DictReader(open(os.path.join(ROOT, 'f_bio_inventory.csv'), encoding='utf-8-sig')):
        p = r['path'].replace(BS, '/')
        parts = p.split('/')
        if p.lower().endswith('.nef') and len(parts) >= 3 and parts[1] >= START and len(parts[1]) == 10:
            rels.append(p)
    print('NEFs in window:', len(rels), flush=True)
    out, stat, per_day = [], collections.Counter(), collections.defaultdict(collections.Counter)
    for rel in sorted(rels):
        h = header(rel)
        day = rel.split('/')[1]
        rec = {'rel': rel, 'day': day, **{k: h.get(k) for k in ('time', 'iso', 'camera', 'gps')}}
        if 'err' in h or not h.get('time'):
            rec['status'] = 'no_time'
        else:
            try:
                t = (datetime.strptime(h['time'][:19], '%Y:%m:%d %H:%M:%S') + EDT).replace(tzinfo=timezone.utc).timestamp()
            except ValueError:
                t = None
            if t is None:
                rec['status'] = 'no_time'
            else:
                ts = t + np.arange(-CLOCK_TOL, CLOCK_TOL + 1, 30)
                i = np.searchsorted(tr['t'], ts)
                lo, hi = np.clip(i - 1, 0, len(tr) - 1), np.clip(i, 0, len(tr) - 1)
                near = np.where(np.abs(ts - tr['t'][lo]) <= np.abs(tr['t'][hi] - ts), lo, hi)
                ok = np.abs(tr['t'][near] - ts) <= 30
                if not ok.all():
                    rec['status'] = 'no_coverage'
                else:
                    la, lo_ = tr['lat'][near], tr['lon'][near]
                    mla, mlo = float(np.median(la)), float(np.median(lo_))
                    spread = float(hav(la, lo_, mla, mlo).max())
                    ctr = near[len(near) // 2]
                    rec.update(lat=float(tr['lat'][ctr]), lon=float(tr['lon'][ctr]), loc_sigma_m=float(tr['sg'][near].max()),
                               loc_spread_m=round(spread, 1), loc_source='gpx_myasshurts')
                    rec['status'] = 'ok' if spread <= SPREAD_M else 'unstable'
        stat[rec['status']] += 1
        per_day[day[:7]][rec['status']] += 1
        out.append(rec)
    json.dump({'offset': 'EDT -4h nominal', 'clock_tol_s': CLOCK_TOL, 'spread_m': SPREAD_M, 'items': out}, open(OUT, 'w'))
    print('status:', dict(stat))
    for k in sorted(per_day):
        print(' ', k, dict(per_day[k]))
    iso = [r['iso'] for r in out if r.get('iso')]
    if iso:
        a = np.array(iso)
        print(f'ISO: median {np.median(a):.0f}, >=800: {int((a >= 800).sum())} of {len(a)}')
    ok = [r for r in out if r['status'] == 'ok']
    if ok:
        print(f"ok accuracy: median sigma {np.median([r['loc_sigma_m'] for r in ok]):.1f} m, median spread {np.median([r['loc_spread_m'] for r in ok]):.1f} m")
    print('with camera GPS in header:', sum(1 for r in out if r.get('gps')))


if __name__ == '__main__':
    main()
