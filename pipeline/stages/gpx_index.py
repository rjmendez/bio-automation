"""Index the myasshurts (field phone) GPX tracks into one compact, time-sorted binary file. Runs in WSL, stdlib only.

Sources (device id prefix 'myasshurts-'): daily ~/gpx/YYYY-MM-DD/*.gpx, ~/gpx-backfilled/YYYY-MM-DD/*.gpx and the monthly
~/gpx/_month/*.gpx bundles (not the *_lite ones). Output records: <t_epoch_s float64, lat float64, lon float64, sigma_m float32>,
duplicate timestamps collapsed (smallest sigma wins). The output holds private location history: it stays in the data dir.
"""
import glob
import os
import re
import struct
import sys
import time
from datetime import datetime, timezone

DEVICE = os.environ.get('BIO_DEVICE', 'myasshurts-')
OUT = os.environ.get('BIO_GPX_OUT', '/mnt/f/bio-automation-data/gpx_index/myasshurts.bin')
HOME = os.path.expanduser('~')
PAT = re.compile(r'<trkpt lat="(-?[\d.]+)" lon="(-?[\d.]+)">(?:.*?)<time>([^<]+)</time>(?:.*?<dama:sigma_m>([\d.]+)</dama:sigma_m>)?')

files = []
for root in ('gpx', 'gpx-backfilled'):
    files += glob.glob(os.path.join(HOME, root, '20*-*-*', DEVICE + '*.gpx'))
    files += [f for f in glob.glob(os.path.join(HOME, root, '_month', DEVICE + '*.gpx')) if not f.endswith('_lite.gpx')]
files = sorted(set(f for f in files if os.path.isfile(f)))
print('files:', len(files), flush=True)

pts = {}
t0 = time.time()
for i, f in enumerate(files, 1):
    n = 0
    with open(f, 'r', encoding='utf-8', errors='replace') as fh:
        for line in fh:
            m = PAT.search(line)
            if not m:
                continue
            try:
                t = datetime.fromisoformat(m.group(3).replace('Z', '+00:00')).astimezone(timezone.utc).timestamp()
            except ValueError:
                continue
            sg = float(m.group(4)) if m.group(4) else 99.0
            key = round(t, 1)
            old = pts.get(key)
            if old is None or sg < old[2]:
                pts[key] = (float(m.group(1)), float(m.group(2)), sg)
            n += 1
    if i % 10 == 0 or i == len(files):
        print(f'{i}/{len(files)} files, {len(pts)} points, {time.time() - t0:.0f}s', flush=True)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
keys = sorted(pts)
with open(OUT, 'wb') as fh:
    for k in keys:
        lat, lon, sg = pts[k]
        fh.write(struct.pack('<dddf', k, lat, lon, sg))
print('points:', len(keys), '| first', datetime.fromtimestamp(keys[0], timezone.utc).isoformat(), '| last', datetime.fromtimestamp(keys[-1], timezone.utc).isoformat(), flush=True)
print('written', OUT, os.path.getsize(OUT), 'bytes', flush=True)
