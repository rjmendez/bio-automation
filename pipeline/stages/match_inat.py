"""Match iNaturalist observation photos to LightCraft library frames by image similarity.

Read-only: queries the running LightCraft app (control channel) and the saved iNat export.
Writes only under F:\\bio-automation-data. Resumable via on-disk caches.

usage: match_inat.py [--days YYYY-MM-DD ...] [--window SECONDS] [--out FILE]
"""
import argparse, base64, datetime as dt, hashlib, io, json, os, subprocess, sys, threading, time, urllib.request
import numpy as np
from PIL import Image

ROOT = r'F:\bio-automation-data'
EXE = r'C:\Program Files\LightCraft\lightcraft-cli.exe'
CACHE_INAT = os.path.join(ROOT, 'cache', 'inat'); CACHE_REND = os.path.join(ROOT, 'cache', 'render')
os.makedirs(CACHE_INAT, exist_ok=True); os.makedirs(CACHE_REND, exist_ok=True)

class Mcp:
    def __init__(self):
        self.p = subprocess.Popen([EXE, 'mcp', '--connect', '127.0.0.1:7980'], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.n = 1
        self._send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18",
                    "capabilities": {}, "clientInfo": {"name": "match_inat", "version": "1"}}})
        self._read(1)
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
    def _send(self, o): self.p.stdin.write(json.dumps(o) + "\n"); self.p.stdin.flush()
    def _read(self, i):
        while True:
            l = self.p.stdout.readline()
            if not l: raise RuntimeError('mcp closed')
            d = json.loads(l)
            if d.get('id') == i: return d
    def call(self, name, args):
        self.n += 1
        self._send({"jsonrpc": "2.0", "id": self.n, "method": "tools/call", "params": {"name": name, "arguments": args}})
        return self._read(self.n)['result']

def feat(img):
    g = np.asarray(img.convert('L').resize((32, 32), Image.LANCZOS), dtype=np.float64)
    g = (g - g.mean()) / (g.std() + 1e-6)
    c = np.asarray(img.convert('RGB').resize((32, 32), Image.LANCZOS)) // 64
    idx = c[..., 0] * 16 + c[..., 1] * 4 + c[..., 2]
    h = np.bincount(idx.ravel(), minlength=64).astype(np.float64); h /= h.sum()
    return g, h

def inat_img(photo):
    pid = photo['id']; path = os.path.join(CACHE_INAT, f'{pid}.jpg')
    if not os.path.exists(path):
        url = photo['url'].replace('square', 'small')
        req = urllib.request.Request(url, headers={'User-Agent': 'bio-automation/0.1 (personal matching)'})
        data = urllib.request.urlopen(req, timeout=30).read()
        open(path, 'wb').write(data)
    return Image.open(path)

def render(mcp, fid):
    path = os.path.join(CACHE_REND, f'{fid}.jpg')
    if not os.path.exists(path):
        if THROTTLE:
            time.sleep(THROTTLE)  # ease contention with the running LightCraft scan
        r = mcp.call('render_preview', {'id': fid, 'max_side': 240})
        for c in r.get('content', []):
            if c.get('type') == 'image':
                open(path, 'wb').write(base64.b64decode(c['data'])); break
        else:
            return None
    return Image.open(path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', nargs='*'); ap.add_argument('--window', type=int, default=180)
    ap.add_argument('--out', default=os.path.join(ROOT, 'matches_stage1.json'))
    ap.add_argument('--throttle', type=float, default=0.0, help='seconds to sleep before each uncached render')
    a = ap.parse_args()
    global THROTTLE
    THROTTLE = a.throttle
    obs = json.load(open(os.path.join(ROOT, 'inat_observations.json')))['observations']
    mcp = Mcp()
    st = json.loads(mcp.call('cmd_catalog_stats', {})['content'][0]['text'])
    libdays = sorted({d for blk in st['byDate'] for d, _ in blk['days']})
    days = a.days or libdays
    P = dt.datetime.fromisoformat
    frames_by_day = {}
    for day in days:
        if day not in libdays: continue
        r = json.loads(mcp.call('query_photos', {'filter': {'dateFrom': day, 'dateTo': day}, 'limit': 5000,
                                                 'sort': {'key': 'captureDate', 'ascending': True}})['content'][0]['text'])
        ph = [p for p in r['photos'] if not p.get('deleted')]
        raws = {p['fileName'].rsplit('.', 1)[0] for p in ph if p['kind'] == 'raw'}
        frames_by_day[day] = [p for p in ph if p['kind'] == 'raw' or p['fileName'].rsplit('.', 1)[0] not in raws]
    results = []; t0 = time.time(); nobs = 0
    todo = [o for o in obs if o['observed_on'] in frames_by_day and o['photos'] and o['time']]
    print(f'days {len(frames_by_day)} | observations to match {len(todo)}', flush=True)
    for o in todo:
        t = P(o['time']); win = dt.timedelta(seconds=a.window)

        def frame_time(s, ref=t):
            d = P(s)
            return d.replace(tzinfo=ref.tzinfo) if d.tzinfo is None else d  # naive capture times: assume the observation's zone
        try:
            cand = [p for p in frames_by_day[o['observed_on']] if abs(frame_time(p['captured']) - t) <= win]
        except Exception as e:
            print('  candidate error obs', o['id'], type(e).__name__, flush=True)
            cand = []
        row = {'obs': o['id'], 'time': o['time'], 'taxon': (o['taxon'] or {}).get('name'), 'common': (o['taxon'] or {}).get('common'),
               'obscured': o['obscured'], 'n_candidates': len(cand), 'photos': []}
        if cand:
            cf = {}
            for p in cand:
                im = render(mcp, p['id'])
                if im is not None: cf[p['id']] = (p, feat(im))
            for ph in o['photos']:
                try:
                    g1, h1 = feat(inat_img(ph))
                except Exception as e:
                    row['photos'].append({'photo': ph['id'], 'error': type(e).__name__}); continue
                sc = []
                for fid, (p, (g2, h2)) in cf.items():
                    corr = float((g1 * g2).mean()); hi = float(np.minimum(h1, h2).sum())
                    sc.append((0.6 * corr + 0.4 * hi, corr, hi, p['id'], p['fileName'], p['captured']))
                sc.sort(reverse=True)
                best = sc[0] if sc else None; second = sc[1] if len(sc) > 1 else None
                row['photos'].append({'photo': ph['id'], 'best': best and {'score': round(best[0], 4), 'corr': round(best[1], 4), 'hist': round(best[2], 4),
                                      'frame_id': best[3], 'file': best[4], 'captured': best[5]},
                                      'second': second and {'score': round(second[0], 4), 'file': second[4]},
                                      'margin': round(best[0] - second[0], 4) if second else None})
        results.append(row); nobs += 1
        if nobs % 50 == 0:
            print(f'{nobs}/{len(todo)} obs, {time.time()-t0:.0f}s', flush=True)
            json.dump(results, open(a.out, 'w'))
    json.dump(results, open(a.out, 'w'))
    print('done', nobs, 'obs ->', a.out, f'{time.time()-t0:.0f}s', flush=True)

if __name__ == '__main__':
    main()
