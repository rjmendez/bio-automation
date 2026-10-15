"""Local review console for the iNaturalist upload queue (reproduces the web uploader's card workflow on your own machine).

Loopback only (127.0.0.1). It never talks to iNaturalist except the public taxon autocomplete lookup (<= 1 request/second, identified
User-Agent). Decisions are appended to inat_decisions.jsonl; NOTHING is posted from here. Posting is a separate command (inat_post.py)
that only sends groups YOU approved one by one.

Per proposed observation you see: the merged photos, location source, date/time, advisory release flags, ranked taxon suggestions with
the accuracy measured on your own photos, and editable fields. Approve / skip / hold. Keys: A approve, S skip, H hold, J/K next/prev.
"""
import collections
import io
import json
import os
import re
import shutil
import sys
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import visionlib
import inat_export

ROOT = os.environ.get('BIO_DATA', r'F:\bio-automation-data')
PORT = int(os.environ.get('BIO_CONSOLE_PORT', '8777'))
TAG = os.environ.get('BIO_QUEUE_TAG', '')      # '' = the exported-files queue, '_lc' = the LightCraft queue
UA = 'bio-automation-console/0.1 (personal photo review tool; owner rjmendez)'
LEDGER = os.path.join(ROOT, f'inat_decisions{TAG}.jsonl')
POSTED = os.path.join(ROOT, f'inat_posted{TAG}.jsonl')
Image.MAX_IMAGE_PIXELS = None
TZ_NAMES = {'America/New_York': 'Eastern Time (US & Canada)', 'America/Chicago': 'Central Time (US & Canada)',
            'America/Denver': 'Mountain Time (US & Canada)', 'America/Los_Angeles': 'Pacific Time (US & Canada)',
            'America/Phoenix': 'Arizona', 'America/Anchorage': 'Alaska', 'Pacific/Honolulu': 'Hawaii', 'UTC': 'UTC'}
TZ_CHOICES = sorted(set(TZ_NAMES.values()))
DEFAULT_PIN_ACCURACY = 500        # metres: a day pin is a place, not a point; iNaturalist then shows the right uncertainty circle


def jload(name, default):
    p = os.path.join(ROOT, name)
    return json.load(open(p, encoding='utf-8')) if os.path.exists(p) else default


QUEUE = jload(f'inat_queue{TAG}.json', [])
SUGG = jload(f'inat_suggestions{TAG}.json', {})
VIS = jload(f'vision_inat{TAG}.json', {})
OBS = jload('inat_observations.json', {}).get('observations', [])
DAY_TZ = {}
for o in OBS:
    if o.get('observed_on') and o.get('tz'):
        DAY_TZ.setdefault(o['observed_on'], collections.Counter())[o['tz']] += 1

GROUPS = collections.OrderedDict()
for idx, it in enumerate(QUEUE):
    it['idx'] = idx
    if it.get('copy'):
        GROUPS.setdefault(it['group'], []).append(it)
LOCK = threading.Lock()


def decisions():
    cur = {}
    if os.path.exists(LEDGER):
        for line in open(LEDGER, encoding='utf-8'):
            try:
                r = json.loads(line)
                cur[r['group']] = r
            except Exception:
                pass
    return cur


PINS_PATH = os.path.join(ROOT, f'inat_daypins{TAG}.jsonl')
_pins = {'mtime': None, 'data': {}}


def daypins():
    """{day: {lat, lon, accuracy_m, label}} from the append-only pin ledger (the latest record per day wins; a clear record removes it)."""
    try:
        m = os.path.getmtime(PINS_PATH)
    except OSError:
        return {}
    if _pins['mtime'] != m:
        data = {}
        for line in open(PINS_PATH, encoding='utf-8'):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get('clear'):
                data.pop(r.get('day'), None)
            elif r.get('day'):
                data[r['day']] = r
        _pins['mtime'], _pins['data'] = m, data
    return _pins['data']


def parse_latlon(s):
    """'39.9305, -77.4406' (as copied from a map) or '39.9305 -77.4406' -> (lat, lon) or None."""
    nums = re.findall(r'-?\d+(?:\.\d+)?', str(s or ''))
    if len(nums) != 2:
        return None
    lat, lon = float(nums[0]), float(nums[1])
    if not (-90 <= lat <= 90 and -180 <= lon <= 180) or (abs(lat) < 1e-6 and abs(lon) < 1e-6):
        return None
    return lat, lon


def posted():
    s = set()
    if os.path.exists(POSTED):
        for line in open(POSTED, encoding='utf-8'):
            try:
                r = json.loads(line)
                if r.get('status') in ('posted', 'partial'):
                    s.add(r['group'])
            except Exception:
                pass
    return s


def flags_for(items):
    out = set()
    for it in items:
        t = (VIS.get(it['copy']) or {}).get('tags')
        if not t:
            out.add('unscreened')
            continue
        for f in visionlib.release_flags(t):
            out.add({'person_visible': 'person', 'identifiable_person': 'person', 'face_visible': 'face', 'text_or_logo_visible': 'text',
                     'brand_or_product_visible': 'brand', 'property_identifiable': 'property'}[f])
    return sorted(out)


def tz_from_offset(off, day):
    h = int(round(off / 3600.0))
    dst = '03-10' <= day[5:] <= '11-02'     # US daylight saving, close enough for a default the reviewer can change
    return {-4: 'Eastern Time (US & Canada)', -5: 'Central Time (US & Canada)' if dst else 'Eastern Time (US & Canada)',
            -6: 'Mountain Time (US & Canada)' if dst else 'Central Time (US & Canada)',
            -7: 'Pacific Time (US & Canada)' if dst else 'Mountain Time (US & Canada)', -8: 'Pacific Time (US & Canada)', 0: 'UTC'}.get(h)


def group_default(items):
    first = min(items, key=lambda i: i.get('t') or 0)
    t = first.get('time') or ''
    try:
        obs_str = datetime.strptime(t[:19], '%Y:%m:%d %H:%M:%S').strftime('%Y-%m-%d %H:%M:%S')
    except ValueError:
        obs_str = first['day']
    tz = None
    c = DAY_TZ.get(first['day'])
    if c:
        tz = TZ_NAMES.get(c.most_common(1)[0][0])
    if tz is None and first.get('tz_offset') is not None:       # LightCraft keeps the UTC offset the photo was taken with
        tz = tz_from_offset(first['tz_offset'], first['day'])
    loc = next((i for i in items if i.get('lat') is not None), None)
    pin = None if loc else daypins().get(first['day'])        # a frame's own location always wins over the day pin
    if loc:
        lat, lon, src, note = loc['lat'], loc['lon'], loc['loc_source'], loc.get('loc_note', '')
        acc = 50 if loc['loc_source'] == 'neighbor' else loc.get('positional_accuracy')
    elif pin:
        lat, lon, src, acc = pin['lat'], pin['lon'], 'day_pin', pin.get('accuracy_m')
        note = f"day pin{' ' + repr(pin['label']) if pin.get('label') else ''} (accuracy {acc} m)"
    else:
        lat = lon = acc = None
        src, note = 'none', 'no location'
    return {'observed_on_string': obs_str, 'time_zone': tz, 'lat': lat, 'lon': lon, 'loc_source': src, 'loc_note': note,
            'positional_accuracy': acc, 'camera': first.get('camera')}


NATURE = re.compile(r'flower|plant|leaf|leaves|fungus|fungi|mushroom|fruiting|insect|animal|bird|spider|butterfly|moth|bee|fly|beetle|tree|bark|wing|'
                    r'petal|bud|berry|seed|cap|gill|lichen|moss|reptile|frog|snake|turtle|mammal|caterpillar|larva|cocoon|egg|web|nest|fruit|stem|branch', re.I)
NOT_NATURE = re.compile(r'vehicle|car|truck|motorcycle|building|house|sign|person|people|human|food|plate|dish|drink|bottle|room|furniture|screen|'
                        r'phone|text|poster|crowd|street|road|boat|machine|tool|toy', re.I)


def nonnature(items):
    """Advisory: most photos of the group have a non-organism subject according to the local vision pass."""
    votes = []
    for it in items:
        t = visionlib.flat((VIS.get(it['copy']) or {}).get('tags'))
        s = str(t.get('subject_part') or '').lower()
        if s:
            votes.append(bool(NOT_NATURE.search(s)) and not NATURE.search(s))
    return bool(votes) and sum(votes) / len(votes) >= 0.5


def summary(g, items, dec):
    d = group_default(items)
    s = SUGG.get(str(g)) or {}
    top = (s.get('species') or [None])[0]
    return {'group': g, 'n': len(items), 'day': items[0]['day'], 'status': (dec.get(g) or {}).get('status', 'pending'),
            'loc': d['loc_source'], 'tz': d['time_zone'], 'flags': flags_for(items), 'top': top['common'] or top['taxon'] if top else None,
            'top_share': top['vote_share'] if top else None, 'top_acc': top['measured_accuracy'] if top else None,
            'iconic': (s.get('iconic') or [{}])[0].get('iconic'), 'nonnature': nonnature(items)}


_last = [0.0]
_cache = {}


def taxa_search(q):
    q = q.strip()
    if len(q) < 3:
        return []
    if q in _cache:
        return _cache[q]
    wait = 1.1 - (time.time() - _last[0])
    if wait > 0:
        time.sleep(wait)
    _last[0] = time.time()
    url = 'https://api.inaturalist.org/v1/taxa/autocomplete?' + urllib.parse.urlencode({'q': q, 'per_page': 8})
    try:
        r = json.load(urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': UA}), timeout=20))
        out = [{'id': t['id'], 'name': t['name'], 'common': t.get('preferred_common_name'), 'rank': t.get('rank')} for t in r.get('results', [])]
    except Exception as e:
        out = [{'id': None, 'name': f'lookup failed: {type(e).__name__}', 'common': None, 'rank': None}]
    _cache[q] = out
    return out


def thumb(idx, side):
    it = QUEUE[idx]
    im = Image.open(it['copy'])
    im.draft('RGB', (side * 2, side * 2))
    im = im.convert('RGB')
    im.thumbnail((side, side))
    b = io.BytesIO()
    im.save(b, 'JPEG', quality=82)
    return b.getvalue()


PAGE = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'inat_console.html'), encoding='utf-8').read()


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype='application/json'):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def _host_ok(self):
        return (self.headers.get('Host') or '').split(':')[0] in ('127.0.0.1', 'localhost')

    def do_GET(self):
        if not self._host_ok():
            return self._send(403, {'error': 'host'})
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        if u.path == '/':
            return self._send(200, PAGE, 'text/html; charset=utf-8')
        if u.path == '/api/state':
            dec, done = decisions(), posted()
            rows = [summary(g, items, dec) for g, items in GROUPS.items()]
            for r in rows:
                r['posted'] = r['group'] in done
            c = collections.Counter(r['status'] for r in rows)
            return self._send(200, {'groups': rows, 'counts': c, 'tz_choices': TZ_CHOICES, 'queue_total': len(QUEUE), 'queue': TAG or 'files',
                                    'without_copy': sum(1 for i in QUEUE if not i.get('copy'))})
        if u.path == '/api/days':
            pins = daypins()
            byday = collections.OrderedDict()
            for g, items in GROUPS.items():
                d = items[0]['day']
                row = byday.setdefault(d, {'day': d, 'groups': 0, 'photos': 0, 'own_location': 0, 'from_pin': 0, 'none': 0})
                row['groups'] += 1
                row['photos'] += len(items)
                if any(i.get('lat') is not None for i in items):
                    row['own_location'] += 1
                elif d in pins:
                    row['from_pin'] += 1
                else:
                    row['none'] += 1
            rows = sorted(byday.values(), key=lambda r: r['day'])
            for r in rows:
                p = pins.get(r['day'])
                r['pin'] = {'lat': p['lat'], 'lon': p['lon'], 'accuracy_m': p.get('accuracy_m'), 'label': p.get('label', '')} if p else None
            return self._send(200, {'days': rows, 'default_accuracy_m': DEFAULT_PIN_ACCURACY})
        if u.path == '/api/group':
            g = int(q['id'][0])
            items = GROUPS.get(g)
            if not items:
                return self._send(404, {'error': 'no such group'})
            dec = decisions().get(g)
            d = group_default(items)
            return self._send(200, {'group': g, 'day': items[0]['day'], 'defaults': d, 'decision': dec, 'flags': flags_for(items),
                                    'suggestions': SUGG.get(str(g)),
                                    'photos': [{'idx': i['idx'], 'name': os.path.basename(i['rel']), 'time': i.get('time'), 'w': i['w'], 'h': i['h'],
                                                'release': i.get('release')} for i in sorted(items, key=lambda i: i.get('t') or 0)]})
        if u.path == '/img':
            i, side = int(q['i'][0]), min(int(q.get('s', ['700'])[0]), 1400)
            if not (0 <= i < len(QUEUE)) or not QUEUE[i].get('copy'):
                return self._send(404, {'error': 'no image'})
            return self._send(200, thumb(i, side), 'image/jpeg')
        if u.path == '/api/taxa':
            return self._send(200, taxa_search(q.get('q', [''])[0]))
        if u.path == '/export':
            scope = q.get('scope', ['approved'])[0]
            dec = decisions()
            if scope == 'group':
                try:
                    g = int(q['id'][0])
                except (KeyError, ValueError):
                    return self._send(400, {'error': 'group id needed'})
                groups = {g: GROUPS[g]} if g in GROUPS else {}
            elif scope == 'approved':
                groups = {g: its for g, its in GROUPS.items() if (dec.get(g) or {}).get('status') == 'approved'}
            elif scope == 'all':
                groups = dict(GROUPS)
            else:
                return self._send(400, {'error': 'scope must be group, approved or all'})
            if not groups:
                return self._send(404, {'error': 'nothing to export for this scope (approve observations first, or use scope=all)'})
            try:
                path, n_groups, n_files = inat_export.build_zip(groups, dec, group_default, scope,
                                                                csv_sidecar=q.get('csv', ['0'])[0] == '1')
            except Exception as e:
                return self._send(500, {'error': f'export failed: {type(e).__name__}: {str(e)[:120]}'})
            self.send_response(200)
            self.send_header('Content-Type', 'application/zip')
            self.send_header('Content-Length', str(os.path.getsize(path)))
            self.send_header('Content-Disposition', f'attachment; filename="{os.path.basename(path)}"')
            self.send_header('X-Export-Observations', str(n_groups))
            self.send_header('X-Export-Photos', str(n_files))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            with open(path, 'rb') as f:
                shutil.copyfileobj(f, self.wfile, 1 << 20)
            return None
        return self._send(404, {'error': 'not found'})

    def do_POST(self):
        if not self._host_ok() or self.headers.get('X-Console') != '1':
            return self._send(403, {'error': 'forbidden'})
        if self.path == '/api/daypin':
            n = int(self.headers.get('Content-Length') or 0)
            d = json.loads(self.rfile.read(n) or b'{}')
            day = str(d.get('day') or '')
            if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', day) or not any(it[0]['day'] == day for it in GROUPS.values()):
                return self._send(400, {'error': 'unknown day'})
            rec = {'day': day, 'ts': datetime.now().isoformat(timespec='seconds')}
            if d.get('clear'):
                rec['clear'] = True
            else:
                ll = parse_latlon(d.get('latlon')) if d.get('latlon') else parse_latlon(f"{d.get('lat')} {d.get('lon')}")
                if not ll:
                    return self._send(400, {'error': 'give a location as "lat, lon" (for example 39.9305, -77.4406)'})
                try:
                    acc = int(d.get('accuracy_m') or DEFAULT_PIN_ACCURACY)
                except (TypeError, ValueError):
                    return self._send(400, {'error': 'accuracy must be a whole number of metres'})
                if not 1 <= acc <= 100000:
                    return self._send(400, {'error': 'accuracy must be between 1 and 100000 metres'})
                rec.update(lat=ll[0], lon=ll[1], accuracy_m=acc, label=str(d.get('label') or '')[:80])
            with LOCK:
                with open(PINS_PATH, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(rec) + '\n')
            return self._send(200, {'ok': True})
        if self.path != '/api/decide':
            return self._send(404, {'error': 'not found'})
        n = int(self.headers.get('Content-Length') or 0)
        d = json.loads(self.rfile.read(n) or b'{}')
        g = d.get('group')
        if g not in GROUPS:
            return self._send(400, {'error': 'unknown group'})
        status = d.get('status')
        if status not in ('approved', 'skipped', 'hold', 'pending'):
            return self._send(400, {'error': 'bad status'})
        if status == 'approved':
            miss = [k for k in ('observed_on_string', 'time_zone', 'lat', 'lon') if d.get(k) in (None, '')]
            if miss:
                return self._send(400, {'error': 'approve needs: ' + ', '.join(miss)})
            if d.get('geoprivacy') not in ('open', 'obscured', 'private'):
                return self._send(400, {'error': 'geoprivacy must be open, obscured or private'})
            if d.get('time_zone') not in TZ_CHOICES:
                return self._send(400, {'error': 'pick a time zone from the list'})
        rec = {k: d.get(k) for k in ('group', 'status', 'taxon_id', 'taxon_name', 'species_guess', 'description', 'tags', 'geoprivacy',
                                     'captive', 'observed_on_string', 'time_zone', 'lat', 'lon', 'positional_accuracy', 'photo_order')}
        rec['ts'] = datetime.now().isoformat(timespec='seconds')
        with LOCK:
            with open(LEDGER, 'a', encoding='utf-8') as f:
                f.write(json.dumps(rec) + '\n')
        return self._send(200, {'ok': True})


def main():
    print(f'console on http://127.0.0.1:{PORT}/  | groups {len(GROUPS)} | photos with copies {sum(len(v) for v in GROUPS.values())} '
          f'| suggestions {len(SUGG)} | screened copies {len(VIS)}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', PORT), H).serve_forever()


if __name__ == '__main__':
    main()
