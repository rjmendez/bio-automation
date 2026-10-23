"""End-to-end test of the per-day location pin against a throwaway ledger (the real inat_daypins*.jsonl is not touched)."""
import http.client
import io
import json
import os
import sys
import tempfile
import threading
import zipfile
from http.server import ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))
from bio_automation import settings

SCRATCH = str(settings.scratch_dir())

os.environ['BIO_QUEUE_TAG'] = '_lc'
os.environ['BIO_EXPORT_DIR'] = os.path.join(SCRATCH, 'daypin_export_test')
sys.path.insert(0, r'F:\bio-automation-data')
import piexif

import inat_console as C

tmp = os.path.join(SCRATCH, 'daypin_test.jsonl')
if os.path.exists(tmp):
    os.remove(tmp)
C.PINS_PATH = tmp
C._pins['mtime'] = None
srv = ThreadingHTTPServer(('127.0.0.1', 0), C.H)
port = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()


def call(method, path, body=None, hdr=None):
    c = http.client.HTTPConnection('127.0.0.1', port, timeout=60)
    h = {'Host': '127.0.0.1', 'X-Console': '1', 'Content-Type': 'application/json'}
    h.update(hdr or {})
    c.request(method, path, json.dumps(body) if body is not None else None, h)
    r = c.getresponse()
    data = r.read()
    return r.status, (json.loads(data) if r.getheader('Content-Type', '').startswith('application/json') else data)


# a day that has frames without a location
st, days = call('GET', '/api/days')
target = next(d for d in days['days'] if d['none'] > 0)
day = target['day']
print('day', day, '| groups', target['groups'], '| none', target['none'], '| own', target['own_location'], '| default accuracy', days['default_accuracy_m'])

print('bad input   ->', call('POST', '/api/daypin', {'day': day, 'latlon': 'nowhere'})[1])
print('0,0 refused ->', call('POST', '/api/daypin', {'day': day, 'latlon': '0, 0'})[1])
print('unknown day ->', call('POST', '/api/daypin', {'day': '1999-01-01', 'latlon': '39.9, -77.4'})[1])
print('no header   ->', call('POST', '/api/daypin', {'day': day, 'latlon': '39.9, -77.4'}, {'X-Console': '0'})[0])

st, r = call('POST', '/api/daypin', {'day': day, 'latlon': '39.9305, -77.4406', 'accuracy_m': 750, 'label': 'test place'})
print('save pin    ->', st, r)
st, days2 = call('GET', '/api/days')
row = next(d for d in days2['days'] if d['day'] == day)
print('after pin: none', row['none'], '| from_pin', row['from_pin'], '| own', row['own_location'], '| pin accuracy', row['pin']['accuracy_m'])
assert row['none'] == 0 and row['from_pin'] == target['none'] and row['own_location'] == target['own_location']

st, state = call('GET', '/api/state')
pinned_groups = [g for g in state['groups'] if g['day'] == day and g['loc'] == 'day_pin']
print('groups shown as day_pin:', len(pinned_groups))
g = pinned_groups[0]['group']
st, det = call('GET', f'/api/group?id={g}')
print('card defaults:', det['defaults']['loc_source'], '|', det['defaults']['loc_note'], '| accuracy', det['defaults']['positional_accuracy'], '| day', det['day'])

# a group that has its own location must keep it
own = next((x for x in state['groups'] if x['day'] == day and x['loc'] not in ('none', 'day_pin')), None)
print('own-location group keeps its source:', own['loc'] if own else '(none on this day)')

st, z = call('GET', f'/export?scope=group&id={g}')
zf = zipfile.ZipFile(io.BytesIO(z))
ex = piexif.load(zf.read(zf.namelist()[0]))
gps = ex['GPS']
print('export: GPS', bool(gps.get(piexif.GPSIFD.GPSLatitude)), '| accuracy', gps.get(piexif.GPSIFD.GPSHPositioningError), '| flat', all('/' not in n for n in zf.namelist()))

print('clear       ->', call('POST', '/api/daypin', {'day': day, 'clear': True})[1])
st, days3 = call('GET', '/api/days')
row3 = next(d for d in days3['days'] if d['day'] == day)
print('after clear: none', row3['none'], '| pin', row3['pin'])
os.remove(tmp)
srv.shutdown()
