"""Post the observations you approved in the console to iNaturalist (one observation, then its photos). Dry run unless --live.

Guard rails (iNaturalist's machine-generated-content rule needs a human decision on every observation):
  * only decisions with status "approved" made in the console are sent; nothing is derived, filled in or retried automatically;
  * a group already in inat_posted.jsonl is never sent again; a failure between creating the observation and attaching photos is
    recorded as "partial" and left for you to look at (no automatic retry, so no duplicates);
  * <= 1 request per second, identifying User-Agent, your own account (JWT in env INAT_JWT, valid 24 h), default --limit 10.
Usage:  inat_post.py                      dry run: shows what would be sent (no coordinates printed)
        INAT_JWT=... inat_post.py --live --limit 10
Get the JWT while logged in: https://www.inaturalist.org/users/api_token  (do not paste it anywhere else).
"""
import argparse
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime

ROOT = os.environ.get('BIO_DATA', r'F:\bio-automation-data')
API = 'https://api.inaturalist.org/v1'
UA = 'bio-automation-poster/0.1 (personal uploader; owner rjmendez)'
LEDGER = os.path.join(ROOT, 'inat_decisions.jsonl')
POSTED = os.path.join(ROOT, 'inat_posted.jsonl')
_last = [0.0]


def pace():
    w = 1.1 - (time.time() - _last[0])
    if w > 0:
        time.sleep(w)
    _last[0] = time.time()


def latest_decisions():
    cur = {}
    if not os.path.exists(LEDGER):
        return cur
    for line in open(LEDGER, encoding='utf-8'):
        try:
            r = json.loads(line)
            cur[r['group']] = r
        except Exception:
            pass
    return cur


def posted_groups():
    s = {}
    if os.path.exists(POSTED):
        for line in open(POSTED, encoding='utf-8'):
            try:
                r = json.loads(line)
                if r.get('status') in ('posted', 'partial'):     # a plain failure (nothing created) may be retried
                    s[r['group']] = r
            except Exception:
                pass
    return s


def request(url, jwt, data=None, headers=None, method=None):
    h = {'User-Agent': UA, 'Authorization': jwt, 'Accept': 'application/json'}
    h.update(headers or {})
    pace()
    req = urllib.request.Request(url, data=data, headers=h, method=method or ('POST' if data is not None else 'GET'))
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def multipart(fields, fname, fbytes):
    b = uuid.uuid4().hex
    parts = []
    for k, v in fields.items():
        parts.append(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    ctype = mimetypes.guess_type(fname)[0] or 'image/jpeg'
    parts.append(f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="{fname}"\r\nContent-Type: {ctype}\r\n\r\n'.encode())
    parts.append(fbytes)
    parts.append(f'\r\n--{b}--\r\n'.encode())
    return b''.join(parts), f'multipart/form-data; boundary={b}'


def payload(d):
    o = {'observed_on_string': d['observed_on_string'], 'time_zone': d['time_zone'], 'latitude': d['lat'], 'longitude': d['lon'],
         'geoprivacy': d['geoprivacy'], 'captive_flag': bool(d.get('captive'))}
    if d.get('taxon_id'):
        o['taxon_id'] = d['taxon_id']
    if d.get('species_guess'):
        o['species_guess'] = d['species_guess']
    if d.get('description'):
        o['description'] = d['description']
    if d.get('tags'):
        o['tag_list'] = d['tags']
    if d.get('positional_accuracy'):
        o['positional_accuracy'] = d['positional_accuracy']
    return {'observation': o}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--live', action='store_true')
    ap.add_argument('--limit', type=int, default=10)
    a = ap.parse_args()
    q = json.load(open(os.path.join(ROOT, 'inat_queue.json')))
    dec, done = latest_decisions(), posted_groups()
    todo = [d for g, d in sorted(dec.items()) if d.get('status') == 'approved' and g not in done]
    print(f'approved in the console: {sum(1 for d in dec.values() if d.get("status") == "approved")} | already posted/partial: {len(done)} | to send: {len(todo)}')
    if not todo:
        return
    batch = todo[:a.limit]
    if not a.live:
        for d in batch[:5]:
            p = payload(d)['observation']
            print(f'  group {d["group"]}: {len(d.get("photo_order") or [])} photo(s), taxon {d.get("taxon_name") or "(unidentified)"}, '
                  f'{p["observed_on_string"]} {p["time_zone"]}, geoprivacy {p["geoprivacy"]}, accuracy {p.get("positional_accuracy")}')
        print(f'DRY RUN: nothing sent. {len(batch)} would be sent with --live (limit {a.limit}), about {len(batch) * 4} requests at 1/s.')
        return
    jwt = os.environ.get('INAT_JWT')
    if not jwt:
        sys.exit('INAT_JWT is not set (get it from https://www.inaturalist.org/users/api_token while logged in)')
    for d in batch:
        g = d['group']
        rec = {'group': g, 'ts': datetime.now().isoformat(timespec='seconds')}
        try:
            r = request(API + '/observations', jwt, json.dumps(payload(d)).encode(), {'Content-Type': 'application/json'})
            res = r['results'][0] if isinstance(r.get('results'), list) else r
            oid = res.get('id')
            rec['observation_id'] = oid
            rec['uuid'] = res.get('uuid')
            n = 0
            for idx in d['photo_order']:
                path = q[idx]['copy']
                body, ctype = multipart({'observation_photo[observation_id]': oid}, os.path.basename(path), open(path, 'rb').read())
                request(API + '/observation_photos', jwt, body, {'Content-Type': ctype})
                n += 1
            rec['photos'] = n
            rec['status'] = 'posted'
        except Exception as e:
            rec['status'] = 'partial' if rec.get('observation_id') else 'failed'
            rec['error'] = f'{type(e).__name__}: {str(e)[:160]}'
        with open(POSTED, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec) + '\n')
        print(f'group {g}: {rec["status"]}' + (f' observation {rec.get("observation_id")}' if rec.get('observation_id') else '') +
              (f' ({rec["error"]})' if rec.get('error') else ''))
        if rec['status'] != 'posted':
            print('stopping after the first problem; look at it before sending more.')
            break


if __name__ == '__main__':
    main()
