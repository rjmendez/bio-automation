"""Fetch taxonomy (ancestors with English common names) for every taxon in the matched observations.
Public iNaturalist API, no auth, 30 ids per request, ~1 request/s. Writes taxa.json (cached, resumable)."""
import json, os, sys, time, urllib.request

ROOT = r'F:\bio-automation-data'
OUT = os.path.join(ROOT, 'taxa.json')
files = sys.argv[1:] or [os.path.join(ROOT, 'matches_stage1c.json')]
inat = {o['id']: o for o in json.load(open(os.path.join(ROOT, 'inat_observations.json'), encoding='utf-8'))['observations']}
need = set()
for f in files:
    for r in json.load(open(f, encoding='utf-8')):
        t = (inat.get(r['obs'], {}).get('taxon') or {})
        if t.get('id'):
            need.add(t['id'])
have = json.load(open(OUT, encoding='utf-8')) if os.path.exists(OUT) else {}
todo = sorted(i for i in need if str(i) not in have)
print(f'taxa needed {len(need)}, cached {len(need) - len(todo)}, to fetch {len(todo)}', flush=True)
for i in range(0, len(todo), 30):
    chunk = todo[i:i + 30]
    url = 'https://api.inaturalist.org/v1/taxa/%s?locale=en' % ','.join(map(str, chunk))
    req = urllib.request.Request(url, headers={'User-Agent': 'bio-automation/0.1 (personal; read-only)'})
    for t in json.load(urllib.request.urlopen(req, timeout=60)).get('results', []):
        have[str(t['id'])] = {'name': t.get('name'), 'rank': t.get('rank'), 'common': t.get('preferred_common_name'),
                              'iconic': t.get('iconic_taxon_name'),
                              'ancestors': [{'name': a.get('name'), 'rank': a.get('rank'), 'common': a.get('preferred_common_name')}
                                            for a in t.get('ancestors', [])]}
    json.dump(have, open(OUT, 'w'))
    print(f'  {min(i + 30, len(todo))}/{len(todo)}', flush=True)
    time.sleep(1.1)
print('done', len(have), flush=True)
