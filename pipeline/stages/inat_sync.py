import os, re, json, math, urllib.request, urllib.parse, http.cookiejar, sys
cfg = {}
for line in open(os.path.expanduser('~/.inat')):
    m = re.match(r'\s*([A-Z_]+)\s*=\s*"?(.*?)"?\s*$', line)
    if m: cfg[m.group(1)] = m.group(2)
cj = http.cookiejar.CookieJar()
op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
op.addheaders = [('User-Agent', 'bio-automation/0.1 (personal library matching; read-only)')]
BASE = 'https://www.inaturalist.org'
html = op.open(BASE + '/users/sign_in').read().decode()
tok = re.search(r'name="csrf-token" content="([^"]+)"', html) or re.search(r'name="authenticity_token" value="([^"]+)"', html)
if not tok: sys.exit('no csrf token found')
data = urllib.parse.urlencode({'authenticity_token': tok.group(1), 'user[email]': cfg['USERNAME'],
                               'user[password]': cfg['PASSWORD'], 'user[remember_me]': '0', 'commit': 'Log in'}).encode()
try:
    op.open(BASE + '/session', data)
except urllib.error.HTTPError as e:
    print('login http', e.code)
req = urllib.request.Request(BASE + '/users/api_token', headers={'Accept': 'application/json'})
try:
    jwt = json.load(op.open(req))['api_token']
except Exception as e:
    sys.exit('api_token failed: %s' % type(e).__name__)
print('login ok; token length', len(jwt))   # token itself never printed

def api(path):
    r = urllib.request.Request('https://api.inaturalist.org/v1' + path,
                               headers={'Authorization': jwt, 'User-Agent': 'bio-automation/0.1'})
    return json.load(urllib.request.urlopen(r))

import time
me = api('/users/me')['results'][0]; L = me['login']
print('observations on account:', me['observations_count'])
out=[]; last=0; pages=0
while True:
    r = api('/observations?user_id=%s&per_page=200&order_by=id&order=asc&id_above=%d&fields=all' % (L, last))
    res = r['results']
    if not res: break
    for o in res:
        t = o.get('taxon') or {}
        out.append({'id':o['id'],'time':o.get('time_observed_at'),'observed_on':o.get('observed_on'),'tz':o.get('observed_time_zone') or o.get('time_zone'),
          'created_at':o.get('created_at'),'uri':o.get('uri'),'quality':o.get('quality_grade'),
          'geoprivacy':o.get('geoprivacy'),'taxon_geoprivacy':o.get('taxon_geoprivacy'),'obscured':o.get('obscured'),
          'taxon':{'id':t.get('id'),'name':t.get('name'),'rank':t.get('rank'),'common':t.get('preferred_common_name'),'iconic':t.get('iconic_taxon_name')} if t else None,
          'public_location':o.get('location'),'private_location':o.get('private_location'),'place_guess':o.get('place_guess'),
          'positional_accuracy':o.get('positional_accuracy'),
          'photos':[{'id':p['id'],'url':p['url'],'dims':p.get('original_dimensions'),'license':p.get('license_code')} for p in o.get('photos',[])]})
    last = res[-1]['id']; pages += 1
    if pages % 10 == 0: print('pages', pages, 'obs', len(out), flush=True)
    time.sleep(1.1)
    if len(res) < 200: break
json.dump({'login':L,'count':len(out),'observations':out}, open('' + os.environ.get('BIO_DATA_WSL', '/mnt/f/bio-automation-data') + '/inat_observations.json','w'))
print('saved', len(out), 'observations')
