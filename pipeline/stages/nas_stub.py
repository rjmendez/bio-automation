"""Convert nas_matches.json into the formats build_preview.py / embed_meta.py already consume."""
import collections, json, os

ROOT = r'F:\bio-automation-data'
nm = json.load(open(os.path.join(ROOT, 'nas_matches.json'), encoding='utf-8'))['matches']
inat = {o['id']: o for o in json.load(open(os.path.join(ROOT, 'inat_observations.json'), encoding='utf-8'))['observations']}

by_obs = collections.defaultdict(list)
by_jpeg = {}
for m in nm:
    by_obs[m['obs']].append(m)
    j = by_jpeg.setdefault(m['jpeg'], {'jpeg': m['jpeg'], 'raw': m['raw'] or '', 'obs': []})
    if m['obs'] not in j['obs']:
        j['obs'].append(m['obs'])

stub = []
for oid, ms in by_obs.items():
    o = inat[oid]
    t = o.get('time') or (o.get('observed_on') + 'T12:00:00-04:00')
    stub.append({'obs': oid, 'time': t, 'n_candidates': len(ms), 'obscured': o.get('obscured'),
                 'photos': [{'photo': m['photo'], 'best': {'corr': m['corr'], 'frame_id': 0,
                                                            'file': os.path.basename(m['raw'] or m['jpeg']), 'captured': t}} for m in ms]})
json.dump(stub, open(os.path.join(ROOT, 'nas_matches_stub.json'), 'w'))
json.dump(list(by_jpeg.values()), open(os.path.join(ROOT, 'export_map_nas.json'), 'w'), indent=1)
no_raw = sum(1 for j in by_jpeg.values() if not j['raw'])
print(f'observations {len(stub)} | distinct JPEG exports {len(by_jpeg)} | without a paired NEF/DNG on the NAS: {no_raw}')
