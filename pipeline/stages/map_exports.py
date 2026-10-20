"""Dry run: map each Adobe-preview observation to the existing JPEG export(s) of its NEF frames in F:\\Bio. Writes nothing but a report."""
import csv, collections, json, os

ROOT = r'F:\bio-automation-data'
BS = chr(92)
inv = [r['path'].replace(BS, '/') for r in csv.DictReader(open(os.path.join(ROOT, 'f_bio_inventory.csv'), encoding='utf-8-sig'))]
jpg_by = collections.defaultdict(list)     # (day folder, stem) -> [relative paths]
raw_by = collections.defaultdict(list)
for p in inv:
    parts = p.split('/')
    if len(parts) < 3:
        continue
    day = '/'.join(parts[:2])               # 2025/2025-06-03
    stem, ext = os.path.splitext(parts[-1])
    key = (day, stem.lower())
    if ext.lower() == '.jpg':
        jpg_by[key].append(p)
    elif ext.lower() in ('.nef', '.dng'):
        raw_by[key].append(p)

prev = json.load(open(os.path.join(ROOT, 'adobe_preview.json'), encoding='utf-8'))
inat = {o['id']: o for o in json.load(open(os.path.join(ROOT, 'inat_observations.json'), encoding='utf-8'))['observations']}
recs = {}          # jpeg rel path -> record
missing = []       # (obs, frame) with no jpeg
no_raw = []
for r in prev:
    o = inat[int(r['obs_id'])]
    year = (o.get('observed_on') or '')[:4]
    day = f"{year}/{o.get('observed_on')}"
    for fn in [x for x in r['frames'].split(';') if x]:
        stem = os.path.splitext(fn)[0].lower()
        cand_keys = [(day, stem), (day, stem.split(' (')[0])]
        jp = next((jpg_by[k] for k in cand_keys if jpg_by.get(k)), None)
        rw = next((raw_by[k] for k in cand_keys if raw_by.get(k)), None)
        if not rw:
            no_raw.append((r['obs_id'], fn))
        if not jp:
            missing.append((r['obs_id'], fn))
            continue
        for path in jp:
            rec = recs.setdefault(path, {'jpeg': path, 'raw': rw[0] if rw else '', 'obs': []})
            rec['obs'].append(r['obs_id'])
json.dump(list(recs.values()), open(os.path.join(ROOT, 'export_map.json'), 'w'), indent=1)
print(f'observations: {len(prev)} | frame links: {sum(r["n_frames"] for r in prev)}')
print(f'distinct existing JPEG exports found: {len(recs)}')
print(f'frame links with NO jpeg export: {len(missing)} | frames whose raw file was not found in F:/Bio by name+day: {len(no_raw)}')
multi = [r for r in recs.values() if len(set(r['obs'])) > 1]
print(f'JPEGs shared by more than one observation (keywords will be merged): {len(multi)}')
sub = collections.Counter(p.split('/')[2] if len(p.split('/')) > 3 else '(day folder)' for p in recs)
print('JPEG location (subfolder):', sub.most_common(6))
print('examples missing:', missing[:5])
