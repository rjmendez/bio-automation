"""Turn per-file vision results (vision_files.json) into per-observation tags (vision_tags.json) for build_preview.py.
Descriptive fields: most common value / union by frequency across the observation's files.
Release fields are OR-ed (any positive wins) and mapped onto people_visible / text_or_logo_visible."""
import collections, csv, json, os, re
import visionlib

ROOT = r'F:\bio-automation-data'
vf = json.load(open(os.path.join(ROOT, 'vision_files.json'), encoding='utf-8'))
man = []
for mf in ('manifest.csv', 'manifest_nas.csv'):          # local-source copies and NAS-source copies
    p_ = os.path.join(r'F:\Bio-export\adobe-stock', mf)
    if os.path.exists(p_):
        man += list(csv.DictReader(open(p_, encoding='utf-8')))


def truthy(x):
    return x is True or (isinstance(x, str) and x.strip().lower() in ('true', 'yes'))


def as_list(v):
    if v is None:
        return []
    if isinstance(v, str):
        v = re.split(r'[,;/]| and ', v)
    return [x.strip().lower() for x in v if isinstance(x, str) and x.strip()]


per_obs = collections.defaultdict(list)
for r in man:
    name = os.path.basename(r['source_jpeg'])
    day = next((c for c in r['source_jpeg'].replace('\\', '/').split('/') if re.fullmatch(r'\d{4}-\d{2}-\d{2}', c)), '')
    key = f'{day}_{name}'
    tags = visionlib.tags_for(key, vf.get(key))      # flat schema; ID-aware re-description overlaid on descriptive fields
    if not tags:
        continue
    for oid in [x for x in r['obs_ids'].split(';') if x]:
        per_obs[oid].append((key, tags))

out = {}
for oid, items in per_obs.items():
    agg = {}
    for f in ('subject_part', 'substrate', 'habitat', 'viewpoint', 'light', 'background'):
        c = collections.Counter(x for _, t in items for x in as_list(t.get(f)))
        if c:
            agg[f] = [k for k, _ in c.most_common(2)]
    for f, n in (('colors', 3), ('adjectives', 4)):
        c = collections.Counter(x for _, t in items for x in as_list(t.get(f)))
        agg[f] = [k for k, _ in c.most_common(n)]
    fls = [visionlib.release_flags(vf[k].get('tags'), k) for k, _ in items]   # nested schema + owner hands rule
    agg['people_visible'] = any(f in fl for fl in fls for f in ('person_visible', 'face_visible', 'identifiable_person'))
    agg['text_or_logo_visible'] = any(f in fl for fl in fls for f in ('text_or_logo_visible', 'brand_or_product_visible', 'property_identifiable'))
    out[oid] = {'tags': agg, 'model': items[0][1] and 'local vision', 'files': [k for k, _ in items]}
json.dump(out, open(os.path.join(ROOT, 'vision_tags.json'), 'w'), indent=1)
n_people = sum(1 for v in out.values() if v['tags']['people_visible'])
n_text = sum(1 for v in out.values() if v['tags']['text_or_logo_visible'])
print(f'observations with vision tags: {len(out)} | people/face flagged: {n_people} | text/logo/brand/property flagged: {n_text}')
files_flag = {k: visionlib.release_flags(v.get('tags'), k) for k, v in vf.items() if v.get('tags')}
c = collections.Counter(f for fl in files_flag.values() for f in fl)
print('file-level positives by field:', dict(c))
print(f'files screened: {len(files_flag)} | files with any release flag: {sum(1 for fl in files_flag.values() if fl)}')
