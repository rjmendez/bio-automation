"""Re-describe photos whose vision description contradicts the iNaturalist ID (e.g. a wood ear fungus called "whole animal").

The first vision pass is ID-blind on purpose (release screening must not be swayed). Its descriptive fields can still contradict the
ID, which then leaks into keywords. This pass:
  1. finds files whose subject_part / notes belong to a different organism class than the observation's taxon;
  2. asks the same local vision model again for the descriptive fields only, telling it the photographer's ID;
  3. also asks whether the image plausibly shows that organism; "no"/"unsure" is recorded for review and never overrides the ID.
Release flags are untouched. Output: vision_redescribed.json (separate file: vision_files.json is checkpointed by other jobs).
Usage: redescribe_vision.py count | run
"""
import collections
import csv
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vision_files as vf  # http, pick_model, small

ROOT = vf.ROOT
OUT = os.path.join(ROOT, 'vision_redescribed.json')
EXPORT = r'F:\Bio-export\adobe-stock'
DIRS = vf.DIRS

ANIMAL = re.compile(r'\b(animal|creature|insect|bug|spider|arachnid|bird|mammal|frog|toad|lizard|snake|rodent|beetle|moth|butterfly|fly|bee|wasp|ant|'
                    r'caterpillar|larva|slug|snail|worm|fish|reptile|amphibian|mouse|squirrel|rabbit)s?\b', re.I)
PLANT = re.compile(r'\b(flower|petal|blossom|leaf|leaves|shrub|tree|sapling|seedling|bud|berry|fern|vine)s?\b', re.I)
FUNGUS = re.compile(r'\b(fruiting bod(y|ies)|mushroom|fungus|fungi|gills?|bracket|polypore|mould|mold|puffball|lichen)s?\b', re.I)
CLASS_OF = {'Fungi': 'fungus', 'Plantae': 'plant', 'Chromista': 'plant', 'Protozoa': 'other'}
PLAIN = {'Fungi': 'fungus', 'Plantae': 'plant', 'Insecta': 'insect', 'Arachnida': 'spider or other arachnid', 'Aves': 'bird',
         'Mammalia': 'mammal', 'Reptilia': 'reptile', 'Amphibia': 'amphibian', 'Mollusca': 'mollusc', 'Actinopterygii': 'fish',
         'Animalia': 'animal', 'Chromista': 'alga or kelp', 'Protozoa': 'protozoan'}


def flat(tags):
    out = {}
    for k, v in (tags or {}).items():
        if isinstance(v, dict):
            out.update(flat(v))
        else:
            out[k.lower()] = v
    return out


def own_class(iconic):
    return CLASS_OF.get(iconic, 'animal')


def conflict(t, iconic):
    """Return the class the description claims when it differs from the taxon's class, else None."""
    text = ' '.join(str(t.get(f) or '') for f in ('subject_part', 'notes'))
    mine = own_class(iconic)
    hits = {c for c, rx in (('animal', ANIMAL), ('plant', PLANT), ('fungus', FUNGUS)) if rx.search(text)}
    if hits and mine not in hits and mine != 'other':
        return sorted(hits)[0]
    return None


def file_map():
    """vision key (basename) -> (path, obs ids)"""
    m = {}
    for mf in ('manifest.csv', 'manifest_nas.csv'):
        p = os.path.join(EXPORT, mf)
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p, encoding='utf-8')):
            # key like the vision pass and the aggregator: <day>_<source basename>; output_jpeg is empty for skipped rows
            src = r['source_jpeg'].replace(chr(92), '/')
            day = next((c for c in src.split('/') if re.fullmatch(r'\d{4}-\d{2}-\d{2}', c)), '')
            name = f'{day}_{os.path.basename(src)}'
            path = next((os.path.join(d, name) for d in DIRS if os.path.exists(os.path.join(d, name))), None)
            if path:
                m[name] = (path, [x for x in r['obs_ids'].split(';') if x])
    return m


def build():
    obs = {str(x['id']): x for x in json.load(open(os.path.join(ROOT, 'inat_observations.json'), encoding='utf-8'))['observations']}
    files = json.load(open(vf.OUT, encoding='utf-8'))
    fm = file_map()
    todo, by_cls, no_taxon = [], collections.Counter(), 0
    for name, e in files.items():
        t = flat(e.get('tags'))
        if not t or name not in fm:
            continue
        path, oids = fm[name]
        tax = next((obs[o]['taxon'] for o in oids if o in obs and obs[o].get('taxon')), None)
        if not tax:
            no_taxon += 1
            continue
        c = conflict(t, tax.get('iconic'))
        if c:
            by_cls[(tax.get('iconic'), c)] += 1
            todo.append((name, path, tax))
    return todo, by_cls, no_taxon, len(files)


def prompt(tax):
    group = PLAIN.get(tax.get('iconic'), (tax.get('iconic') or 'organism').lower())
    ident = tax.get('common') or tax.get('name')
    if tax.get('common') and tax.get('name'):
        ident += f" ({tax['name']})"
    return (
        f'The photographer identified the subject of this nature photograph as: {ident}, a {group}. '
        'Describe only what is visible, consistent with that identification. Answer with JSON only, with exactly these keys: '
        '"subject_part" (the part or form shown, such as fruiting body, cap, gills, cluster, bracket, flower, leaf, whole animal, wing), '
        '"substrate" (e.g. wood, bark, leaf litter, soil, moss, grass, hand), '
        '"habitat" (e.g. forest floor, meadow, garden, woodland, wetland), "colors" (up to 3 main colors of the subject), '
        '"viewpoint" (one of: close-up, macro, high-angle view, low-angle view, eye-level view, wide shot), '
        '"light" (one of: daylight, shade, backlit, overcast, flash, low light), "background" (short phrase), '
        '"adjectives" (up to 4 plain visual adjectives), '
        '"caption" (one plain sentence of at most 14 words saying what is shown and where; do not name the species), '
        f'"id_plausible" ("yes", "no" or "unsure": does the image plausibly show a {group} like the stated identification?). '
        'If a hand or person is visible, say so in the caption.')


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'count'
    todo, by_cls, no_taxon, n = build()
    print(f'files with vision tags {n} | description contradicts the ID class: {len(todo)} | no taxon: {no_taxon}', flush=True)
    for (ic, c), k in by_cls.most_common():
        print(f'   taxon {ic or "?"} described as {c}: {k}')
    if mode != 'run':
        return
    done = json.load(open(OUT, encoding='utf-8')) if os.path.exists(OUT) else {}
    todo = [x for x in todo if x[0] not in done or 'error' in done[x[0]]]
    model = vf.pick_model()
    print('model:', model, '| to do', len(todo), flush=True)
    gpu = {'main_gpu': 1}
    t0 = time.time()
    for i, (name, path, tax) in enumerate(todo, 1):
        try:
            body = {'model': model, 'stream': False, 'format': 'json', 'think': False, 'keep_alive': '30m',
                    'options': {'temperature': 0, **gpu},
                    'messages': [{'role': 'user', 'content': prompt(tax), 'images': [vf.small(path)]}]}
            try:
                r = vf.http('/api/chat', body)
            except Exception:
                body.pop('think', None)
                r = vf.http('/api/chat', body)
            done[name] = {'tags': json.loads(r['message']['content']), 'taxon': tax.get('name'), 'iconic': tax.get('iconic'), 'model': model}
        except Exception as e:
            done[name] = {'error': f'{type(e).__name__}: {str(e)[:100]}'}
            if gpu and i < 5:
                gpu = {}
        if i % 20 == 0 or i == len(todo):
            json.dump(done, open(OUT, 'w'))
            el = time.time() - t0
            print(f'{i}/{len(todo)} {el / i:.1f}s per file eta {(len(todo) - i) * el / i / 60:.0f} min', flush=True)
    json.dump(done, open(OUT, 'w'))
    c = collections.Counter((v.get('tags') or {}).get('id_plausible') for v in done.values())
    print('done', len(done), '| id_plausible:', dict(c), flush=True)


if __name__ == '__main__':
    main()
