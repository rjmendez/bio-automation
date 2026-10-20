"""Build the Adobe Stock title/keyword PREVIEW (nothing is written to the library).

Input : matches file (confident pixel matches only, corr >= 0.95), taxa.json, vision_tags.json, inat export.
Output: adobe_preview.csv + adobe_preview.json
Rules (Adobe contributor page): <=49 keywords, each once, one language, ID/important terms first, title <70 chars,
plain language, no brands/names/IP; general+specific taxonomy; coarse location only; no coordinates ever.
"""
import csv, collections, json, os, re, sys

ROOT = r'F:\bio-automation-data'
MATCHES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'matches_stage1c.json')
OUTP = os.environ.get('BIO_PREVIEW_PREFIX', 'adobe_preview')
ALLOW_JPEG = os.environ.get('BIO_ALLOW_JPEG') == '1'
TH = 0.95
MAXKW, MAXTITLE = 49, 69

res = json.load(open(MATCHES, encoding='utf-8'))
inat = {o['id']: o for o in json.load(open(os.path.join(ROOT, 'inat_observations.json'), encoding='utf-8'))['observations']}
taxa = json.load(open(os.path.join(ROOT, 'taxa.json'), encoding='utf-8'))
vision = json.load(open(os.path.join(ROOT, 'vision_tags.json'), encoding='utf-8')) if os.path.exists(os.path.join(ROOT, 'vision_tags.json')) else {}

STOP_ANC = {'life', 'eukaryotes', 'opisthokonts', 'dikarya', 'bilateria', 'protostomes', 'deuterostomes', 'protostomia',
            'tracheophytes', 'angiosperms', 'ascomycetes', 'basidiomycetes', 'hexapods', 'mandibulates', 'neoptera',
            'holometabola', 'euarthropoda', 'panarthropoda', 'ecdysozoa', 'chelicerates', 'metazoa'}
ICONIC = {'Fungi': ['fungus'], 'Plantae': ['plant'], 'Insecta': ['insect', 'arthropod'], 'Arachnida': ['arachnid', 'arthropod'],
          'Aves': ['bird'], 'Mammalia': ['mammal'], 'Reptilia': ['reptile'], 'Amphibia': ['amphibian'], 'Mollusca': ['mollusk'],
          'Protozoa': ['slime mold'], 'Animalia': ['animal']}
SEASON = {12: 'winter', 1: 'winter', 2: 'winter', 3: 'spring', 4: 'spring', 5: 'spring', 6: 'summer', 7: 'summer', 8: 'summer',
          9: 'autumn', 10: 'autumn', 11: 'autumn'}
JARGON = {'vascular plants', 'basidiomycete fungi', 'dicots', 'monocots', 'eudicots', 'magnoliids', 'rosids', 'asterids', 'ferns and allies', 'seed plants',
          'cnidarians', 'sac fungi', 'club fungi', 'true bugs', 'true fungi'}
US_STATES = {'AL': 'alabama', 'AK': 'alaska', 'AZ': 'arizona', 'AR': 'arkansas', 'CA': 'california', 'CO': 'colorado',
             'CT': 'connecticut', 'DE': 'delaware', 'FL': 'florida', 'GA': 'georgia', 'HI': 'hawaii', 'ID': 'idaho',
             'IL': 'illinois', 'IN': 'indiana', 'IA': 'iowa', 'KS': 'kansas', 'KY': 'kentucky', 'LA': 'louisiana',
             'ME': 'maine', 'MD': 'maryland', 'MA': 'massachusetts', 'MI': 'michigan', 'MN': 'minnesota',
             'MS': 'mississippi', 'MO': 'missouri', 'MT': 'montana', 'NE': 'nebraska', 'NV': 'nevada',
             'NH': 'new hampshire', 'NJ': 'new jersey', 'NM': 'new mexico', 'NY': 'new york', 'NC': 'north carolina',
             'ND': 'north dakota', 'OH': 'ohio', 'OK': 'oklahoma', 'OR': 'oregon', 'PA': 'pennsylvania',
             'RI': 'rhode island', 'SC': 'south carolina', 'SD': 'south dakota', 'TN': 'tennessee', 'TX': 'texas',
             'UT': 'utah', 'VT': 'vermont', 'VA': 'virginia', 'WA': 'washington', 'WV': 'west virginia',
             'WI': 'wisconsin', 'WY': 'wyoming', 'DC': 'district of columbia'}
LOW_VALUE = {'stem', 'leaf', 'stem and leaves', 'stem_and_leaves', 'whole animal', 'whole insect', 'fruiting body', 'grass'}
NONVISUAL = {'natural', 'fresh', 'vibrant', 'lush', 'delicate', 'tender', 'small', 'bright', 'sparse', 'clustered', 'fuzzy',
             'creamy', 'muddy', 'rough', 'dark', 'soft'}


def lst(v):
    if v is None:
        return []
    if isinstance(v, str):
        v = re.split(r'[,;/]| and ', v)
    out = []
    for x in v:
        if isinstance(x, str):
            x = x.strip().lower()
            if x and x not in out:
                out.append(x)
    return out


# frequency of (field, value) across vision entries -> demote constants
freq = collections.Counter()
nv = 0
for v in vision.values():
    t = v.get('tags')
    if not t:
        continue
    nv += 1
    for f in ('viewpoint', 'light', 'background', 'habitat', 'substrate'):
        for x in lst(t.get(f)):
            freq[(f, x)] += 1


def is_constant(f, x):
    return nv >= 10 and freq[(f, x)] / nv > 0.6


rows, detail = [], []
for r in res:
    # raw frames only: a JPEG match is an earlier export of the photo, not an original
    good = [(p['best']['frame_id'], p['best']['file'], p['best']['captured']) for p in r['photos']
            if p.get('best') and p['best']['corr'] >= TH
            and (ALLOW_JPEG or p['best']['file'].lower().endswith(('.nef', '.dng', '.cr2', '.cr3', '.arw', '.raf', '.orf', '.rw2')))]
    if not good:
        continue
    frames = sorted({(fid, fn) for fid, fn, _ in good}, key=lambda x: x[1])
    o = inat[r['obs']]
    t = o.get('taxon') or {}
    info = taxa.get(str(t.get('id')))
    if not t.get('name') or not info:
        continue
    flags = []
    rank, research = info['rank'], (o.get('quality') == 'research')
    anc = info['ancestors']
    id_name, id_common, id_rank = info['name'], info.get('common'), rank
    if rank in ('species', 'subspecies', 'variety', 'form', 'hybrid') and not research:
        g = next((a for a in reversed(anc) if a['rank'] == 'genus'), None)
        if g:
            id_name, id_common, id_rank = g['name'], g.get('common'), 'genus'
            flags.append('tentative_species_tagged_at_genus')
    kw = []
    seen_norm = set()

    def add(x):
        if x:
            x = str(x).strip().lower()
            if ',' in x or ' and allies' in x or 'subtribe' in x:   # commas would split into junk keywords in IPTC/Adobe's parser
                return
            norm = x[:-1] if (x.endswith('s') and len(x) > 4 and not x.endswith('ss')) else x
            if x and norm not in seen_norm and len(x) <= 40:
                kw.append(x)
                seen_norm.add(norm)

    def buyer_ok(c):
        c = c.lower()
        return (',' not in c and ' and allies' not in c and 'tribe' not in c and c not in STOP_ANC
                and c not in JARGON and len(c) <= 30)

    tg = (vision.get(str(r['obs'])) or {}).get('tags') or {}
    sub = [('whole animal' if (x == 'whole insect' and info.get('iconic') != 'Insecta') else x) for x in lst(tg.get('subject_part'))]
    if 'subject_part' in tg:
        tg = {**tg, 'subject_part': sub}
    iconic = info.get('iconic')
    group = ICONIC.get(iconic, [])
    is_mushroom = 'fruiting body' in sub and iconic == 'Fungi'
    is_flower = 'flower' in sub and iconic == 'Plantae'
    ancs = [a for a in anc if a.get('common') and a['rank'] in ('phylum', 'class', 'order', 'family') and buyer_ok(a['common'])]

    # TIER 1 (positions 1-10, the ranking-critical ones): what a buyer actually searches for
    t1 = [id_common, id_name]
    t1 += group[:1]
    if is_mushroom:
        t1.append('mushroom')
    if is_flower:
        t1 += ['flower', 'wildflower']
    t1 += lst(tg.get('colors'))[:2]
    for f in ('habitat', 'substrate'):
        t1 += [x for x in lst(tg.get(f)) if not is_constant(f, x) and x not in LOW_VALUE]
    t1 += lst(tg.get('viewpoint'))
    # TIER 2: general -> specific taxonomy in plain words, then generic setting terms
    t2 = [a['common'] for a in reversed(ancs[-3:])]
    t2 += group[1:]
    t2 += ['nature', 'outdoors']
    if tg and tg.get('people_visible') is False and tg.get('text_or_logo_visible') is False:
        t2.append('no people')
    t2 += lst(tg.get('colors'))[2:3]
    t2 += [x for x in lst(tg.get('subject_part')) if x not in LOW_VALUE and not is_constant('subject_part', x)]
    for x in lst(tg.get('background')):
        t2.append('blurred background' if 'blur' in x else (x if not is_constant('background', x) else None))
    # TIER 3: specialist and context terms
    t3 = []
    for rk in ('genus', 'family'):
        a_ = next((a for a in anc if a['rank'] == rk), None)
        if a_ and a_['name'].lower() != (id_name or '').lower():
            t3.append(a_['name'])
    t3 += [x for x in lst(tg.get('adjectives')) if x not in NONVISUAL]
    od = o.get('observed_on')
    if od and len(od) >= 7:
        t3.append(SEASON.get(int(od[5:7])))
    sensitive = bool(o.get('obscured') or o.get('geoprivacy') or o.get('taxon_geoprivacy'))
    if sensitive:
        flags.append('location_withheld_obscured_or_sensitive')
    else:
        pg = [x.strip() for x in (o.get('place_guess') or '').split(',') if x.strip() and not re.search(r'\d', x)]
        for x in pg[-3:]:
            x = US_STATES.get(x.upper(), x)
            x = re.sub(r'\s+Co\.?$', ' County', x, flags=re.I)          # 'Berks Co.' -> 'Berks County'
            x = re.sub(r'\bTwp\.?(?=\s|$)', 'Township', x, flags=re.I)
            x = re.sub(r'\bMt\.?(?=\s)', 'Mount', x, flags=re.I)
            if x.lower() in ('usa', 'us', 'united states of america'):
                x = 'united states'
            t3.append(x)
    for x in t1 + t2 + t3:
        add(x)
    kws = kw[:MAXKW]
    # title
    if id_common:
        base = id_common
    else:
        fb = next((a['common'] for a in reversed(anc) if a.get('common') and a['rank'] in ('genus', 'subfamily', 'family', 'order', 'class')
                   and buyer_ok(a['common'])), None)
        base = fb or id_name
    base = base[:1].upper() + base[1:]
    phr = ''
    s_ = [x for x in lst(tg.get('substrate'))]
    h_ = [x for x in lst(tg.get('habitat'))]
    if s_ and s_[0] not in ('stem', 'leaf'):
        phr = ' on ' + s_[0]
    elif h_:
        phr = ' in ' + h_[0]
    part = ''
    if is_flower and 'flower' not in base.lower():
        part = ' flower'
    elif is_mushroom and 'mushroom' not in base.lower():
        part = ' mushroom'
    title = (base + part + phr)
    if len(title) > MAXTITLE:
        title = base + part
    if len(title) > MAXTITLE:
        title = base[:MAXTITLE]
    if not tg:
        flags.append('no_vision_tags_yet')
    if tg.get('people_visible'):
        flags.append('PEOPLE_VISIBLE_check_release')
    if tg.get('text_or_logo_visible'):
        flags.append('TEXT_OR_LOGO_check')
    if not research:
        flags.append('needs_id_on_inat')
    rows.append({'obs_id': r['obs'], 'n_frames': len(frames), 'frames': ';'.join(fn for _, fn in frames),
                 'title': title, 'n_keywords': len(kws), 'keywords': ', '.join(kws), 'id_level': id_rank,
                 'quality': o.get('quality'), 'flags': ' | '.join(flags)})
    detail.append({**rows[-1], 'frame_ids': [fid for fid, _ in frames]})

out = os.path.join(ROOT, OUTP + '.csv')
with open(out, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
json.dump(detail, open(os.path.join(ROOT, OUTP + '.json'), 'w'), indent=1)
print('observations in preview:', len(rows), '| frames covered:', sum(r['n_frames'] for r in rows))
print('with vision tags:', sum(1 for r in rows if 'no_vision_tags_yet' not in r['flags']),
      '| keyword count median:', sorted(r['n_keywords'] for r in rows)[len(rows) // 2])
print('people flagged:', sum(1 for r in rows if 'PEOPLE_VISIBLE' in r['flags']),
      '| text/logo flagged:', sum(1 for r in rows if 'TEXT_OR_LOGO' in r['flags']),
      '| tentative->genus:', sum(1 for r in rows if 'tentative' in r['flags']),
      '| location withheld:', sum(1 for r in rows if 'location_withheld' in r['flags']))
