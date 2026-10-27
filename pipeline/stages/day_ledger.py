"""Per-day record of the owner's own work, to find where the backlog was left off.

A day is WORKED when at least one of these exists for it: a JPEG you exported (any subfolder of the day, outside our own outputs),
an iNaturalist observation observed that day, or Lightroom catalog work on photos of that day (develop adjustments, rating,
pick/reject, colour label, collection, keywords). Everything is read-only (inventories, iNat export, .lrcat opened immutable).
Output: day_ledger.json {day: {...}} and a printed chronological table for F:\\Bio (the LightCraft library).
"""
import collections
import csv
import json
import os
import re
import sqlite3
import sys

ROOT = os.environ.get('BIO_DATA', r'F:\bio-automation-data')
CAT = os.environ.get('BIO_LRCAT', os.path.join(os.path.expanduser('~'), 'Pictures', 'Lightroom', 'Lightroom Catalog.lrcat'))
BS = chr(92)
RAWX = ('.nef', '.dng', '.arw', '.cr2', '.cr3', '.raf', '.orf', '.rw2')
JX = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')


def main():
    days = collections.defaultdict(lambda: collections.Counter())
    # inventories: raws and the owner's JPEGs per day folder (F only here; the NAS is reported separately)
    for tag, inv in (('F', 'f_bio_inventory.csv'), ('NAS', 'nas_inventory.csv')):
        for r in csv.DictReader(open(os.path.join(ROOT, inv), encoding='utf-8-sig')):
            p = r['path'].replace(BS, '/').split('/')
            if len(p) < 3 or not p[0].isdigit() or len(p[1]) != 10:
                continue
            name = p[-1].lower()
            if name.endswith(RAWX):
                days[p[1]][f'{tag}_raw'] += 1
            elif name.endswith(JX):
                days[p[1]][f'{tag}_jpeg'] += 1
    # iNat observations by observed day, and when they were created (when the owner actually uploaded)
    obs = json.load(open(os.path.join(ROOT, 'inat_observations.json'), encoding='utf-8'))['observations']
    created = collections.defaultdict(list)
    for o in obs:
        d = o.get('observed_on')
        if d:
            days[d]['inat_obs'] += 1
            if o.get('created_at'):
                created[d].append(o['created_at'][:10])
    # Lightroom catalog work per day
    db = sqlite3.connect('file:' + CAT.replace(BS, '/') + '?mode=ro&immutable=1', uri=True)
    q = '''select fo.pathFromRoot, i.rating, i.pick, i.colorLabels, coalesce(d.hasDevelopAdjustments,0),
                  (select count(*) from AgLibraryCollectionImage c where c.image=i.id_local),
                  (select count(*) from AgLibraryKeywordImage k where k.image=i.id_local)
           from Adobe_images i join AgLibraryFile f on f.id_local=i.rootFile join AgLibraryFolder fo on fo.id_local=f.folder
           left join Adobe_imageDevelopSettings d on d.image=i.id_local'''
    for rel, rating, pick, label, dev, coll, kw in db.execute(q):
        day = next((x for x in (rel or '').replace(BS, '/').split('/') if len(x) == 10 and x[4] == '-' and x[7] == '-'), None)
        if not day:
            continue
        if dev:
            days[day]['lr_adjusted'] += 1
        if rating:
            days[day]['lr_rated'] += 1
        if pick:
            days[day]['lr_pick'] += 1
        if label:
            days[day]['lr_label'] += 1
        if coll:
            days[day]['lr_collection'] += 1
        if kw:
            days[day]['lr_keywords'] += 1
    out = {}
    for d, c in days.items():
        worked = bool(c['F_jpeg'] or c['NAS_jpeg'] or c['inat_obs'] or c['lr_adjusted'] or c['lr_rated'] or c['lr_label'] or c['lr_collection'] or c['lr_keywords'])
        out[d] = dict(c, worked=worked, uploaded_on=sorted(set(created.get(d, [])))[-1:] or None)
    json.dump(out, open(os.path.join(ROOT, 'day_ledger.json'), 'w'), indent=0)

    print(f'{"day":10s} {"Fraw":>5s} {"Fjpg":>5s} {"iNat":>5s} {"LRadj":>5s} {"pick":>4s}  status')
    fdays = sorted(d for d, v in out.items() if v.get('F_raw'))
    for d in fdays:
        v = out[d]
        print(f'{d:10s} {v.get("F_raw", 0):5d} {v.get("F_jpeg", 0):5d} {v.get("inat_obs", 0):5d} {v.get("lr_adjusted", 0):5d} {v.get("lr_pick", 0):4d}  '
              f'{"worked" if v["worked"] else "UNTOUCHED"}')


if __name__ == '__main__':
    main()
