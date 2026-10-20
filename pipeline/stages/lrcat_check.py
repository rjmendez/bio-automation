"""Read-only look at the Lightroom catalog: which of the triage frames did the owner already go through by hand?

Opens the .lrcat with SQLite mode=ro&immutable=1 (never writes, never takes a lock). Signals per frame, matched by
(folder day, file base name): present in the catalog, develop adjustments, rating, pick/reject flag, colour label, in a collection,
keyworded. Output: lrcat_signals.json {frame id: [signals]} and a printed summary.
"""
import collections
import json
import os
import sqlite3
import sys
import time

ROOT = r'F:\bio-automation-data'
CAT = r'C:\Users\rjmendez-admin\Pictures\Lightroom\Lightroom Catalog.lrcat'
BS = chr(92)


def main():
    print('catalog MB', round(os.path.getsize(CAT) / 1e6, 1), '| last modified', time.ctime(os.path.getmtime(CAT)))
    db = sqlite3.connect('file:' + CAT.replace(BS, '/') + '?mode=ro&immutable=1', uri=True)
    tabs = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
    print('tables present:', {t: (t in tabs) for t in ('Adobe_images', 'AgLibraryFile', 'AgLibraryFolder', 'AgLibraryRootFolder',
                                                       'Adobe_imageDevelopSettings', 'AgLibraryCollectionImage', 'AgLibraryKeywordImage')})
    print('images in catalog:', db.execute('select count(*) from Adobe_images').fetchone()[0])
    print('root folders:', [r[0] for r in db.execute('select absolutePath from AgLibraryRootFolder')][:6])
    cols = [r[1] for r in db.execute('pragma table_info(Adobe_images)')]
    print('has rating/pick/colorLabels:', [c for c in ('rating', 'pick', 'colorLabels', 'captureTime') if c in cols])

    q = '''
    select i.id_local, f.baseName, f.extension, fo.pathFromRoot, ro.absolutePath, i.rating, i.pick, i.colorLabels,
           coalesce(d.hasDevelopAdjustments, 0),
           (select count(*) from AgLibraryCollectionImage c where c.image = i.id_local),
           (select count(*) from AgLibraryKeywordImage k where k.image = i.id_local)
    from Adobe_images i
    join AgLibraryFile f on f.id_local = i.rootFile
    join AgLibraryFolder fo on fo.id_local = f.folder
    join AgLibraryRootFolder ro on ro.id_local = fo.rootFolder
    left join Adobe_imageDevelopSettings d on d.image = i.id_local
    '''
    cat = {}
    for iid, base, ext, rel, root, rating, pick, label, dev, coll, kw in db.execute(q):
        day = None
        for part in (rel or '').replace(BS, '/').split('/'):
            if len(part) == 10 and part[4] == '-' and part[7] == '-':
                day = part
        sig = []
        if dev:
            sig.append('develop_adjustments')
        if rating:
            sig.append(f'rating_{int(rating)}')
        if pick:
            sig.append('pick' if pick > 0 else 'reject')
        if label:
            sig.append('label')
        if coll:
            sig.append('in_collection')
        if kw:
            sig.append('keywords')
        cat[((day or ''), (base or '').lower())] = (ext, sig, rel)
    idx = json.load(open(os.path.join(ROOT, 'triage_index.json')))
    found, with_sig, bysig = 0, 0, collections.Counter()
    out = {}
    for it in idx:
        stem = os.path.splitext(os.path.basename(it['file']))[0].lower()
        hit = cat.get((it['day'], stem))
        if hit:
            found += 1
            if hit[1]:
                with_sig += 1
                out[it['id']] = hit[1]
                for s in hit[1]:
                    bysig[s] += 1
    json.dump(out, open(os.path.join(ROOT, 'lrcat_signals.json'), 'w'))
    print(f'triage frames {len(idx)} | present in the Lightroom catalog {found} | with a manual-work signal {with_sig}')
    print('signals:', dict(bysig))
    days = collections.Counter(it['day'] for it in idx if it['id'] in out)
    print('days with signals (top):', days.most_common(6))


if __name__ == '__main__':
    main()
