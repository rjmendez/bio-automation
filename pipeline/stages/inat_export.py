"""Bulk export of the photos queued for iNaturalist, with the review console's data embedded in each file (EXIF + XMP + IPTC).

The files are the upload copies already on disk; only metadata segments are rewritten, pixel data is byte-identical. What goes in:
  EXIF   camera data kept; GPS lat/lon (+ HPositioningError = accuracy) from the decision or the proposed location;
         DateTimeOriginal + OffsetTimeOriginal when the reviewer changed the observation time (else each photo keeps its own capture time);
         ImageDescription = the description.
  XMP/IPTC  title (taxon / what you saw), keywords (existing + taxon + common name + your tags), caption.
Things EXIF cannot carry (geoprivacy, captive, taxon id, status) go to observations.csv in the zip, one row per observation.
Layout: <group folder>/<nn>_<stem>.jpg so photos of one observation stay together and sort in capture order.
Contains private coordinates when you set them: treat the zip as private.
"""
import csv
import json
import io
import json
import os
import re
import time
import zipfile
from datetime import datetime

import piexif
from PIL import IptcImagePlugin, Image

import embed_meta

EXPORT_DIR = os.environ.get('BIO_EXPORT_DIR', r'F:\services\inat-console\export')
# US zones (the console's choices); DST window is approximate (second Sunday of March .. first Sunday of November)
BASE = {'Eastern Time (US & Canada)': -5, 'Central Time (US & Canada)': -6, 'Mountain Time (US & Canada)': -7,
        'Pacific Time (US & Canada)': -8, 'Alaska': -9, 'Hawaii': -10, 'UTC': 0}
NO_DST = {'Arizona': -7, 'Hawaii': -10, 'UTC': 0}


def offset_hours(tz, day):
    if tz in NO_DST:
        return NO_DST[tz]
    base = BASE.get(tz)
    if base is None:
        return None
    try:
        m, d = int(day[5:7]), int(day[8:10])
    except (ValueError, IndexError):
        return base
    dst = (3, 10) <= (m, d) < (11, 3)
    return base + (1 if dst else 0)


def fmt_offset(h):
    return None if h is None else f'{"+" if h >= 0 else "-"}{abs(h):02d}:00'


def dms(v):
    v = abs(v)
    d = int(v)
    mi = int((v - d) * 60)
    return ((d, 1), (mi, 1), (round((v - d - mi / 60) * 3600 * 10000), 10000))


AUTO_GROUP_WORDS = {'fungus', 'plant', 'insect', 'arthropod', 'arachnid', 'spider', 'bird', 'mammal', 'reptile', 'amphibian', 'mollusc',
                    'animal', 'slime mold', 'alga'}
_TL = None


def _tag_log():
    global _TL
    if _TL is None:
        p = os.path.join(os.environ.get('BIO_DATA', r'F:\bio-automation-data'), 'tag_raws_log.json')
        _TL = json.load(open(p, encoding='utf-8')) if os.path.exists(p) else {}
    return _TL


def existing_iptc(data):
    try:
        info = IptcImagePlugin.getiptcinfo(Image.open(io.BytesIO(data))) or {}
    except Exception:
        return '', []
    t = info.get((2, 5))
    k = info.get((2, 25))
    ks = [x.decode('utf-8', 'replace') for x in (k if isinstance(k, list) else ([k] if k else []))]
    return (t.decode('utf-8', 'replace') if t else ''), ks


def ascii_safe(s):
    return (s or '').encode('ascii', 'replace').decode('ascii')


def embed(path, rec, default, first_time_changed):
    data = open(path, 'rb').read()
    old_title, old_kw = existing_iptc(data)
    what = (rec.get('species_guess') or rec.get('taxon_name') or '').strip()
    if what:                                    # a chosen taxon replaces the auto-generated ID keywords (group word, auto species)
        m = re.search(r'_(\d+)\.jpg$', path)
        basis = (_tag_log().get(m.group(1)) or {}).get('id_basis', '') if m else ''
        if basis.startswith('species'):
            old_kw = old_kw[2:]
        old_kw = [k for k in old_kw if k.lower() not in AUTO_GROUP_WORDS]
    title = what or old_title or 'Nature observation'
    tags = [t.strip() for t in re.split(r'[,;]', rec.get('tags') or '') if t.strip()]
    kws, seen = [], set()
    for k in [rec.get('taxon_name'), rec.get('species_guess')] + tags + old_kw:
        k = re.sub(r'\s+', ' ', (k or '').replace(',', ' ')).strip().lower()
        if k and k not in seen:
            seen.add(k)
            kws.append(k)
    kws = kws[:49]
    caption = (rec.get('description') or '').strip() or title + '.'
    new = embed_meta.rewrite(data, title, kws, caption)       # XMP + IPTC; this also drops EXIF GPS, which is re-added below
    ex = piexif.load(new)
    lat, lon = rec.get('lat'), rec.get('lon')
    if lat is None:
        lat, lon = default.get('lat'), default.get('lon')
    if lat is not None and lon is not None:
        g = ex.setdefault('GPS', {})
        g[piexif.GPSIFD.GPSVersionID] = (2, 3, 0, 0)
        g[piexif.GPSIFD.GPSLatitudeRef] = 'N' if lat >= 0 else 'S'
        g[piexif.GPSIFD.GPSLatitude] = dms(lat)
        g[piexif.GPSIFD.GPSLongitudeRef] = 'E' if lon >= 0 else 'W'
        g[piexif.GPSIFD.GPSLongitude] = dms(lon)
        acc = rec.get('positional_accuracy') or default.get('positional_accuracy')
        if acc:
            g[piexif.GPSIFD.GPSHPositioningError] = (int(acc), 1)
    if first_time_changed:
        try:
            dt = datetime.strptime(first_time_changed[:19], '%Y-%m-%d %H:%M:%S')
            s = dt.strftime('%Y:%m:%d %H:%M:%S').encode()
            ex.setdefault('Exif', {})[piexif.ExifIFD.DateTimeOriginal] = s
            ex.setdefault('Exif', {})[piexif.ExifIFD.DateTimeDigitized] = s
            ex.setdefault('0th', {})[piexif.ImageIFD.DateTime] = s
            off = fmt_offset(offset_hours(rec.get('time_zone') or default.get('time_zone'), dt.strftime('%Y-%m-%d')))
            if off:
                for name in ('OffsetTime', 'OffsetTimeOriginal', 'OffsetTimeDigitized'):
                    tag = getattr(piexif.ExifIFD, name, None)
                    if tag:
                        ex['Exif'][tag] = off.encode()
        except ValueError:
            pass
    if caption:
        ex.setdefault('0th', {})[piexif.ImageIFD.ImageDescription] = ascii_safe(caption)[:2000].encode('ascii')
    ex['thumbnail'] = None
    ex.pop('1st', None)
    out = io.BytesIO()
    piexif.insert(piexif.dump(ex), new, out)
    return out.getvalue()


def build_zip(groups, decisions, group_default, scope):
    """groups: {group id: [queue items]} to export. Returns the path of the zip."""
    os.makedirs(EXPORT_DIR, exist_ok=True)
    for f in os.listdir(EXPORT_DIR):                      # keep the folder small: drop exports older than two days
        p = os.path.join(EXPORT_DIR, f)
        if f.endswith('.zip') and time.time() - os.path.getmtime(p) > 2 * 86400:
            try:
                os.remove(p)
            except OSError:
                pass
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    path = os.path.join(EXPORT_DIR, f'inat-export-{scope}-{stamp}.zip')
    rows = []
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_STORED) as z:
        for g, items in groups.items():
            rec = decisions.get(g) or {}
            default = group_default(items)
            items = sorted(items, key=lambda i: i.get('t') or 0)
            order = rec.get('photo_order')
            if order:
                items = [i for i in items if i['idx'] in order]
            what = re.sub(r'[^A-Za-z0-9 _-]+', '', (rec.get('species_guess') or rec.get('taxon_name') or 'unidentified')).strip().replace(' ', '_')[:40]
            folder = f'g{g:04d}_{what}'
            changed = rec.get('observed_on_string') if rec.get('observed_on_string') and rec['observed_on_string'] != default.get('observed_on_string') else None
            names = []
            for n, it in enumerate(items, 1):
                stem = re.sub(r'[^A-Za-z0-9._-]+', '_', os.path.splitext(os.path.basename(it['copy']))[0])[:60]
                name = f'{n:02d}_{stem}.jpg'
                try:
                    z.writestr(f'{folder}/{name}', embed(it['copy'], rec, default, changed))
                except Exception:
                    z.write(it['copy'], f'{folder}/{name}')       # never lose a photo to a metadata problem
                names.append(name)
            rows.append({'group': g, 'folder': folder, 'files': ';'.join(names), 'status': rec.get('status', 'pending'),
                         'taxon_name': rec.get('taxon_name') or '', 'taxon_id': rec.get('taxon_id') or '',
                         'species_guess': rec.get('species_guess') or '',
                         'observed_on_string': rec.get('observed_on_string') or default.get('observed_on_string'),
                         'time_zone': rec.get('time_zone') or default.get('time_zone') or '',
                         'latitude': rec.get('lat') if rec.get('lat') is not None else default.get('lat'),
                         'longitude': rec.get('lon') if rec.get('lon') is not None else default.get('lon'),
                         'positional_accuracy': rec.get('positional_accuracy') or default.get('positional_accuracy') or '',
                         'geoprivacy': rec.get('geoprivacy') or '', 'captive': rec.get('captive') or False,
                         'tags': rec.get('tags') or '', 'description': rec.get('description') or ''})
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(rows[0]) if rows else ['group'])
        w.writeheader()
        w.writerows(rows)
        z.writestr('observations.csv', buf.getvalue())
    return path, len(rows), sum(len(r['files'].split(';')) for r in rows if r['files'])
