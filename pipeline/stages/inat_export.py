"""Bulk export of the photos queued for iNaturalist: ONE FLAT folder of JPEGs carrying time and GPS, nothing else.

Owner rule 2026-10-15: what an iNaturalist submission needs is the capture time, the GPS position and the auto-adjusted pixels.
No identification, no title, no keywords, no description: those come back FROM iNaturalist data after the owner identifies the
observation, and Adobe Stock titles/keywords are a later step that starts only from verified IDs.

Per file (the upload copy already on disk; pixel data is byte-identical):
  * any XMP / IPTC segment is dropped (nothing but EXIF + colour profile is left);
  * EXIF keeps the camera data and the photo's own DateTimeOriginal; OffsetTimeOriginal is set from the UTC offset LightCraft recorded
    (or, failing that, the reviewer's time zone); if the reviewer edited the observation time it is written to every photo of the group;
  * GPS latitude/longitude (+ HPositioningError = accuracy) from the reviewer's decision, else the proposed location.
Layout: a single directory, no subfolders: g<group>_<nn>_<stem>.jpg (photos of one observation stay adjacent when sorted).
observations.csv (per-observation fields EXIF cannot carry) is only included when asked for (csv=True); it is not part of the default.
Contains private coordinates when a location is set: treat the zip as private.
"""
import csv
import io
import os
import re
import struct
import time
import zipfile
from datetime import datetime

import piexif

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
    return base + (1 if (3, 10) <= (m, d) < (11, 3) else 0)


def fmt_offset(seconds_or_hours, seconds=True):
    if seconds_or_hours is None:
        return None
    s = int(round(seconds_or_hours if seconds else seconds_or_hours * 3600))
    sign = '+' if s >= 0 else '-'
    s = abs(s)
    return f'{sign}{s // 3600:02d}:{(s % 3600) // 60:02d}'


def dms(v):
    v = abs(v)
    d = int(v)
    mi = int((v - d) * 60)
    return ((d, 1), (mi, 1), (round((v - d - mi / 60) * 3600 * 10000), 10000))


def drop_tag_segments(data):
    """Remove XMP (APP1 'http://ns.adobe.com/xap/1.0/') and IPTC/Photoshop (APP13) segments; everything from the scan header on is kept."""
    if data[:2] != b'\xff\xd8':
        raise ValueError('not a JPEG')
    out, i = bytearray(data[:2]), 2
    while i < len(data) - 4 and data[i] == 0xFF:
        m = data[i + 1]
        if m == 0xDA:
            break
        ln = struct.unpack('>H', data[i + 2:i + 4])[0]
        seg = data[i:i + 2 + ln]
        payload = seg[4:]
        if not ((m == 0xE1 and payload.startswith(b'http://ns.adobe.com/xap/1.0/')) or m == 0xED):
            out += seg
        i += 2 + ln
    out += data[i:]
    return bytes(out)


def export_photo(path, item, rec, default, time_changed):
    """Bytes of one upload file carrying only camera EXIF + time + GPS."""
    data = drop_tag_segments(open(path, 'rb').read())
    ex = piexif.load(data)
    ex.pop('1st', None)
    ex['thumbnail'] = None
    ex.setdefault('0th', {}).pop(piexif.ImageIFD.ImageDescription, None)        # no description, no title
    lat, lon = rec.get('lat'), rec.get('lon')
    if lat is None:
        lat, lon = default.get('lat'), default.get('lon')
    ex['GPS'] = {}
    if lat is not None and lon is not None:
        g = ex['GPS']
        g[piexif.GPSIFD.GPSVersionID] = (2, 3, 0, 0)
        g[piexif.GPSIFD.GPSLatitudeRef] = 'N' if lat >= 0 else 'S'
        g[piexif.GPSIFD.GPSLatitude] = dms(lat)
        g[piexif.GPSIFD.GPSLongitudeRef] = 'E' if lon >= 0 else 'W'
        g[piexif.GPSIFD.GPSLongitude] = dms(lon)
        acc = rec.get('positional_accuracy') or default.get('positional_accuracy') or item.get('positional_accuracy')
        if acc:
            g[piexif.GPSIFD.GPSHPositioningError] = (int(acc), 1)
    exif = ex.setdefault('Exif', {})
    off = None
    if time_changed:
        try:
            dt = datetime.strptime(time_changed[:19], '%Y-%m-%d %H:%M:%S')
            s = dt.strftime('%Y:%m:%d %H:%M:%S').encode()
            exif[piexif.ExifIFD.DateTimeOriginal] = s
            exif[piexif.ExifIFD.DateTimeDigitized] = s
            ex.setdefault('0th', {})[piexif.ImageIFD.DateTime] = s
            off = fmt_offset(offset_hours(rec.get('time_zone') or default.get('time_zone'), dt.strftime('%Y-%m-%d')), seconds=False)
        except ValueError:
            pass
    if off is None:
        off = fmt_offset(item.get('tz_offset'))                   # the UTC offset LightCraft recorded for this photo
    if off is None and exif.get(piexif.ExifIFD.DateTimeOriginal):
        day = exif[piexif.ExifIFD.DateTimeOriginal][:10].decode('ascii', 'ignore').replace(':', '-')
        off = fmt_offset(offset_hours(rec.get('time_zone') or default.get('time_zone'), day), seconds=False)
    if off:
        for name in ('OffsetTime', 'OffsetTimeOriginal', 'OffsetTimeDigitized'):
            tag = getattr(piexif.ExifIFD, name, None)
            if tag:
                exif[tag] = off.encode()
    out = io.BytesIO()
    piexif.insert(piexif.dump(ex), data, out)
    return out.getvalue()


def build_zip(groups, decisions, group_default, scope, csv_sidecar=False):
    """groups: {group id: [queue items]}. Returns (zip path, observations, photos). Flat: no folders inside the zip."""
    os.makedirs(EXPORT_DIR, exist_ok=True)
    for f in os.listdir(EXPORT_DIR):                      # keep the folder small: drop exports older than two days
        p = os.path.join(EXPORT_DIR, f)
        if f.endswith('.zip') and time.time() - os.path.getmtime(p) > 2 * 86400:
            try:
                os.remove(p)
            except OSError:
                pass
    path = os.path.join(EXPORT_DIR, f'inat-export-{scope}-{datetime.now().strftime("%Y%m%d-%H%M%S")}.zip')
    rows, n_photos = [], 0
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_STORED) as z:
        for g, items in groups.items():
            rec = decisions.get(g) or {}
            default = group_default(items)
            items = sorted(items, key=lambda i: i.get('t') or 0)
            order = rec.get('photo_order')
            if order:
                items = [i for i in items if i['idx'] in order]
            changed = rec.get('observed_on_string') if rec.get('observed_on_string') and \
                rec['observed_on_string'] != default.get('observed_on_string') else None
            names = []
            for n, it in enumerate(items, 1):
                stem = re.sub(r'[^A-Za-z0-9._-]+', '_', os.path.splitext(os.path.basename(it['copy']))[0])[:70]
                name = f'g{g:04d}_{n:02d}_{stem}.jpg'
                try:
                    z.writestr(name, export_photo(it['copy'], it, rec, default, changed))
                except Exception:
                    z.write(it['copy'], name)                 # never lose a photo to a metadata problem
                names.append(name)
                n_photos += 1
            rows.append({'group': g, 'files': ';'.join(names), 'status': rec.get('status', 'pending'),
                         'observed_on_string': rec.get('observed_on_string') or default.get('observed_on_string'),
                         'time_zone': rec.get('time_zone') or default.get('time_zone') or '',
                         'latitude': rec.get('lat') if rec.get('lat') is not None else default.get('lat'),
                         'longitude': rec.get('lon') if rec.get('lon') is not None else default.get('lon'),
                         'positional_accuracy': rec.get('positional_accuracy') or default.get('positional_accuracy') or '',
                         'geoprivacy': rec.get('geoprivacy') or '', 'captive': rec.get('captive') or False})
        if csv_sidecar and rows:
            buf = io.StringIO()
            w = csv.DictWriter(buf, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
            z.writestr('observations.csv', buf.getvalue())
    return path, len(rows), n_photos
