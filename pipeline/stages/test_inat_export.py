"""Smoke test for inat_export: fabricated decision, temp output dir. Checks: flat layout, only time + GPS + camera EXIF, no XMP/IPTC,
no title/keywords/description, pixels identical. Writes nothing to the ledger."""
import hashlib
import io
import os
import sys
import zipfile

os.environ['BIO_EXPORT_DIR'] = r'F:\bio-automation-data\tmp_export_test'
os.environ['BIO_QUEUE_TAG'] = '_lc'
sys.path.insert(0, r'F:\bio-automation-data')
import piexif
from PIL import Image

import inat_console as C
import inat_export as X
import jpeg_segments as J

g = next(iter(C.GROUPS))
items = C.GROUPS[g]
default = C.group_default(items)
rec = {'group': g, 'status': 'approved', 'taxon_id': 47219, 'taxon_name': 'Apis mellifera', 'species_guess': 'Western Honey Bee',
       'description': 'must NOT be embedded', 'tags': 'must, not, be, embedded', 'geoprivacy': 'obscured', 'captive': False,
       'observed_on_string': '2026-07-30 09:15:00', 'time_zone': 'Eastern Time (US & Canada)', 'lat': 39.9305, 'lon': -77.4406,
       'positional_accuracy': 25, 'photo_order': [i['idx'] for i in items]}
path, ng, nf = X.build_zip({g: items}, {g: rec}, C.group_default, 'test')
z = zipfile.ZipFile(path)
names = z.namelist()
print('entries:', names)
print('flat (no folders):', all('/' not in n for n in names), '| no csv by default:', 'observations.csv' not in names)
jp = [n for n in names if n.endswith('.jpg')]
data = z.read(jp[0])
segs = [k for _, _, k in J.segments(data)]
ex = piexif.load(data)
gps = ex.get('GPS', {})
print('segments:', [s for s in segs if s], '| XMP/IPTC present:', any(k in ('XMP', 'IPTC/Photoshop') for k in segs))
print('GPS:', bool(gps.get(piexif.GPSIFD.GPSLatitude)), gps.get(piexif.GPSIFD.GPSLatitudeRef), gps.get(piexif.GPSIFD.GPSLongitudeRef),
      '| accuracy', gps.get(piexif.GPSIFD.GPSHPositioningError))
print('DateTimeOriginal:', ex['Exif'].get(piexif.ExifIFD.DateTimeOriginal), '| offset:', ex['Exif'].get(0x9011), '| camera:', ex['0th'].get(piexif.ImageIFD.Model))
print('description in EXIF:', ex['0th'].get(piexif.ImageIFD.ImageDescription))
raw = data
print('taxon/tag words anywhere in the file:', any(w in raw for w in (b'Honey', b'Apis', b'embedded')))
src = sorted(items, key=lambda i: i.get('t') or 0)[0]['copy']
print('pixels identical:', hashlib.sha1(Image.open(src).convert('RGB').tobytes()).hexdigest() == hashlib.sha1(Image.open(io.BytesIO(data)).convert('RGB').tobytes()).hexdigest())
# default (no decision): the photo's own time is kept, offset taken from LightCraft's recorded UTC offset, no GPS unless a location is known
path2, _, _ = X.build_zip({g: items}, {}, C.group_default, 'test2')
d2 = zipfile.ZipFile(path2).read([n for n in zipfile.ZipFile(path2).namelist() if n.endswith('.jpg')][0])
e2 = piexif.load(d2)
print('no-decision export: time', e2['Exif'].get(piexif.ExifIFD.DateTimeOriginal), '| offset', e2['Exif'].get(0x9011), '| GPS', bool(e2.get('GPS')))
