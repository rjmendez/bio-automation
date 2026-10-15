"""Smoke test for inat_export: fabricated decision, temp output dir, verifies metadata and pixel identity. Writes nothing to the ledger."""
import hashlib
import io
import json
import os
import sys
import zipfile

os.environ['BIO_EXPORT_DIR'] = r'F:\bio-automation-data\tmp_export_test'
os.environ['BIO_QUEUE_TAG'] = '_lc'
sys.path.insert(0, r'F:\bio-automation-data')
import piexif
from PIL import Image, IptcImagePlugin

import inat_console as C
import inat_export as X

g = next(iter(C.GROUPS))
items = C.GROUPS[g]
default = C.group_default(items)
rec = {'group': g, 'status': 'approved', 'taxon_id': 47219, 'taxon_name': 'Apis mellifera', 'species_guess': 'Western Honey Bee',
       'description': 'On a flower at the edge of the meadow', 'tags': 'bee, pollinator, test tag', 'geoprivacy': 'obscured', 'captive': False,
       'observed_on_string': '2026-07-30 09:15:00', 'time_zone': 'Eastern Time (US & Canada)', 'lat': 39.9305, 'lon': -77.4406,
       'positional_accuracy': 25, 'photo_order': [i['idx'] for i in items]}
path, ng, nf = X.build_zip({g: items}, {g: rec}, C.group_default, 'test')
print('zip:', path, '| observations', ng, 'photos', nf, '| MB', round(os.path.getsize(path) / 1e6, 1))
z = zipfile.ZipFile(path)
names = z.namelist()
print('entries:', names[:4], '...', len(names))
jp = [n for n in names if n.endswith('.jpg')]
src = sorted(items, key=lambda i: i.get('t') or 0)[0]['copy']
data = z.read(jp[0])
im = Image.open(io.BytesIO(data))
ex = piexif.load(data)
gps = ex.get('GPS', {})
print('GPS present:', bool(gps.get(piexif.GPSIFD.GPSLatitude)), '| ref', gps.get(piexif.GPSIFD.GPSLatitudeRef), gps.get(piexif.GPSIFD.GPSLongitudeRef),
      '| accuracy', gps.get(piexif.GPSIFD.GPSHPositioningError))
print('DateTimeOriginal:', ex['Exif'].get(piexif.ExifIFD.DateTimeOriginal), '| OffsetTimeOriginal:', ex['Exif'].get(0x9011))
print('camera kept:', ex['0th'].get(piexif.ImageIFD.Model), '| description:', ex['0th'].get(piexif.ImageIFD.ImageDescription))
iptc = IptcImagePlugin.getiptcinfo(im) or {}
print('IPTC title:', (iptc.get((2, 5)) or b'').decode(), '| keywords:', [k.decode() for k in (iptc.get((2, 25)) or [])][:8])
print('XMP present:', b'http://ns.adobe.com/xap/1.0/' in data)
a = hashlib.sha1(Image.open(src).convert('RGB').tobytes()).hexdigest()
b = hashlib.sha1(im.convert('RGB').tobytes()).hexdigest()
print('pixels identical to the source copy:', a == b)
print('observations.csv header:', z.read('observations.csv').decode().splitlines()[0][:150])
