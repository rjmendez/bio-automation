"""List the metadata segments of JPEG files (marker, size, what it is) and key EXIF facts. Read-only."""
import glob
import io
import os
import struct
import sys

import piexif
from PIL import Image


def segments(data):
    out, i = [], 2
    while i < len(data) - 4 and data[i] == 0xFF:
        m = data[i + 1]
        if m == 0xDA:                       # start of scan
            break
        ln = struct.unpack('>H', data[i + 2:i + 4])[0]
        payload = data[i + 4:i + 2 + ln]
        kind = ''
        if m == 0xE1 and payload.startswith(b'Exif'):
            kind = 'EXIF'
        elif m == 0xE1 and payload.startswith(b'http://ns.adobe.com/xap/1.0/'):
            kind = 'XMP'
        elif m == 0xED:
            kind = 'IPTC/Photoshop'
        elif m == 0xE2:
            kind = 'ICC'
        out.append((hex(m), ln, kind))
        i += 2 + ln
    return out


def describe(path):
    data = open(path, 'rb').read()
    ex = piexif.load(data)
    return {
        'segments': segments(data),
        'camera': ex['0th'].get(piexif.ImageIFD.Model),
        'time': ex['Exif'].get(piexif.ExifIFD.DateTimeOriginal),
        'gps': bool(ex.get('GPS')),
        'description': ex['0th'].get(piexif.ImageIFD.ImageDescription),
    }


if __name__ == '__main__':
    for p in sys.argv[1:]:
        print(os.path.basename(p), describe(p))
