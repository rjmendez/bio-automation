"""Remove the auto-generated identification/tag metadata (title, caption, keywords) from the rendered raws.

Owner rule 2026-10-15: identifications and tags for photos that have not been identified yet are added FROM iNaturalist data after the
owner identifies them. What an iNat submission needs is time, GPS and the auto-adjusted pixels, nothing else.

For every JPEG under the target folder this drops the XMP APP1 and IPTC/Photoshop APP13 segments (the only places tag_raws.py wrote
to) and keeps everything else untouched: EXIF (camera, capture time), ICC profile, the image data. Per file it verifies that the bytes from
the start-of-scan marker on are identical before/after, and refuses to write otherwise. Atomic (temp + replace). --dry-run counts only.
"""
import argparse
import os
import struct
import sys
import time

DROP = ('XMP', 'IPTC')


def parse(data):
    """Return (list of (start, end, kind)), sos_offset)."""
    segs, i = [], 2
    while i < len(data) - 4 and data[i] == 0xFF:
        m = data[i + 1]
        if m == 0xDA:
            return segs, i
        ln = struct.unpack('>H', data[i + 2:i + 4])[0]
        payload = data[i + 4:i + 2 + ln]
        kind = ''
        if m == 0xE1 and payload.startswith(b'http://ns.adobe.com/xap/1.0/'):
            kind = 'XMP'
        elif m == 0xED:
            kind = 'IPTC'
        segs.append((i, i + 2 + ln, kind))
        i += 2 + ln
    return segs, None


def purge(data):
    if data[:2] != b'\xff\xd8':
        raise ValueError('not a JPEG')
    segs, sos = parse(data)
    if sos is None:
        raise ValueError('no start-of-scan marker')
    drop = [s for s in segs if s[2] in DROP]
    if not drop:
        return data, 0
    out = bytearray(data[:2])
    for s, e, kind in segs:
        if kind not in DROP:
            out += data[s:e]
    out += data[sos:]
    if bytes(out[len(out) - (len(data) - sos):]) != data[sos:]:
        raise ValueError('scan data changed')
    return bytes(out), len(drop)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    n = changed = segs_dropped = failed = bytes_saved = 0
    t0 = time.time()
    for r, _, fs in os.walk(a.root):
        for f in fs:
            if not f.lower().endswith(('.jpg', '.jpeg')):
                continue
            p = os.path.join(r, f)
            n += 1
            try:
                data = open(p, 'rb').read()
                new, k = purge(data)
                if k:
                    changed += 1
                    segs_dropped += k
                    bytes_saved += len(data) - len(new)
                    if not a.dry_run:
                        tmp = p + '.part'
                        open(tmp, 'wb').write(new)
                        os.replace(tmp, p)
            except Exception as e:
                failed += 1
                print('FAILED', p, type(e).__name__, str(e)[:80], flush=True)
    print(f'{"DRY RUN: " if a.dry_run else ""}files {n} | with tag segments {changed} | segments {"to drop" if a.dry_run else "dropped"} {segs_dropped} '
          f'| bytes {bytes_saved} | failed {failed} | {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
