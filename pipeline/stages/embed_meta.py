"""Write title / keywords (IPTC-IIM + XMP) into COPIES of existing JPEG exports, remove GPS, keep pixels untouched.

Lossless: only metadata segments are rewritten; the compressed image data is copied byte for byte.
usage: embed_meta.py --test N | --all
"""
import csv, hashlib, io, json, os, shutil, struct, sys, time
from xml.sax.saxutils import escape
import piexif
from PIL import Image
Image.MAX_IMAGE_PIXELS = None   # sizes are checked explicitly below against Adobe's 4-100 MP limits

ROOT = r'F:\bio-automation-data'
SRC = os.environ.get('BIO_SRC', r'F:\Bio')
MAP_FILE = os.environ.get('BIO_MAP', 'export_map.json')
PREVIEW_FILE = os.environ.get('BIO_PREVIEW', 'adobe_preview.csv')
MANIFEST_NAME = os.environ.get('BIO_MANIFEST', 'manifest.csv')
SKIP_EXISTING = os.environ.get('BIO_SKIP_EXISTING') == '1'
DST = r'F:\Bio-export\adobe-stock'
XMP_NS = b'http://ns.adobe.com/xap/1.0/\x00'


def segments(data):
    """Yield (marker, payload_bytes_incl_length_or_None, start, end) up to and including SOS; rest is raw scan data."""
    assert data[:2] == b'\xff\xd8'
    i = 2
    out = []
    while i < len(data):
        if data[i] != 0xFF:
            raise ValueError('bad marker stream')
        marker = data[i + 1]
        if marker == 0xDA:                       # SOS: header then entropy-coded data to EOI
            return out, i
        if marker in (0x01,) or 0xD0 <= marker <= 0xD9:
            out.append((marker, b'', i, i + 2)); i += 2; continue
        ln = struct.unpack('>H', data[i + 2:i + 4])[0]
        out.append((marker, data[i + 4:i + 2 + ln], i, i + 2 + ln))
        i += 2 + ln
    raise ValueError('no SOS')


def seg(marker, payload):
    assert len(payload) + 2 <= 65535, 'segment too large'
    return b'\xff' + bytes([marker]) + struct.pack('>H', len(payload) + 2) + payload


def iptc_record(rec, tag, value):
    v = value if isinstance(value, bytes) else value.encode('utf-8')
    return b'\x1c' + bytes([rec, tag]) + struct.pack('>H', len(v)) + v


def build_iptc(title, keywords, caption):
    b = iptc_record(1, 90, b'\x1b%G')                         # UTF-8
    b += iptc_record(2, 5, title.encode('utf-8')[:64].decode('utf-8', 'ignore'))        # ObjectName (max 64 bytes)
    for k in keywords:
        b += iptc_record(2, 25, k.encode('utf-8')[:64].decode('utf-8', 'ignore'))
    b += iptc_record(2, 105, title[:256])                      # Headline
    b += iptc_record(2, 120, caption[:2000])                   # Caption/Abstract
    if len(b) % 2:
        b += b'\x00'
    res = b'8BIM' + struct.pack('>H', 0x0404) + b'\x00\x00' + struct.pack('>I', len(b)) + b
    return b'Photoshop 3.0\x00' + res


def build_xmp(title, keywords, caption):
    li = ''.join(f'<rdf:li>{escape(k)}</rdf:li>' for k in keywords)
    x = ('<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>'
         '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
         '<rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:photoshop="http://ns.adobe.com/photoshop/1.0/">'
         f'<dc:title><rdf:Alt><rdf:li xml:lang="x-default">{escape(title)}</rdf:li></rdf:Alt></dc:title>'
         f'<dc:description><rdf:Alt><rdf:li xml:lang="x-default">{escape(caption)}</rdf:li></rdf:Alt></dc:description>'
         f'<dc:subject><rdf:Bag>{li}</rdf:Bag></dc:subject>'
         f'<photoshop:Headline>{escape(title)}</photoshop:Headline>'
         '</rdf:Description></rdf:RDF></x:xmpmeta><?xpacket end="w"?>')
    return XMP_NS + x.encode('utf-8')


def rewrite(src_bytes, title, keywords, caption):
    segs, sos = segments(src_bytes)
    exif_bytes = None
    keep = []
    for marker, payload, a, b in segs:
        if marker == 0xE1 and payload.startswith(b'Exif\x00\x00'):
            exif_bytes = src_bytes[a:b]
            continue
        if marker == 0xE1 and payload.startswith(XMP_NS):
            continue                                # old XMP dropped (develop history etc. does not ship)
        if marker == 0xED and payload.startswith(b'Photoshop 3.0'):
            continue                                # old IPTC/Photoshop block replaced
        keep.append(src_bytes[a:b])
    new_exif = b''
    if exif_bytes is not None:
        try:
            d = piexif.load(exif_bytes[4:])          # drop the marker+length, keep 'Exif\0\0...'
        except Exception:
            d = piexif.load(exif_bytes[10:] if exif_bytes[4:10] == b'Exif\x00\x00' else exif_bytes)
        d['GPS'] = {}                                 # remove every GPS tag
        d['thumbnail'] = None                         # drop embedded thumbnail (can hold the original frame incl. GPS-free but stale)
        d.pop('1st', None)
        new_exif = seg(0xE1, piexif.dump({k: v for k, v in d.items() if k in ('0th', 'Exif', 'GPS', 'Interop', '1st', 'thumbnail')}))
    out = [b'\xff\xd8']
    # APP0 (JFIF) first if present, then EXIF, XMP, IPTC, then the remaining kept segments
    app0 = [s for s in keep if s[:2] == b'\xff\xe0']
    rest = [s for s in keep if s[:2] != b'\xff\xe0']
    out += app0
    if new_exif:
        out.append(new_exif)
    out.append(seg(0xE1, build_xmp(title, keywords, caption)))
    out.append(seg(0xED, build_iptc(title, keywords, caption)))
    out += rest
    out.append(src_bytes[sos:])                       # SOS + entropy-coded data + EOI, untouched
    return b''.join(out)


def verify(orig_path, new_path, title, keywords, resized=False):
    from PIL import IptcImagePlugin
    errs = []
    with Image.open(orig_path) as a, Image.open(new_path) as b:
        if resized:
            # over-limit source: pixels are resampled on purpose; check the result is within the limit and keeps the aspect ratio
            if b.width * b.height > 100e6:
                errs.append('still over 100 MP')
            if abs(b.width / b.height - a.width / a.height) > 0.005 * (a.width / a.height):
                errs.append('aspect ratio changed')
        else:
            if a.size != b.size:
                errs.append('size differs')
            ha = hashlib.sha1(a.tobytes()).hexdigest()
            hb = hashlib.sha1(b.tobytes()).hexdigest()
            if ha != hb:
                errs.append('PIXELS DIFFER')
        iptc = IptcImagePlugin.getiptcinfo(b) or {}
        kws = iptc.get((2, 25))
        kws = [k.decode('utf-8') for k in (kws if isinstance(kws, list) else [kws] if kws else [])]
        if kws != keywords:
            errs.append(f'iptc keywords mismatch ({len(kws)} vs {len(keywords)})')
        try:
            ex = piexif.load(new_path)
            if ex.get('GPS'):
                errs.append('GPS still present')
        except Exception as e:
            errs.append('exif unreadable: ' + type(e).__name__)
        raw = open(new_path, 'rb').read()
        if XMP_NS not in raw or escape(title).encode() not in raw:
            errs.append('xmp missing')
    return errs


RELEASE_FIELDS = ('person_visible', 'face_visible', 'identifiable_person', 'text_or_logo_visible', 'brand_or_product_visible', 'property_identifiable')
_VF = os.path.join(ROOT, 'vision_files.json')
VISION = json.load(open(_VF, encoding='utf-8')) if os.path.exists(_VF) else {}


def release_flags(name):
    import visionlib                                  # nested vision answers must be flattened or flagged files look clear
    v = (VISION.get(name) or {}).get('tags') or {}
    if not v:
        return None                                   # not screened yet
    return visionlib.release_flags(v, name)


def resize_to_limit(src_bytes, target_mp=95.0):
    """Re-encode an over-100 MP JPEG at <= target_mp (quality 95, ICC kept). Pixels are resampled: documented as resized."""
    im = Image.open(io.BytesIO(src_bytes))
    w, h = im.size
    s = (target_mp * 1e6 / (w * h)) ** 0.5
    icc, exif = im.info.get('icc_profile'), im.info.get('exif')
    im = im.convert('RGB').resize((int(w * s), int(h * s)), Image.LANCZOS)
    b = io.BytesIO()
    im.save(b, 'JPEG', quality=95, subsampling=0, icc_profile=icc, exif=exif if exif else b'')
    return b.getvalue(), im.size


def load_records():
    mp = json.load(open(os.path.join(ROOT, MAP_FILE), encoding='utf-8'))
    prev = {r['obs_id']: r for r in csv.DictReader(open(os.path.join(ROOT, PREVIEW_FILE), encoding='utf-8'))}
    recs = []
    for r in mp:
        kws, titles, flags = [], [], set()
        for oid in dict.fromkeys(r['obs']):
            p = prev.get(str(oid))
            if not p:
                continue
            titles.append(p['title'])
            for k in [x.strip() for x in p['keywords'].split(', ') if x.strip()]:
                if k not in kws:
                    kws.append(k)
            for f in [x.strip() for x in p['flags'].split('|') if x.strip()]:
                flags.add(f)
        if not titles:
            continue
        recs.append({'jpeg': r['jpeg'], 'raw': r['raw'], 'title': titles[0][:70], 'keywords': kws[:49], 'flags': sorted(flags),
                     'obs': list(dict.fromkeys(r['obs'])), 'merged': len(titles) > 1})
    return recs


def main():
    recs = load_records()
    n = int(sys.argv[sys.argv.index('--test') + 1]) if '--test' in sys.argv else None
    todo = recs[:n] if n else recs
    ready = os.path.join(DST, 'ready')
    review = os.path.join(DST, '_needs_review')
    os.makedirs(ready, exist_ok=True)
    os.makedirs(review, exist_ok=True)
    manifest = []
    bad = 0
    skipped = 0
    t0 = time.time()
    for i, r in enumerate(todo, 1):
        src = os.path.join(SRC, r['jpeg'].replace('/', os.sep))
        day = r['jpeg'].split('/')[1]
        name = f"{day}_{os.path.basename(r['jpeg'])}"
        rel = release_flags(name.replace('_resized.jpg', '.jpg'))
        needs = (rel is None) or bool(rel) or any(f.startswith(('PEOPLE_VISIBLE', 'TEXT_OR_LOGO')) for f in r['flags'])
        out = os.path.join(review if needs else ready, name)
        if SKIP_EXISTING and any(os.path.exists(os.path.join(d_, name)) for d_ in (ready, review)):
            manifest.append({**r, 'status': 'SKIPPED: already produced from the local source', 'out': ''})
            skipped += 1
            continue
        try:
            with Image.open(src) as _im:
                mp = _im.width * _im.height / 1e6
            resized_to = None
            data = open(src, 'rb').read()
            if mp > 100:
                data, rs = resize_to_limit(data)
                resized_to = f'{rs[0]}x{rs[1]}'
                name = name.replace('.jpg', '_resized.jpg')
            elif mp < 4:
                manifest.append({**r, 'status': f'SKIPPED: {mp:.1f} MP is below Adobe Stock minimum (4 MP)', 'out': ''})
                skipped += 1
                continue
            caption = r['title'] + '.'
            new = rewrite(data, r['title'], r['keywords'], caption)
            tmp = out + '.part'
            open(tmp, 'wb').write(new)
            errs = verify(src, tmp, r['title'], r['keywords'], resized=resized_to is not None)
            if errs:
                os.remove(tmp)
                bad += 1
                manifest.append({**r, 'status': 'FAILED: ' + '; '.join(errs), 'out': ''})
            else:
                os.replace(tmp, out)
                # a file lives in exactly one of ready / _needs_review: drop the stale counterpart so routing changes move it
                other = os.path.join(ready if os.path.dirname(out) == review else review, os.path.basename(out))
                if os.path.exists(other):
                    os.remove(other)
                manifest.append({**r, 'status': 'ok', 'out': out})
        except Exception as e:
            bad += 1
            for _t in (out + '.part',):
                if os.path.exists(_t):
                    os.remove(_t)
            manifest.append({**r, 'status': f'ERROR {type(e).__name__}: {str(e)[:80]}', 'out': ''})
        if i % 100 == 0 or n:
            print(f'{i}/{len(todo)}  skipped(out of Adobe size range) {skipped}  failed {bad}  {time.time() - t0:.0f}s', flush=True)
    mpath = os.path.join(DST, ('test_' + MANIFEST_NAME) if n else MANIFEST_NAME)
    with open(mpath, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['status', 'output_jpeg', 'source_jpeg', 'paired_raw', 'title', 'n_keywords', 'keywords', 'obs_ids', 'flags'])
        for m in manifest:
            w.writerow([m['status'], m['out'], os.path.join(SRC, m['jpeg']), os.path.join(SRC, m['raw']), m['title'], len(m['keywords']),
                        ', '.join(m['keywords']), ';'.join(map(str, m['obs'])), ' | '.join(m['flags'])])
    ok = sum(1 for m in manifest if m['status'] == 'ok')
    print(f'done: ok {ok}, skipped {skipped}, failed {bad}, manifest {mpath}', flush=True)


if __name__ == '__main__':
    main()
