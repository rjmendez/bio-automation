"""Calibrate Topaz noise_det (GPU) against synthetic noise, then score every q95 frame. GPU inference only."""
import sys, os, json, glob, time, collections
sys.path.insert(0, r'D:\bio-cache\photos\topaz-lab')
import numpy as np, lab

s = lab.session('noise_det-v1-fp16-128x128-1x-ox')
P = 128


def det(img, n=24, seed=0):
    """median noise_det response over n random 128 px patches of an HxWx3 float image in [0,1]."""
    H, W, _ = img.shape
    r = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        y = int(r.integers(0, H - P)); x = int(r.integers(0, W - P))
        t = ((img[y:y + P, x:x + P] - 0.5) / 0.5).astype(np.float16)[None]
        out.append(float(s.run(None, {'patch0': t})[0].ravel()[0]))
    return float(np.median(out))


# ---- 1. calibration on clean crops + known noise
crops = lab.pick_crops(8, size=1024, seed=3)
imgs = [lab.load(f)[y:y + 1024, x:x + 1024] for f, y, x, lv in crops]
rng = np.random.default_rng(11)
levels = [('clean', None), ('g2', 2), ('g4', 4), ('g8', 8), ('g12', 12), ('g16', 16), ('g24', 24), ('g32', 32)]
print('synthetic calibration (median over 24 patches, mean over 8 crops, +- sd across crops):')
cal = {}
for name, sg in levels:
    v = [det(im if sg is None else np.float32(lab.add_gauss(im, sg, rng)), seed=i) for i, im in enumerate(imgs)]
    cal[name] = (float(np.mean(v)), float(np.std(v)))
    print('  %-6s %.4f +- %.4f' % (name, *cal[name]), flush=True)
for name, peak, rd in [('shot400', 400, 2), ('shot100', 100, 3), ('shot25', 25, 4)]:
    v = [det(np.float32(lab.add_shot(im, peak, rd, rng)), seed=i) for i, im in enumerate(imgs)]
    cal[name] = (float(np.mean(v)), float(np.std(v)))
    print('  %-8s %.4f +- %.4f' % (name, *cal[name]), flush=True)
json.dump(cal, open(r'D:\bio-cache\photos\topaz-lab\noise_det_cal.json', 'w'))

# ---- 2. every real frame (q95 set)
files = sorted(glob.glob(lab.SRC + r'\**\*.jpg', recursive=True))
print('scoring', len(files), 'real frames', flush=True)
rows = []; t0 = time.time()
from PIL import Image
for i, f in enumerate(files):
    im = Image.open(f)
    try: iso = im.getexif().get_ifd(0x8769).get(0x8827)
    except Exception: iso = None
    a = np.asarray(im.convert('RGB'), np.float32) / 255.0
    rows.append((os.path.basename(f), iso, det(a, n=24, seed=i)))
    if (i + 1) % 200 == 0: print(' ', i + 1, 'done', round(time.time() - t0), 's', flush=True)
json.dump(rows, open(r'D:\bio-cache\photos\topaz-lab\noise_det_frames.json', 'w'))
by = collections.defaultdict(list)
for fn, iso, d in rows: by[iso].append(d)
print('\nreal frames: noise_det by ISO (n, mean, max)')
for iso in sorted(k for k in by if k is not None):
    v = np.array(by[iso]); print('  ISO %-5s n=%-4d mean %.4f  max %.4f' % (iso, len(v), v.mean(), v.max()))
allv = np.array([d for _, _, d in rows])
for name in ('g4', 'g8', 'g12', 'g16'):
    print('frames with noise_det >= synthetic %s level (%.4f): %d of %d' % (name, cal[name][0], int((allv >= cal[name][0]).sum()), len(allv)))
