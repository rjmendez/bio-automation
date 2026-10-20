"""Bulk 1x detail pass over the q95 renders: ghqv2_ldn 2x (GPU, onnxruntime CUDA, NO CPU fallback) then BOX-down to 1x.

Every frame is measured and every decision is logged (csv). A frame keeps its original when a guard trips:
  PSNR vs original < MIN_PSNR (drifted too far) or edge-energy ratio > MAX_LAP (invented texture).
Outputs go to D:\\bio-cache\\photos\\enhanced (photo cache), never over the originals. EXIF and ICC are copied; saved q95 4:4:4.
Resumable: frames already in the log are skipped. Env: LAB_LIMIT (test on N frames), LAB_GPU (PCI index, default 0).
"""
import sys, os, csv, time, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import lab
from PIL import Image

OUT = os.environ.get('LAB_ENHANCED_OUT', r'D:\bio-cache\photos\enhanced')      # LAB_SRC / LAB_ENHANCED_OUT keep a second render set from mixing with the first
LOG = os.path.join(OUT, 'enhance_detail_log.csv')
STEM = 'ghqv2_ldn-v1-fp32-128x128-2x-ox'
PN, PB, PC = 0.3, 0.0, 0.3
MIN_PSNR, MAX_LAP = 30.0, 3.0
LIMIT = int(os.environ.get('LAB_LIMIT', '0')) or None

os.makedirs(OUT, exist_ok=True)
s = lab.session(STEM)


def up2(a):
    def fn(t):
        return s.run(None, {'netInput': t[None].astype(np.float32), 'pn': np.array([PN], np.float32),
                            'pb': np.array([PB], np.float32), 'pc': np.array([PC], np.float32)})[0][0]
    return lab.run_tiled(a, fn, tile=128, overlap=32, scale=2)


def box_down(a):
    h, w = a.shape[0] // 2 * 2, a.shape[1] // 2 * 2
    a = a[:h, :w]
    return a.reshape(h // 2, 2, w // 2, 2, 3).mean((1, 3))


done = set()
if os.path.exists(LOG):
    with open(LOG, newline='') as f:
        done = {r['file'] for r in csv.DictReader(f)}
new = not os.path.exists(LOG)
logf = open(LOG, 'a', newline='')
w = csv.writer(logf)
if new:
    w.writerow(['file', 'iso', 'psnr', 'ssim_center', 'lap_ratio', 'decision', 'seconds'])
files = sorted(glob.glob(lab.SRC + r'\**\*.jpg', recursive=True))
todo = [f for f in files if os.path.relpath(f, lab.SRC) not in done][:LIMIT]
print(f'{len(files)} frames, {len(done)} already logged, {len(todo)} to do', flush=True)
t_all = time.time()
for i, f in enumerate(todo, 1):
    rel = os.path.relpath(f, lab.SRC)
    t0 = time.time()
    im = Image.open(f)
    exif, icc = im.info.get('exif'), im.info.get('icc_profile')
    try: iso = im.getexif().get_ifd(0x8769).get(0x8827)
    except Exception: iso = None
    a = np.asarray(im.convert('RGB'), np.float32) / 255.0
    o = np.clip(box_down(up2(a)), 0, 1)
    if o.shape != a.shape:
        o = np.pad(o, ((0, a.shape[0] - o.shape[0]), (0, a.shape[1] - o.shape[1]), (0, 0)), mode='edge')
    p = lab.psnr(o, a)
    cy, cx = a.shape[0] // 2, a.shape[1] // 2
    c = (slice(cy - 512, cy + 512) if a.shape[0] >= 1024 else slice(None), slice(cx - 512, cx + 512) if a.shape[1] >= 1024 else slice(None))
    ss = lab.ssim(o[c], a[c])
    lr = lab.lap_var(o) / max(lab.lap_var(a), 1e-5)     # floor: near-flat frames have ~0 edge energy, the ratio is meaningless there
    keep = (p >= MIN_PSNR) and (lr <= MAX_LAP)
    if keep:
        dst = os.path.join(OUT, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        kw = {}
        if exif: kw['exif'] = exif
        if icc: kw['icc_profile'] = icc
        Image.fromarray((o * 255 + 0.5).astype(np.uint8)).save(dst, 'JPEG', quality=95, subsampling=0, **kw)
    w.writerow([rel, iso, round(p, 3), round(ss, 4), round(lr, 3), 'enhanced' if keep else 'original', round(time.time() - t0, 1)])
    logf.flush()
    if i % 25 == 0 or i == len(todo):
        print(f'{i}/{len(todo)}  {(time.time() - t_all) / i:.1f} s/frame', flush=True)
print('finished', flush=True)
