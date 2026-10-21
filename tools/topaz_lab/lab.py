"""Topaz model lab: GPU (onnxruntime CUDA) tiled inference + synthetic-degradation scoring.
Models and their extracted cache stay on C:\\TopazModels; everything written here is under D:."""
import os, glob, time
os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
import numpy as np
# onnxruntime 1.31's CUDA provider is a CUDA 13 build (needs cudart64_13 / cublas64_13), NOT the CUDA 12 pip nvidia packages.
for _d in (r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\x64',
           r'C:\Program Files\NVIDIA\CUDNN\v9.13\bin\13.0'):
    if os.path.isdir(_d):
        os.add_dll_directory(_d)
        os.environ['PATH'] = _d + os.pathsep + os.environ.get('PATH', '')   # cuDNN loads its sub-libraries via PATH
import onnxruntime as ort
from PIL import Image
from scipy.ndimage import gaussian_filter, uniform_filter

SRC = os.environ.get('LAB_SRC', r'F:\Bio-export\inat-raws-q95-auto')      # the q95 renders the lab reads; LAB_SRC points it at another render set
                                                                           # (the pre-2026-10-21 non-auto-retouched set moved to NAS: \\172.16.100.5\Shared\bio-automation-archive\inat-raws-q95)
GPU = int(os.environ.get('LAB_GPU', '0'))      # PCI order: 0 = RTX 2080 Ti (more free memory), 1 = RTX 4070 Ti
_sess = {}


def session(stem):
    if stem not in _sess:
        onnx = glob.glob(r'C:\TopazModels\%s\*.onnx' % stem)
        if not onnx:
            raise FileNotFoundError(f'no extracted .onnx for {stem} in C:\\TopazModels')
        so = ort.SessionOptions(); so.log_severity_level = 3
        # GPU ONLY (owner rule 2026-10-20: no CPU processing). No CPU fallback: if CUDA cannot load, fail loudly.
        s = ort.InferenceSession(onnx[0], so, providers=[('CUDAExecutionProvider', {'device_id': GPU})])
        if s.get_providers()[0] != 'CUDAExecutionProvider':
            raise RuntimeError(f'{stem}: CUDA provider not active, got {s.get_providers()}; refusing to run on CPU')
        _sess[stem] = s
    return _sess[stem]


def window(n, eps=1e-3):
    w = np.sin(np.linspace(0, np.pi, n + 2)[1:-1]) ** 2 + eps
    return np.outer(w, w).astype(np.float32)[..., None]


def run_tiled(img, fn, tile=512, overlap=64, scale=1):
    """img HxWx3 float32; fn(tile HxWx3)->HxWx3 (scale x). Symmetric window blend, reflect padded."""
    H, W, _ = img.shape
    stride = tile - overlap
    ph = (-(H - tile) % stride) if H > tile else tile - H
    pw = (-(W - tile) % stride) if W > tile else tile - W
    p = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode='reflect')
    PH, PW, _ = p.shape
    out = np.zeros((PH * scale, PW * scale, 3), np.float32); acc = np.zeros((PH * scale, PW * scale, 1), np.float32)
    win = window(tile * scale)
    for y in range(0, PH - tile + 1, stride):
        for x in range(0, PW - tile + 1, stride):
            r = fn(p[y:y + tile, x:x + tile]).astype(np.float32)
            out[y * scale:(y + tile) * scale, x * scale:(x + tile) * scale] += r * win
            acc[y * scale:(y + tile) * scale, x * scale:(x + tile) * scale] += win
    return (out / acc)[:H * scale, :W * scale]


def load(path):
    return np.asarray(Image.open(path).convert('RGB'), np.float32) / 255.0


def save(a, path, quality=95):
    Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)).save(path, 'JPEG', quality=quality, subsampling=0)


def psnr(a, b):
    a = np.clip(a, 0, 1); b = np.clip(b, 0, 1)
    m = float(np.mean((a - b) ** 2))
    return 99.0 if m < 1e-12 else 10 * np.log10(1.0 / m)


def ssim(a, b, sigma=1.5):
    """Own SSIM (gaussian window), mean over channels; not skimage."""
    a = np.clip(a, 0, 1).astype(np.float64); b = np.clip(b, 0, 1).astype(np.float64)
    c1, c2 = 0.01 ** 2, 0.03 ** 2; vals = []
    for c in range(3):
        x, y = a[..., c], b[..., c]
        mx, my = gaussian_filter(x, sigma), gaussian_filter(y, sigma)
        sxx = gaussian_filter(x * x, sigma) - mx * mx; syy = gaussian_filter(y * y, sigma) - my * my; sxy = gaussian_filter(x * y, sigma) - mx * my
        vals.append(((2 * mx * my + c1) * (2 * sxy + c2) / ((mx * mx + my * my + c1) * (sxx + syy + c2))).mean())
    return float(np.mean(vals))


def lap_var(a):
    g = a.mean(-1); lap = g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(lap.var())


def add_gauss(img, sigma8, rng):
    return np.clip(img + rng.normal(0, sigma8 / 255.0, img.shape).astype(np.float32), 0, 1)


def add_shot(img, peak, read8, rng):
    """Poisson (shot) noise in linear light with `peak` photons at white, plus gaussian read noise (8-bit units), back to sRGB-ish gamma 2.2."""
    lin = np.clip(img, 0, 1) ** 2.2
    noisy = rng.poisson(lin * peak).astype(np.float32) / peak
    noisy = noisy + rng.normal(0, read8 / 255.0, img.shape).astype(np.float32) * 0.5
    return np.clip(noisy, 0, 1) ** (1 / 2.2)


def pick_crops(n, size=1024, seed=0, pool=400):
    """n high-detail crops from ISO-64 frames of different days (clean reference)."""
    rng = np.random.default_rng(seed)
    files = sorted(glob.glob(SRC + r'\2026\*\*.jpg'))
    files = [files[i] for i in rng.permutation(len(files))[:pool]]
    cands = []
    for f in files:
        im = Image.open(f)
        if im.size[0] < size or im.size[1] < size: continue
        try: iso = im.getexif().get_ifd(0x8769).get(0x8827)
        except Exception: iso = None
        if iso != 64: continue
        a = np.asarray(im.convert('L'), np.float32) / 255
        H, W = a.shape
        best = None
        for (y, x) in [(0, 0), (0, W - size), (H - size, 0), (H - size, W - size), ((H - size) // 2, (W - size) // 2)]:
            lv = lap_var(a[y:y + size, x:x + size, None].repeat(3, -1))
            if best is None or lv > best[0]: best = (lv, y, x)
        cands.append((best[0], f, best[1], best[2]))
    cands.sort(reverse=True)
    out, days = [], set()
    for lv, f, y, x in cands:
        day = f.split('\\')[-2]
        if day in days and len(days) < 8: continue
        days.add(day); out.append((f, y, x, lv))
        if len(out) == n: break
    return out
