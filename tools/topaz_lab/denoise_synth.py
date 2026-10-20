"""Synthetic ground-truth test of dnt_beta (GPU, onnxruntime CUDA, no CPU fallback).
Clean ISO-64 crops -> add known noise -> denoise -> score vs the clean original. Also: harm on the clean crop itself."""
import sys, os, json, time
sys.path.insert(0, r'D:\bio-cache\photos\topaz-lab')
import numpy as np, lab

N = int(os.environ.get('LAB_N', '6'))
crops = lab.pick_crops(N, size=512, seed=2)
s = lab.session('dnt_beta-v5-fp32-512x512-ox')


def dn(tile, norm, p0, p1=0.0):
    t = tile * 2 - 1 if norm == 'pm1' else tile
    o = s.run(None, {'netInput': t[None].astype(np.float32), 't_param': np.array([p0], np.float32),
                     't_param1': np.array([p1], np.float32)})[0][0]
    return (o + 1) / 2 if norm == 'pm1' else o


def affine_match(o, ref):
    """per-channel gain/offset fit of o to ref (removes a global tone/colour shift so we score detail/noise, not tone)."""
    out = np.empty_like(o)
    for c in range(3):
        A = np.stack([o[..., c].ravel(), np.ones(o[..., c].size)], 1)
        k, b = np.linalg.lstsq(A, ref[..., c].ravel(), rcond=None)[0]
        out[..., c] = o[..., c] * k + b
    return out


rng = np.random.default_rng(7)
noises = [('clean', lambda im: im), ('gauss4', lambda im: lab.add_gauss(im, 4, rng)), ('gauss8', lambda im: lab.add_gauss(im, 8, rng)),
          ('gauss16', lambda im: lab.add_gauss(im, 16, rng)), ('gauss32', lambda im: lab.add_gauss(im, 32, rng)),
          ('shot400', lambda im: lab.add_shot(im, 400, 2, rng)), ('shot100', lambda im: lab.add_shot(im, 100, 3, rng)),
          ('shot25', lambda im: lab.add_shot(im, 25, 4, rng))]
rows = {}
imgs = [lab.load(f)[y:y + 512, x:x + 512] for f, y, x, lv in crops]
print('crops', [os.path.basename(c[0]) for c in crops])
for nname, nf in noises:
    noisy = [np.float32(nf(im)) for im in imgs]
    base_p = np.mean([lab.psnr(n, im) for n, im in zip(noisy, imgs)]); base_s = np.mean([lab.ssim(n, im) for n, im in zip(noisy, imgs)])
    rows[(nname, 'noisy-baseline', None)] = (base_p, base_s, None, base_p)
    for norm in ('01', 'pm1'):
        for p0 in (0.0, 0.25, 0.5, 0.75, 1.0):
            P, S, L, PA = [], [], [], []
            for n, im in zip(noisy, imgs):
                o = dn(n, norm, p0)
                P.append(lab.psnr(o, im)); S.append(lab.ssim(o, im)); L.append(lab.lap_var(o) / lab.lap_var(im)); PA.append(lab.psnr(affine_match(o, im), im))
            rows[(nname, norm, p0)] = (np.mean(P), np.mean(S), np.mean(L), np.mean(PA))
    print(nname, 'baseline PSNR %.2f SSIM %.4f' % (base_p, base_s), flush=True)
print()
print('%-9s %-5s %-5s | PSNR  dPSNR  PSNR(tone-matched)  SSIM   lapvar/clean' % ('noise', 'norm', 'tp'))
for (nname, norm, p0), (p, ss, l, pa) in rows.items():
    if norm == 'noisy-baseline': continue
    b = rows[(nname, 'noisy-baseline', None)][0]
    print('%-9s %-5s %-5s | %5.2f %+5.2f   %5.2f            %.4f  %s' % (nname, norm, p0, p, p - b, pa, ss, '%.2f' % l))
json.dump({'|'.join(map(str, k)): v for k, v in rows.items()}, open(r'D:\bio-cache\photos\topaz-lab\denoise_synth.json', 'w'), default=float)
