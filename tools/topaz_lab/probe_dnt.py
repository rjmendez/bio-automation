import sys, time, os
sys.path.insert(0, r'D:\bio-cache\photos\topaz-lab')
import numpy as np, lab

crops = lab.pick_crops(3, size=1024, seed=1)
print('crops:', [(os.path.basename(c[0]), round(c[3], 5)) for c in crops])
f, y, x, lv = crops[0]
img = lab.load(f)[y:y + 512, x:x + 512]
s = lab.session('dnt_beta-v5-fp32-512x512-ox')
print('providers', s.get_providers())


def run(tile, norm, p0, p1):
    t = tile * 2 - 1 if norm == 'pm1' else tile
    o = s.run(None, {'netInput': t[None].astype(np.float32), 't_param': np.array([p0], np.float32),
                     't_param1': np.array([p1], np.float32)})[0][0]
    return (o + 1) / 2 if norm == 'pm1' else o


for norm in ('01', 'pm1'):
    for p0 in (0.0, 0.5, 1.0):
        t0 = time.time(); o = run(img, norm, p0, 0.0); dt = time.time() - t0
        print(f'norm={norm} t_param={p0}: out {o.min():.2f}..{o.max():.2f}  PSNR vs input {lab.psnr(o, img):.2f} dB  '
              f'SSIM {lab.ssim(o, img):.4f}  lapvar ratio {lab.lap_var(o) / lab.lap_var(img):.3f}  {dt * 1000:.0f} ms')
