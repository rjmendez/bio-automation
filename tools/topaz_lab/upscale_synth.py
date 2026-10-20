"""Synthetic ground truth for Gigapixel ghqv2 / ghqv2_ldn 2x (GPU only): downscale clean crops 2x, upscale back, score vs the original."""
import sys, os, json
sys.path.insert(0, r'D:\bio-cache\photos\topaz-lab')
import numpy as np, lab
from PIL import Image

crops = lab.pick_crops(6, size=1024, seed=5)
imgs = [lab.load(f)[y:y + 1024, x:x + 1024] for f, y, x, lv in crops]
print('crops', [os.path.basename(c[0]) for c in crops])


def down2(a):
    im = Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8))
    return np.asarray(im.resize((a.shape[1] // 2, a.shape[0] // 2), Image.BOX), np.float32) / 255.0


def up_pil(a, method):
    im = Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8))
    return np.asarray(im.resize((a.shape[1] * 2, a.shape[0] * 2), method), np.float32) / 255.0


def up_model(stem, a, pn, pb, pc, norm):
    s = lab.session(stem)

    def fn(tile):
        t = (tile * 2 - 1) if norm == 'pm1' else tile
        o = s.run(None, {'netInput': t[None].astype(np.float32), 'pn': np.array([pn], np.float32),
                         'pb': np.array([pb], np.float32), 'pc': np.array([pc], np.float32)})[0][0]
        return (o + 1) / 2 if norm == 'pm1' else o
    return lab.run_tiled(a, fn, tile=128, overlap=32, scale=2)


lows = [down2(im) for im in imgs]
res = {}
for name, up in (('bicubic', lambda a: up_pil(a, Image.BICUBIC)), ('lanczos', lambda a: up_pil(a, Image.LANCZOS))):
    o = [up(l) for l in lows]
    res[name] = (np.mean([lab.psnr(x, im) for x, im in zip(o, imgs)]), np.mean([lab.ssim(x, im) for x, im in zip(o, imgs)]),
                 np.mean([lab.lap_var(x) / lab.lap_var(im) for x, im in zip(o, imgs)]))
    print('%-34s PSNR %.2f  SSIM %.4f  lapvar/orig %.2f' % (name, *res[name]), flush=True)
for stem in ('ghqv2-v1-fp32-128x128-2x-ox', 'ghqv2_ldn-v1-fp32-128x128-2x-ox'):
    for norm in ('01', 'pm1'):
        for (pn, pb, pc) in ((0, 0, 0), (0, 0, 0.3), (0.3, 0, 0.3), (0, 0.3, 0.3)):
            o = [up_model(stem, l, pn, pb, pc, norm) for l in lows]
            k = f'{stem.split("-")[0]} {norm} pn{pn} pb{pb} pc{pc}'
            res[k] = (np.mean([lab.psnr(x, im) for x, im in zip(o, imgs)]), np.mean([lab.ssim(x, im) for x, im in zip(o, imgs)]),
                      np.mean([lab.lap_var(x) / lab.lap_var(im) for x, im in zip(o, imgs)]))
            print('%-34s PSNR %.2f  SSIM %.4f  lapvar/orig %.2f' % (k, *res[k]), flush=True)
json.dump(res, open(r'D:\bio-cache\photos\topaz-lab\upscale_synth.json', 'w'), default=float)
