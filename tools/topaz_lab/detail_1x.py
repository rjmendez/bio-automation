"""1x 'detail' pass = ghqv2_ldn 2x then BOX-down to 1x, on real clean crops. Measures drift from the original (any change is invented or removed detail)
and writes a side-by-side sheet for visual judgement. GPU only."""
import sys, os
sys.path.insert(0, r'D:\bio-cache\photos\topaz-lab')
import numpy as np, lab
from PIL import Image

crops = lab.pick_crops(6, size=512, seed=9)
imgs = [lab.load(f)[y:y + 512, x:x + 512] for f, y, x, lv in crops]
s = lab.session('ghqv2_ldn-v1-fp32-128x128-2x-ox')


def up2(a, pn, pb, pc):
    def fn(t):
        return s.run(None, {'netInput': t[None].astype(np.float32), 'pn': np.array([pn], np.float32),
                            'pb': np.array([pb], np.float32), 'pc': np.array([pc], np.float32)})[0][0]
    return lab.run_tiled(a, fn, tile=128, overlap=32, scale=2)


def down2(a):
    im = Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8))
    return np.asarray(im.resize((a.shape[1] // 2, a.shape[0] // 2), Image.BOX), np.float32) / 255.0


os.makedirs(r'D:\bio-cache\photos\eval-detail', exist_ok=True)
for (pn, pb, pc) in ((0, 0, 0), (0.3, 0, 0.3), (0, 0.3, 0.3)):
    P, S, L = [], [], []
    outs = []
    for im in imgs:
        o = np.clip(down2(up2(im, pn, pb, pc)), 0, 1); outs.append(o)
        P.append(lab.psnr(o, im)); S.append(lab.ssim(o, im)); L.append(lab.lap_var(o) / lab.lap_var(im))
    print('1x detail pass pn%s pb%s pc%s: PSNR vs original %.2f  SSIM %.4f  lapvar ratio %.2f' % (pn, pb, pc, np.mean(P), np.mean(S), np.mean(L)), flush=True)
    if (pn, pb, pc) == (0.3, 0, 0.3):
        for k in range(3):
            sheet = np.concatenate([imgs[k][:, :, :], outs[k]], 1)
            lab.save(sheet, rf'D:\bio-cache\photos\eval-detail\sheet_{k}_original_left_detail_right.jpg', 95)
print('sheets in D:\\bio-cache\\photos\\eval-detail')
