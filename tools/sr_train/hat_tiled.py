"""Tiled HAT-S inference for full-size photos. Ported (not imported) from
bio_automation/topaz.py's _run_tiled/_cosine_window -- same windowed-blend approach
already proven in production for Topaz's sharpen()/upscale_2x(), adapted here for a
torch model feed_fn instead of an OpenVINO compiled model. 256x256 input tiles (the
size this project's whole eval/training pipeline was built and validated on; 512 OOMs
on this desktop's cards -- measured directly, see gallery_upscale_examples.py), 32px
overlap (same ~12.5% ratio as topaz.py's upscale_2x 128-tile/16-overlap), scale=2.
"""
import numpy as np
import torch


def _cosine_window(size: int) -> np.ndarray:
    t = np.linspace(0, np.pi, size + 2, dtype=np.float32)[1:-1]
    return np.sin(t) ** 2


def run_tiled(img_chw_f32: np.ndarray, tile_h: int, tile_w: int, overlap: int,
              scale: int, feed_fn) -> np.ndarray:
    """Tile *img_chw_f32* [C,H,W] through *feed_fn* with window blending.

    feed_fn(tile_chw_f32) -> [C, tile_h*scale, tile_w*scale] float32. Returns a
    float32 [C, H*scale, W*scale] array in [0,1]."""
    C, H, W = img_chw_f32.shape
    stride_h = tile_h - overlap
    stride_w = tile_w - overlap
    out_H, out_W = H * scale, W * scale

    win_h = _cosine_window(tile_h * scale)[np.newaxis, :, np.newaxis]
    win_w = _cosine_window(tile_w * scale)[np.newaxis, np.newaxis, :]
    win2d = win_h * win_w

    pad_h = max(0, tile_h - H)
    pad_w = max(0, tile_w - W)
    if pad_h or pad_w:
        img_chw_f32 = np.pad(img_chw_f32, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    _, pH, pW = img_chw_f32.shape

    output = np.zeros((C, pH * scale, pW * scale), dtype=np.float32)
    weight = np.zeros((1, pH * scale, pW * scale), dtype=np.float32)

    def _starts(total, tile, stride):
        pts = list(range(0, total - tile + 1, stride))
        if not pts or pts[-1] + tile < total:
            pts.append(total - tile)
        return pts

    y_starts = _starts(pH, tile_h, stride_h)
    x_starts = _starts(pW, tile_w, stride_w)
    n_tiles = len(y_starts) * len(x_starts)
    done = 0
    for y in y_starts:
        for x in x_starts:
            tile = img_chw_f32[:, y:y + tile_h, x:x + tile_w]
            result = np.clip(feed_fn(tile), 0.0, 1.0)
            oy, ox = y * scale, x * scale
            output[:, oy:oy + tile_h * scale, ox:ox + tile_w * scale] += result * win2d
            weight[:, oy:oy + tile_h * scale, ox:ox + tile_w * scale] += win2d
            done += 1
            if done % 50 == 0 or done == n_tiles:
                print(f"  tile {done}/{n_tiles}")

    output /= np.maximum(weight, 1e-12)
    return np.clip(output[:, :out_H, :out_W], 0.0, 1.0)


def upscale_2x_tiled(img_rgb_u8_hwc: np.ndarray, model, device, tile: int = 256, overlap: int = 32) -> np.ndarray:
    """Full-image 2x upscale. img_rgb_u8_hwc: [H,W,3] uint8. Returns [2H,2W,3] uint8."""
    img_chw_f32 = np.transpose(img_rgb_u8_hwc.astype(np.float32) / 255.0, (2, 0, 1))

    def feed(tile_chw_f32):
        t = torch.from_numpy(tile_chw_f32).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(t)[0].clamp(0, 1).cpu().numpy()
        return out

    result_chw = run_tiled(img_chw_f32, tile, tile, overlap, 2, feed)
    return (np.transpose(result_chw, (1, 2, 0)) * 255.0).round().clip(0, 255).astype(np.uint8)
