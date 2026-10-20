# topaz_lab

GPU-only (onnxruntime CUDA, no CPU fallback) experiments with the owner's Topaz models (read from C:\TopazModels, never copied).
Scores outputs against synthetic ground truth. Working files go under D:\bio-cache\photos.

- `lab.py`: session loader (fails loudly if CUDA is not active), tiled runner (symmetric window), PSNR/own SSIM, noise helpers.
- `probe_dnt.py`: which input range dnt_beta expects ([-1,1]).
- `denoise_synth.py`: clean ISO-64 crops + known noise -> denoise -> score vs clean. Result: helps only when noise PSNR < ~30 dB.

onnxruntime 1.31 needs CUDA 13 DLLs (see lab.py); the CUDA 12 pip nvidia packages do not work.

## Measured results (2026-10-20, GPU, 6-8 clean ISO-64 crops, own PSNR/SSIM)
- `dnt_beta` (denoise): only helps when noise PSNR < ~30 dB (sigma >= ~10-16/255). On clean/low-noise input it re-synthesises texture and LOSES quality (sigma4: -5.4 dB). Takes input in [-1,1].
- `noise_det` (Topaz noise detector): monotonic up to sigma16, non-monotonic above; on all 1,657 real frames the max reading (0.67) is below its reading for sigma8 (0.72): no real frame qualifies for denoise.
- `ghqv2`/`ghqv2_ldn` 2x upscale (input range [0,1]; [-1,1] gives garbage): PSNR 27.0-27.3 vs Lanczos 32.4: worse than plain resizing, 2.5x edge energy (invented detail).
- 1x detail pass (ghqv2_ldn 2x then BOX-down, pn.3 pb0 pc.3): 33.6 dB / SSIM .957 vs original, +59% edge energy; visually crisper on sheets in D:\bio-cache\photos\eval-detail.
- `slb` Refocus sharpen (CPU/OpenVINO only): not worth applying (halos, invented texture); see audit.
