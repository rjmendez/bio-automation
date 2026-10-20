# topaz_lab

GPU-only (onnxruntime CUDA, no CPU fallback) experiments with the owner's Topaz models (read from C:\TopazModels, never copied).
Scores outputs against synthetic ground truth. Working files go under D:\bio-cache\photos.

- `lab.py`: session loader (fails loudly if CUDA is not active), tiled runner (symmetric window), PSNR/own SSIM, noise helpers.
- `probe_dnt.py`: which input range dnt_beta expects ([-1,1]).
- `denoise_synth.py`: clean ISO-64 crops + known noise -> denoise -> score vs clean. Result: helps only when noise PSNR < ~30 dB.

onnxruntime 1.31 needs CUDA 13 DLLs (see lab.py); the CUDA 12 pip nvidia packages do not work.
