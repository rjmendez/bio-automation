# Provenance

`hat_arch.py` is vendored from [XPixelGroup/HAT](https://github.com/XPixelGroup/HAT),
licensed Apache-2.0 (confirmed via `gh api repos/XPixelGroup/HAT --jq '.license'`,
2026-10-23). The upstream file carries no per-file header; this notice exists so
the license is still explicit at the point of use, not just in this session's log.

`_compat.py` is new code written for this project (not derived from basicsr),
replacing basicsr's `ARCH_REGISTRY`/`to_2tuple`/`trunc_normal_` so loading HAT for
inference doesn't require installing the full basicsr framework.

Pretrained weights (`HAT-S_SRx2.pth`, not committed to git -- see
`D:\bio-cache\sr-train\hat-weights\`) are the official checkpoint from the HAT
README's linked Google Drive folder (file id `1Y7-3IgWfIAui9BMQIsFT9CzZXVMlx__e`),
not from `lce`'s vendored `ArturWoz/HAT-Sentinel-onnx` (an unrelated,
satellite-imagery-tuned fine-tune with no stated license -- deliberately not used,
see the license audit in this session's history).
