# Provenance

`NAFNet_arch.py`, `arch_util.py`, `local_arch.py` are vendored from
[megvii-research/NAFNet](https://github.com/megvii-research/NAFNet). The repo's
GitHub license tag reads "Other"/NOASSERTION because its LICENSE file concatenates
two blocks; read directly (`gh api repos/megvii-research/NAFNet/contents/LICENSE`,
2026-10-23) it is MIT for NAFNet's own code plus an Apache-2.0 attribution notice
for BasicSR, a dependency it builds on. Both headers are preserved in full in the
LICENSE file fetched that day (not duplicated here).

`arch_util.py`'s unused `from basicsr.utils import get_root_logger` import was
removed (only referenced in an already-commented-out block, never called) so this
file loads standalone without installing basicsr. `NAFNet_arch.py`'s two
`from basicsr.models.archs...` imports were pointed at the local `arch_util.py`/
`local_arch.py` instead, for the same reason. No other logic was changed.

Pretrained weights (`NAFNet-SIDD-width64.pth`, not committed to git -- see
`D:\bio-cache\sr-train\nafnet-weights\`) are the official SIDD-denoising checkpoint
linked from `docs/SIDD.md` (file id `14Fht1QQJ2gMlk4N1ERCRuElg8JfjrWWR`) -- the
general-restoration NAFNet trained for denoising, not `lce`'s vendored
`opencv/deblurring_nafnet` (a deblurring-specific variant; also MIT, but a
different task than ours).
