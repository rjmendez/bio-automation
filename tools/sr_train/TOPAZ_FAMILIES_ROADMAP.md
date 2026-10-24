# Topaz Photo AI Model Families — Research + Benchmark Roadmap

Session: 2026-10-23. Worktree: `D:\dev\bio-sr-train` (branch `feat/custom-sr-model`).
Scope: five unused Topaz Photo AI model families in `C:\TopazModels`, evaluated as
candidates to improve or replace what `bio_automation`'s `enhance()` pipeline currently
wires in. Nothing here is wired into `D:\apps\bio-automation` — that stays a separate,
deliberate decision per this project's standing rule.

Harness: `tools/sr_train/topaz_bench.py` + `tools/sr_train/topaz_bench_config.json`
(both in this worktree, untracked, not yet committed). Logic ported — not imported —
from the production `bio_automation/topaz.py` (`ModelCache`, `_run_tiled`, `_cosine_window`,
`_laplacian_var`, `_centre_gray`, `_overshoot_fraction`); that file was read-only reference
and was never edited.

---

## 1. raw_denoise (dnt_beta, draw_linear, drw_native, drw_standard)

**What it is (confidence: high):** Topaz's RAW-sensor denoisers — small (~tens of MB)
OpenVINO/ONNX CNNs, one family tuned for sRGB-domain denoise (dnt_beta, draw_linear,
3-channel, 512×512, 1×) and one tuned for packed raw-Bayer-domain denoise+demosaic+2×
(drw_native, drw_standard — 4-channel `[512,512,4]` in, `[1024,1024,3]` out; discovered
while building the harness, not predicted by the prior research pass).

**Benchmarked:** Yes. `dnt_beta-v5` (fp32, OpenVINO) run against the real held-out
NEF/Enhanced-NR-DNG set via `topaz_bench.py --family raw_denoise --task denoise`.
n=34 usable pairs (6 bit-identical skips, 0 decode errors) out of 40 requested.
Result: `tools/sr_train/results/dnt_beta_v5_fp32_ov_denoise.json`:

| metric | model | plain decode |
|---|---|---|
| PSNR (dB) | 40.141 | 55.956 |
| LPIPS | 0.163 | 0.006 |
| DISTS | 0.071 | 0.0007 |
| wins | 0 / 34 | — |

Delta: **-15.8 dB**, 0/34 wins, LPIPS ~27× worse, DISTS ~100× worse. A smaller n=3 smoke
run earlier in the session (`-12.7dB`) showed the same direction at lower confidence; the
n=34 run is the number that counts.

**Verdict: DROP (dnt_beta-v5).** Decisively falsified — same failure signature as ghqv2
and NAFNet-SIDD this session (destroys real sharp detail rather than removing noise).
Do not wire into `enhance()`.

**Not yet run, still open:** `dnt_beta-v5-fp16` (precision check on an architecture
already failed at fp32 — low expected value), `draw_linear-v1` fp32/fp16 (a genuinely
different architecture — real information value), and `drw_native-v1`/`drw_standard-v1`
(onnxruntime, fp32) — these are 4-channel packed-Bayer models and need a
raw-Bayer-mosaic extraction step (e.g. `rawpy.raw_image_visible`) that does not exist in
this harness yet; `task_denoise()` correctly refuses to feed them 3-channel sRGB rather
than silently mis-running them. **Honest reason these are unresolved, not dropped:** the
one `draw_linear`/`drw_*` run queued got stalled by machine-wide CPU contention
(100% load / 925 concurrent processes from other sessions) before it produced output —
not evidence against the models, just not yet measured.

---

## 2. alt_upscale (gclc-v1, ggnv2-v3, ggi-v1/v2, gmp-v1/v2)

**What it is (confidence: high for gclc/ggnv2, medium for ggi/gmp):** gclc-v1 is a fixed
Interpolate+blur+clamp graph (712 bytes of "weights" — not a learned model at all).
ggnv2-v3 is architecturally identical (same op counts, same 109,321,828-byte weight blob)
to ghqv2-v1, the already-measured Topaz default. ggi/gmp are a different, unpublished
shallow PReLU-residual CNN (31 conv layers); their v2 weights only exist as ONNX, not
OpenVINO IR.

**Benchmarked:** No.

**Verdict: SKIP (skip_low_value) — not dropped after testing, dropped before testing.**
One-line reason: ghqv2 (same vendor, and for ggnv2 the *same skeleton*) already lost to
bicubic by 5dB this session, and HAT-S already beats bicubic by +1.35dB/39-40 wins with a
published, Apache-2.0 architecture and a proven LoRA path — spending effort on four more
closed, architecture-opaque Topaz CNNs (one not even a neural net, one a literal twin of
a measured failure, two with no track record) has no expected value against continuing
the HAT-S LoRA line. Logged in `topaz_bench_config.json`'s `skipped` list, not silently
dropped.

---

## 3. auto_classifiers (noise_det-v1, liting_det-v1, gendet-v1)

**What it is (confidence: high for noise_det/liting_det behavior, low for gendet's 18
output classes):** Topaz Autopilot's per-image routing scalars. noise_det-v1 (5.8MB) and
liting_det-v1 (1.4MB) are tiny ONNX nets; a quick synthetic-image probe this session
(not a full benchmark) showed noise_det-v1 behaving like a monotonic noise-intensity
regressor and liting_det-v1's second output like a monotonic overexposure score.
gendet-v1 (115 conv layers + SE blocks, 18 undocumented logits) was inconclusive to probe
with synthetic images.

**Benchmarked:** Partially — infrastructure only. The harness's `characterize` task
runner was smoke-tested end-to-end on `gendet-v1-fp32-256x256-1x-ox` with n=5 real Stock
JPEGs, producing a valid stats JSON. **No family-scale run against real labeled/triaged
photos was completed this session** — that's the actual test this family needs (does
noise_det-v1's scalar correlate with real noisy-vs-clean photos; does liting_det-v1's
2nd output correlate with real under/over-exposed photos) and it did not happen yet.

**Verdict: NEEDS MORE WORK.** The cheap, concrete members (noise_det-v1, liting_det-v1)
are free to run, already shown sane on synthetic probes, and measure something genuinely
different from what's already in the pipeline (blur via Laplacian variance; general
zero-shot content/degradation via DA-CLIP) — worth the small next step of running them
against real triaged photos before any adopt/drop call. gendet-v1 is lower priority:
DA-CLIP already covers "what kind of content/defect is this" more flexibly and
interpretably than an 18-way opaque head, so it's deprioritized within the family but not
dropped outright (it's free to run; the smoke test already proves it loads and executes).

---

## 4. face_recovery (gfrf-v2, gfrg-v3, gfp-l, gfp-s)

**What it is (confidence: high these are wrappers around known open models):** gfrf-v2 ≈
GFPGAN (TencentARC, Apache-2.0), gfrg-v3 ≈ CodeFormer (S-Lab, non-commercial license),
gfp-l/gfp-s ≈ RetinaFace face detection (MIT).

**Benchmarked:** No.

**Verdict: DROP (skip_irrelevant).** One-line reason: `bio_automation`'s source tree has
zero face/portrait/person code paths, and the real photo library (`F:\Bio`) is entirely
macro insect/wildlife field-session folders (`<year>/<YYYY-MM-DD>` shoot days) — there is
no mechanism in the pipeline that would ever route a photo to a face-restoration model,
and the restoration premise itself (face landmark priors, face-codebook generative priors)
doesn't apply to arthropod subjects regardless of wiring. Checked via grep of the source
tree and inspection of the library folder structure, not assumed.

---

## 5. segmentation (samenc-v1, samdec-v1)

**What it is (confidence: high — strong circumstantial evidence, not a cited Topaz
writeup):** almost certainly a vanilla Meta Segment Anything Model wrapper — the
"samenc"/"samdec" naming matches the community `samexporter` convention, 1024×1024 is
SAM's fixed input size (every other Topaz model here uses 512/256/320), and the
179MB-encoder-vs-10MB-decoder size ratio matches SAM's known ViT-B-backbone-vs-tiny-decoder
shape. Correction to the task's framing: the encoder is fp32-only (`samenc-v1-fp32-1024x1024-ov`,
179MB, OpenVINO-only, no ONNX variant found), not fp16.

**Benchmarked:** No.

**Verdict: SKIP (skip_low_value), log as a future idea.** One-line reason: if it's
vanilla SAM there's no Topaz-proprietary training to measure against upstream SAM, and
`bio_automation` has no masked/subject-isolated enhancement step yet to attach a
segmentation model to — standing up a mask-IoU eval axis (which doesn't exist in this
project's harness) for a feature that doesn't exist yet is premature infrastructure. If a
masked-enhancement step (e.g. sharpen-only-the-subject) is ever built, go straight to
upstream MobileSAM (Apache-2.0, ~10MB, near-identical mask quality) or full SAM/SAM2
(Apache-2.0) rather than reverse-engineering Topaz's prompt-wrapper conventions. Note in
passing: a separate, smaller `smp_flexible-v2`/`sms_new-v3`/`s_mask_l_flexible-v2` family
(320×320) also exists in `C:\TopazModels` and looks like Topaz's own lighter subject-mask
net, distinct from the SAM pair — out of scope here, worth knowing it exists if "subject
mask" comes up again.

---

## Prioritized next steps

1. **Continue the HAT-S LoRA fine-tune line (upscale).** This remains the only
   from-scratch-vs-pretrained approach that has won in this whole project (zero-shot
   HAT-S: +1.35dB PSNR, 39/40 wins, better LPIPS/DISTS/deployment-sharpness vs bicubic;
   LoRA weights already saved at `D:\bio-cache\sr-train\hat_s_lora_r8.pt`). Highest
   expected value of anything in scope right now — keep investing here before any more
   Topaz variants.
2. **Finish the raw_denoise family with real numbers, not guesswork.** Rerun
   `topaz_bench.py` for `draw_linear-v1` (fp32/fp16 — a genuinely different architecture
   than the already-failed dnt_beta, so still informative) when CPU is available; lower
   priority: `dnt_beta-v5-fp16` (precision check on an architecture that already failed).
   Both are single CLI invocations against the existing harness and dataset.
3. **Build the raw-Bayer-mosaic extraction path**, then run `drw_native-v1`/`drw_standard-v1`
   (onnxruntime, 4-channel packed-Bayer input) — a structurally distinct model class from
   dnt_beta/draw_linear, so dnt_beta's failure is not evidence against it either way.
4. **Run `auto_classifiers`' noise_det-v1 and liting_det-v1 against real triaged photos**
   (noisy-vs-clean; under/over-exposed vs normal) to decide whether they add signal
   beyond the existing Laplacian-variance blur gate and DA-CLIP. Cheap (CPU, tiny ONNX
   models, harness already built) — the main unfinished piece of this family.
5. **If time remains, a cheap no-priority sanity check only:** one quick gclc-v1 or
   ggnv2-v3 run, strictly as idle-compute filler — not worth displacing effort from 1-4.
6. **Do not** benchmark face_recovery or segmentation further, and do not revisit
   alt_upscale beyond item 5, absent new evidence changing the reasoning above.
7. **No production wiring decision from this document.** Any adopt call (most plausibly
   HAT-S-LoRA for upscale, conditionally noise_det/liting_det for routing) goes through
   this project's existing "never swap a model into production silently" process as its
   own separate step, after the above finishes.

All work in this document ran on local CPU (OpenVINO default device) or the desktop GPU
inside `D:\dev\bio-sr-train` (worktree `feat/custom-sr-model`); no AWS spend, no EC2
instances touched, no edits to `D:\apps\bio-automation`.
