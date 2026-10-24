# SAM 3 vs. Topaz `s_mask_l_flexible` — real spider test (2026-10-23)

Test image: `F:\Bio-export\adobe-stock\rendered\2025\2025-06-10\DSC_4412.jpg` (confirmed
"Orchard Orbweaver on web", obs_id 289797733) — a genuinely hard subject-masking case:
thin wiry legs plus an elaborate web whose thin lines cross the entire 8256x5504 frame.
(First attempt grabbed the wrong `DSC_4412` -- filenames recur across shoot days; caught
by eyeballing the thumbnail before trusting any result, not before.)

## Current production approach (preclean.py's subject_mask, Topaz s_mask_l_flexible)

Ported verbatim (not imported) into `topaz_subject_mask_port.py` and run on the real
frame:

- `mask_mean = 0.994`, `mask_area_frac_gt_0.5 = 1.0` -- the model flags the **entire
  frame** as subject. Visually confirmed: the mask PNG is blank white.
- Root cause (consistent with the model's documented shadow-inclusion problem): the
  web's thin lines span edge-to-edge, so `_local_background_hue()`'s search for a
  reliable background-color reference near the detected subject region likely finds
  none, or the raw saliency map itself is already saturated everywhere.
- **Verdict: complete failure on this real, representative composition.**

## SAM 3 (ungated mirror, local checkpoint, promptable concept segmentation)

Ran `set_text_prompt()` with several concept prompts on the same frame (via
`build_sam3_image_model(checkpoint_path=...)`, `torch.autocast("cuda", dtype=bfloat16)`
required -- the checkpoint's weights are bf16 and crash without it):

| prompt | top score | instances > 0.5 | box area / frame |
|---|---|---|---|
| "spider" | 0.969 | 1 | 0.43% |
| "a spider" | 0.969 | 1 | 0.43% |
| "orb weaver" | 0.973 | 1 | 0.43% |
| "insect" | 0.965 | 1 | 0.43% |
| "bug" | 0.965 | 1 | 0.43% |
| "spider web" | 0.840 | 1 | 98.05% (correctly grabs the whole web, not the spider) |

All five spider/insect-referring prompts converge on the *same* tight box
(`[3641,2125]-[4689,3299]`), confirmed visually: a pixel-accurate crop of just the
spider's body and legs, cleanly excluding the surrounding web. Mask overlay confirmed
the silhouette matches the real animal, not a rough blob.

**Note:** an earlier mis-identified test image (wrong `DSC_4412`, actually a jelly
fungus) scored near-zero for "spider" (max 0.034) -- expected and correct behavior, not
a model failure, once the image mismatch was found.

## Conclusion

On this real, hard fixture, SAM 3's concept-prompted segmentation is decisively more
accurate than the current production masking path, which degrades to a useless
all-white mask on web-heavy compositions. This is a genuine candidate to replace (or
gate/fall back onto) `s_mask_l_flexible` in `preclean.py::subject_mask()` for
orb-weaver/web shots specifically -- not yet validated on a broader sample, and not
wired into production (deliberate, separate decision per this project's standing rule).

## Honest gaps / not yet done

- n=1 test image. No broader sample run yet (e.g. non-web spiders, other arachnids,
  insects without webs, multi-subject frames) -- do not generalize "SAM3 beats Topaz
  masking" beyond this one hard case without more data.
- No wall-clock/VRAM cost comparison (SAM3's ViT-based detector is far heavier than the
  tiny `s_mask_l_flexible` CNN -- full install pulled ~5GB of CUDA wheels + a 3.45GB
  checkpoint; real per-image latency wasn't measured here).
- Confidence threshold (0.5 default) worked cleanly here; untested on ambiguous/
  low-contrast subjects where it may need tuning.
- `.[train,dev,notebooks]` extras were installed to satisfy the package's own eager
  import chain (pycocotools, fvcore, jupyter, etc.) -- most of that is unused dead
  weight for this inference-only use case; a minimal production install would need to
  patch those imports out, not just pip-install around them.
