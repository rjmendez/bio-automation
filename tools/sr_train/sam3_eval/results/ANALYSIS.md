# SAM3 vs Topaz `s_mask_l_flexible` -- 50-photo real spider/arachnid batch (2026-10-23)

Sample: 50 real photos, one per distinct iNat taxon, **randomly sampled** (seed 20261023)
from 245 disk-verified Arachnida photos resolved via the pipeline's own authoritative
`manifest.csv`/`manifest_nas.csv` (obs_id -> output_jpeg) cross-referenced against
`inat_observations.json`'s `taxon.iconic == "Arachnida"`. Covers 87 distinct taxa: true
spiders (orb weavers, jumping spiders, cobweb/widow spiders, fishing spiders, crab
spiders, cellar spiders, sheet weavers, sac spiders), plus non-spider arachnids for
contrast (ticks, mites, harvestmen/Opiliones).

**Correction on method**: a first 30-photo attempt accidentally selected taxa
alphabetically (A-F only) due to a truncation bug in the resolver -- caught before
drawing any conclusion from it, archived at `../results_run1_biased_sample/` for
provenance, not used below.

## Aggregate result

| | count | rate |
|---|---|---|
| Topaz mask degenerate (>85% of frame flagged "subject" -- a failed mask) | 11/50 | 22% |
| Topaz hard errors | 0/50 | -- |
| SAM3 found the subject ("spider" or "arachnid" prompt, confidence>0.5) | 42/50 | 84% |
| SAM3 hard errors | 0/50 | -- |

| outcome | count |
|---|---|
| Both worked | 35 |
| **SAM3 worked, Topaz failed** | **7** |
| Topaz worked, SAM3 missed | 4 |
| Both failed | 4 |

(Full per-photo table: `REPORT.md` in this directory.)

## What this shows

**Topaz's degenerate-mask failure is not limited to the web-heavy case found earlier --
it recurs on 22% of a random sample, including non-web subjects**: a jumping spider
(Salticidae, 1.000), a ground crab spider (Xysticus fraternus, 0.856), a black widow
(Latrodectus variolus, 1.000), a nursery web spider (Pisaurina mira, 0.860), a cellar
spider (Pholcus manueli, 0.992), a harvestman (Phalangiidae, 0.896), plus the original
orb-weaver-on-web case (Leucauge venusta, 1.000) confirmed again on re-test. This is a
real, broad reliability problem in the current production masking path, not a single
edge case.

**Where SAM3 beats Topaz** (7 cases): every one is a case where Topaz produced a useless
all-frame mask and SAM3 found a confident (0.68-0.98), tightly-scoped detection instead --
spanning orb weaver, crab spider, widow, nursery web spider, cellar spider, and a
harvestman.

**Where SAM3 missed with Topaz still reasonable** (4 cases): all are either harvestmen
(Leiobuninae, Leiobunum vittatum, Leiobunum) or a generic family-level label ("Araneae" =
literally "Spiders") rather than a specific species -- worth a closer look at whether
those specific frames are unusually hard (small/obscured subject) rather than a concept
gap; not yet investigated further.

**Both failed** (4 cases): Phalangioidea, Phalangium (harvestmen), Salticidae (a jumping
spider -- a genuine SAM3 miss, not excusable as "not really a spider"), and
Demodex folliculorum (a microscopic mite).

**A real, specific SAM3 limitation found**: harvestmen and mites correctly score ~0 on
"spider" (taxonomically correct -- they aren't spiders), but *also* score ~0 on
"arachnid" in every case tested here -- SAM3's "arachnid" concept appears spider-biased,
not covering the broader order the way a biologist would use the word. The one exception:
a tick (Dermacentor variabilis) scored 0.816 on "arachnid" -- ticks are recognized as
arachnids but harvestmen and mites are not, in this model's vocabulary.

## Honest limitations

- The ">85% of frame" degenerate threshold is a weak proxy -- it only catches Topaz's
  *extreme* failure mode (near-total whiteout), confirmed by direct visual check on one
  case (the original Leucauge venusta web photo). It does NOT measure Topaz's more
  moderate mask-quality problems (partial shadow inclusion, eroded leg tips) that
  `preclean.py`'s own docstring documents -- the real Topaz failure rate on "does the
  mask tightly and correctly bound just the subject" is very likely higher than 22%, not
  measured here.
- SAM3's "confidence>0.5, box present" is similarly a coarse proxy -- confirmed visually
  accurate on only one case (the original spider-on-web test), not individually verified
  for all 42 hits in this batch.
- No wall-clock/VRAM cost comparison. SAM3 is a far heavier model (ViT-based detector,
  ~5GB install, 3.45GB checkpoint) vs. Topaz's tiny 320x320 CNN; per-image latency with
  SAM3 was fast once warm (<1s/prompt), but model load and the one-time environment setup
  (torch, triton-windows, train/dev/notebooks extras) took real time this session.
- Nothing wired into production. This remains evidence for a decision, not a decision.

## Artifacts

- `../spider_set.json` -- the 50 resolved photos (obs_id, taxon, path).
- `topaz/_summary.json`, `sam3/_summary.json` -- raw per-image results.
- `merged.json` -- the joined table behind this report.
- `../results_run1_biased_sample/` -- the discarded, alphabetically-biased first attempt
  (kept for provenance, not evidence).
