# bio pipeline

Photo tagging and identification, automated up to the two uploads that stay manual (Adobe Stock and iNaturalist).

```
python pipeline/run.py status     # what exists, what is waiting on a human
python pipeline/run.py plan       # stages + preflight result
python pipeline/run.py run [--stages a,b] [--force] [--dry-run]
```

## What is automatic

| Stage | Does | Notes |
|---|---|---|
| `inat_sync` | authenticated iNaturalist export, incl. obscured locations | credential file read by the stage, never printed; output stays in the data dir, not in git |
| `nas_match` | match iNat photos to existing JPEG exports by 32x32 image correlation (>= 0.95, aspect-gated) | read-only on the NAS; per-day, resumable |
| `nas_stub`, `taxa` | formats + taxonomy with English common names | public iNat API |
| `preview_*` | titles (<70 chars) and up to 49 keywords, buyer-first order | no coordinates, no iNat IDs, no brands, commas dropped |
| `package_*` | copies of the existing exports with IPTC + XMP title/keywords, GPS removed, size limits enforced | pixels byte-identical (verified); oversize files resized into their own file |
| `screen` | local vision model screens every copy for people, faces, text, brands, property + descriptive tags | model picked by capability, never by name; flagged files go to `_needs_review` |
| `aggregate`, `review` | per-observation tags; the review page data | |

## What stays manual (the gates)

1. **Release review**: only a person can judge a release. The pipeline produces a short list, not a decision.
2. **Adobe Stock upload**: the contributor portal; files in `ready/` already carry their metadata.
3. **iNaturalist upload / ID confirmation**: new photos are uploaded and identified by you.

## Preflight guards (learned from real failures)

* **Commit headroom >= 25 GB and <= 85% of the Windows commit limit.** Commit charge, not free RAM, is what makes Rust programs
  abort with `0xc0000409`; three LightCraft aborts in one day were this.
* **No pile of orphaned `git.exe` processes.** Claude Code's untracked-file listing in a very large home-folder repo leaked 49 of them
  (94 GB of commit, 56 CPU-hours).
* Disk headroom on the export drive, NAS mapped, Ollama reachable, iNat API reachable.
* Long stages are resumable and checkpoint (per day, per 25 files, per file).

## Rules the stages follow

* Never upload anywhere; never delete or modify originals; outputs are new files.
* Oversize (> 100 MP) and undersize (< 4 MP) files are skipped or resized with the reason recorded in the manifest.
* An unscreened copy is held out of `ready/` until the vision pass has seen it.
* Calibrate a matcher with simulated-absent negatives, not just positives (an embedding "99% precision" turned out to be 75-83%).

## Layout

`stages/` are the working scripts. Paths are workstation defaults; `BIO_DATA`, `BIO_EXPORT`, `BIO_SRC`, `BIO_NAS`, `BIO_PY` override.
Data (inventories, iNat export with private coordinates, caches, venv) lives outside the repo.

## Roadmap

* watch mode for new files under the library root (scheduled by you; the pipeline does not install tasks itself)
* iNat upload queue: unmatched new photos with identification suggestions from embeddings + vision
* fold the stage scripts into the `bio_automation` package behind one config file
