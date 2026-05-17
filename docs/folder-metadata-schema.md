# Folder and metadata schema

Goal: keep the library filesystem-first, iNaturalist-friendly, and easy to open source later.

## Principles

- Originals stay immutable.
- Raw camera files are the long-term source of truth, but only for frames worth keeping.
- Every derived file should be reproducible from the source plus metadata.
- Folder names should be human-readable first, machine-readable second.
- Metadata should live in plain text sidecars or a small local index, not a proprietary catalog.
- One observation can drive multiple downstream outputs, including iNaturalist uploads and stock-ready exports.
- Lower-resolution web exports are disposable unless you explicitly choose to keep them.

## Recommended layout

```text
F:\Bio\
  2026\
    2026-03-01\
      obs-20260301-001-prairie-warbler\
        source\
          raw\
            IMG_1234.CR2
            IMG_1235.CR2
            IMG_1236.NEF
        work\
          IMG_1234.dng
        review\
          IMG_1234.jpg
        export\
          inat\
            IMG_1234_upload.jpg
          stock\
            IMG_1234_stock.jpg
        metadata\
          observation.json
          assets.jsonl
          keywords.json
          release.json
```

## Folder meaning

| Folder | Purpose |
|---|---|
| `source/raw/` | Camera originals retained long term if selected |
| `work/` | Temporary or intermediate processing outputs |
| `review/` | Human-review copies for culling or comparison |
| `export/inat/` | Upload-ready images for iNaturalist; delete after upload if not needed |
| `export/stock/` | Stock-ready exports and derivatives; keep only if they are still useful |
| `metadata/` | Observation and per-file sidecars |

## Naming rules

- Observation folders: `obs-YYYYMMDD-###-slug`
- Keep the slug short and descriptive.
- Use the same stem for all derivatives of one image.
- Prefer lowercase, hyphen-separated names for derived artifacts.

## Metadata files

### `metadata/observation.json`

Observation-level data pulled from iNaturalist or entered locally.

```json
{
  "observation_id": 123456789,
  "obsdate": "2026-03-01",
  "taxon_name": "Prairie Warbler",
  "taxon_id": 12345,
  "place_guess": "F:\\Bio\\2026\\2026-03-01",
  "locality": "nearfield notes",
  "license": "CC BY-NC",
  "quality_grade": "research",
  "source": "iNaturalist",
  "notes": "Imported from existing observation"
}
```

### `metadata/assets.jsonl`

One row per file for file-level processing state.

```json
{"file":"source/IMG_1234.CR2","role":"original","status":"kept"}
{"file":"review/IMG_1234.jpg","role":"review","status":"selected"}
{"file":"export/inat/IMG_1234_upload.jpg","role":"upload","status":"ready"}
{"file":"export/stock/IMG_1234_stock.jpg","role":"stock","status":"ready"}
{"file":"source/raw/IMG_0001.CR2","role":"original","status":"rejected"}
```

### `metadata/keywords.json`

Shared keyword/caption data that can be reused months later for stock submission.

```json
{
  "title": "Prairie Warbler in field grass",
  "caption": "Prairie Warbler (Setophaga discolor) foraging in grassland habitat.",
  "keywords": [
    "prairie warbler",
    "setophaga discolor",
    "bird",
    "warbler",
    "grassland",
    "wildlife"
  ],
  "source_taxon": "iNaturalist observation",
  "source_observation_id": 123456789
}
```

### `metadata/release.json`

Rights and usage notes for stock uploads.

```json
{
  "model_released": false,
  "property_released": false,
  "location_sensitive": true,
  "stock_status": "draft"
}
```

## Processing states

- `ingested`
- `selected`
- `rejected`
- `processed`
- `exported`
- `uploaded`
- `stock-ready`
- `stock-submitted`

## Retention policy

- Keep `source/raw/` forever unless you intentionally prune failed shots.
- Delete rejected raw files after culling so accidental frames, missed shots, and unusable duplicates do not clutter the archive.
- Keep `metadata/` long term because it preserves the link between iNat and stock workflows.
- Treat `review/` and `export/` as working files.
- Remove `export/inat/` once the upload is complete if you do not need local copies.
- Remove `export/stock/` after submission unless you want an archive of stock-ready renders.

## What to automate first

1. Ingest into `source/`.
2. Create the observation folder and `observation.json`.
3. Rename files by capture time if needed.
4. Generate a review JPEG.
5. Export an iNat-sized upload copy.
6. Reuse the same observation metadata to generate stock captions and keywords.

## What to leave manual at first

- Auto-tagging from pixels.
- Auto-cropping.
- Aggressive denoise on every frame.
