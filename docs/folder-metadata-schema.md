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
- Script-only first: no GUI dependency is required for ingest, tagging, or export.

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
| `source/raw/` | Selected raw files retained long term |
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

### `metadata/image.exif.json`

Per-file EXIF extracted from the raw or export copy.

```json
{
  "file": "source/raw/DSC_0008.CR2",
  "datetime_original": "2025-10-09 15:42:31",
  "camera_make": "Nikon",
  "camera_model": "Z 6",
  "gps_latitude": 41.1234,
  "gps_longitude": -72.5678,
  "gps_altitude": 12.3
}
```

### `metadata/image.xmp`

Sidecar used by tooling that prefers XMP. Keep this in sync with EXIF where possible.

### `metadata/assets.jsonl`

One row per file for file-level processing state.

```json
{"file":"source/raw/IMG_1234.CR2","role":"original","status":"kept"}
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
- Store EXIF in both the file and a JSON sidecar when practical so the archive survives tooling changes.
- Treat `review/` and `export/` as working files.
- Remove `export/inat/` once the upload is complete if you do not need local copies.
- Remove `export/stock/` after submission unless you want an archive of stock-ready renders.

## What to automate first

1. Ingest into `source/`.
2. Create the observation folder and `observation.json`.
3. Extract EXIF into `metadata/image.exif.json` and write matching XMP where useful.
4. Rename files by capture time if needed.
5. Generate a review JPEG.
6. Export an iNat-sized upload copy.
7. Reuse the same observation metadata to generate stock captions and keywords.

## First-pass script architecture

Keep it script-only and command-oriented. One thin CLI should orchestrate a few small library modules.

```text
scripts/
  bio.py                  # single entrypoint / CLI dispatcher
  ingest.py               # copy card -> raw inbox
  cull.py                 # mark keep/reject, delete rejected raws
  metadata.py             # iNat fetch, EXIF/XMP write, local sidecars
  render.py               # review JPEGs and upload/stock exports
  stock.py                # captions/keywords from observation data

lib/
  paths.py                # folder naming, stem rules, path helpers
  models.py               # dataclasses / typed dicts for JSON
  io.py                   # read/write json, jsonl, xmp, sidecars
  exif.py                 # exiftool wrapper or equivalent
  inat.py                 # iNat read-only API client
  pipeline.py             # state transitions and retention rules
```

### Workflow

1. **ingest**
   - Copy camera files from card/import folder into an observation work folder.
   - Create observation id + folder name.
   - Record every file in `assets.jsonl`.

2. **cull**
   - Generate review JPEGs.
   - Mark files `kept` or `rejected`.
   - Delete rejected raws immediately after culling.
   - Move only kept raws into long-term `source/raw/`.

3. **metadata pull**
   - Fetch read-only iNaturalist observation data only.
   - Do not create/upload/sync observations to iNat.
   - Store the fetched observation payload locally.

4. **sidecar write**
   - Write EXIF JSON sidecars from the raw file.
   - Write XMP sidecars when the target app can consume them.
   - Keep EXIF/XMP values aligned where possible.

5. **export**
   - Build iNat-ready JPEGs from selected raws.
   - Build stock-ready derivatives and generate captions/keywords from the same observation record.

6. **retain**
   - Keep selected raws and metadata long term.
   - Treat review files and exports as disposable unless explicitly archived.

## Minimal data schema

Use one folder-level observation record plus one file-level asset log.

### `metadata/observation.json`

```json
{
  "observation_id": 123456789,
  "obsdate": "2026-03-01",
  "source": "inat-readonly",
  "taxon": {
    "id": 12345,
    "name": "Prairie Warbler",
    "rank": "species"
  },
  "place_guess": "F:\\Bio\\2026\\2026-03-01",
  "locality": "nearfield notes",
  "license": "CC BY-NC",
  "quality_grade": "research",
  "keywords_seed": [
    "prairie warbler",
    "setophaga discolor",
    "grassland",
    "wildlife"
  ]
}
```

### `metadata/assets.jsonl`

One row per file and per processing state.

```json
{"file":"source/raw/IMG_1234.CR2","role":"original","status":"kept","selected":true}
{"file":"source/raw/IMG_0001.CR2","role":"original","status":"rejected","selected":false}
{"file":"review/IMG_1234.jpg","role":"review","status":"generated"}
{"file":"export/inat/IMG_1234.jpg","role":"upload","status":"ready"}
{"file":"export/stock/IMG_1234.jpg","role":"stock","status":"ready"}
```

### `metadata/image.exif.json`

```json
{
  "file": "source/raw/IMG_1234.CR2",
  "datetime_original": "2025-10-09T15:42:31",
  "camera_make": "Nikon",
  "camera_model": "Z 6",
  "lens_model": "NIKKOR Z 180-600mm",
  "gps": {
    "lat": 41.1234,
    "lon": -72.5678,
    "alt": 12.3
  }
}
```

### `metadata/image.xmp`

Sidecar mirror for apps that prefer XMP.

## Minimal command set

```text
bio ingest <card-or-folder> --obsdate YYYY-MM-DD --slug "prairie-warbler"
bio cull <obs-folder>
bio fetch-inat <obs-folder> --observation-id 123456789
bio write-sidecars <obs-folder>
bio render-review <obs-folder>
bio export-inat <obs-folder>
bio export-stock <obs-folder>
bio build-keywords <obs-folder>
bio prune <obs-folder> --rejected-only
```

## Minimal functions

- `make_observation_path(obsdate, slug) -> Path`
- `load_observation(folder) / save_observation(folder, data)`
- `scan_assets(folder) -> list[Asset]`
- `mark_asset(file, status, role)`
- `delete_rejected_raws(folder)`
- `fetch_inat_observation(observation_id) -> dict`
- `extract_exif(raw_path) -> dict`
- `write_exif_json(asset_path, data)`
- `write_xmp_sidecar(asset_path, data)`
- `render_review_jpeg(raw_path) -> Path`
- `export_inat_jpeg(raw_path) -> Path`
- `export_stock_jpeg(raw_path) -> Path`
- `build_stock_metadata(observation, asset) -> {title, caption, keywords}`

## Implementation notes

- No GUI dependency in v1; every action must be runnable from the CLI.
- Use read-only iNat metadata fetches only.
- Keep raw retention strict: only selected raws survive long term.
- Prefer plain JSON, JSONL, and XMP over any catalog database for v1.
- Make the pipeline idempotent so reruns do not duplicate exports or metadata.

## What to leave manual at first

- Auto-tagging from pixels.
- Auto-cropping.
- Aggressive denoise on every frame.
