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
            IMG_1234.jpg
          stock\
            IMG_1234.jpg
        metadata\
          observation.json
          assets.jsonl
          IMG_1234.exif.json
          IMG_1234.xmp
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

## Additional metadata files

`observation.json`, `assets.jsonl`, `<stem>.exif.json`, and `<stem>.xmp` are
specified once, precisely, in **Minimal data schema** below — not repeated
here. This section covers the two metadata files that section doesn't:
`keywords.json` and `release.json`.

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
3. Extract EXIF into `metadata/<stem>.exif.json` and write matching XMP where useful.
4. Rename files by capture time if needed.
5. Generate a review JPEG.
6. Export an iNat-sized upload copy.
7. Reuse the same observation metadata to generate stock captions and keywords.

## First-pass script architecture

Keep it script-only and command-oriented. One thin CLI should orchestrate a few small library modules.

Note: the `scripts/` + `lib/` split below was the original planning sketch
and was never implemented. It is kept only for history; the actual layout is
a flat `src/bio_automation/` package:

```text
src/bio_automation/
  cli.py                  # single entrypoint / CLI dispatcher
  paths.py                # folder naming, stem rules, path helpers
  models.py               # Asset/Observation dataclasses for JSON
  io.py                   # read/write json, jsonl
  exif.py                 # exiftool wrapper; EXIF/XMP sidecar helpers
  inat.py                 # iNat read-only API client
  pipeline.py             # state transitions, retention rules, orchestration
  render.py               # review JPEGs and upload/stock exports (stub)
  stock.py                # captions/keywords from observation data (stub)
```

### Workflow

1. **ingest**
   - Copy camera files from card/import folder directly into `source/raw/`.
   - Create observation id + folder name.
   - Record every file in `assets.jsonl` with status `ingested`.

2. **cull**
   - Mark files `kept` or `rejected` in `assets.jsonl` (raws already live in `source/raw/`; culling never moves them).
   - Delete rejected raws in place, immediately after marking.
   - Review-JPEG generation is a separate step (see **render-review** in the minimal command set) and does not gate where a raw lives — it can run before, after, or independent of culling.

3. **metadata pull**
   - Fetch read-only iNaturalist observation data only.
   - Do not create/upload/sync observations to iNat.
   - Store the fetched observation payload locally by merging it into the
     observation record `ingest` already created: `metadata/observation.json`
     must already exist (a missing file is a user error — run `bio ingest`
     first), and the merge only overwrites the iNat-owned fields
     (`observation_id`, `source`, `taxon`, `place_guess`, `license`,
     `quality_grade`); `keywords_seed` is unioned rather than replaced.
     Everything else — notably `obsdate` and `locality` — is local-only and
     is never touched, since `obsdate` is already baked into the folder path.
     If the folder is already linked to a different `observation_id`, abort
     unless `--force` is passed, to avoid silently re-pointing a folder at a
     different iNat record.

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

`source` is a two-state lifecycle, not a free-form string: `bio ingest`
always writes `"local-ingest"`; `bio fetch-inat` overwrites it to
`"iNaturalist"` only after a successful merge. Right after ingest, before any
`fetch-inat` call:

```json
{
  "observation_id": null,
  "obsdate": "2026-03-01",
  "source": "local-ingest",
  "taxon": {},
  "place_guess": "F:\\Bio\\2026\\2026-03-01",
  "locality": null,
  "license": null,
  "quality_grade": null,
  "keywords_seed": []
}
```

After `bio fetch-inat <folder> --observation-id 123456789` merges in the
iNat-owned fields (see **Workflow** step 3):

```json
{
  "observation_id": 123456789,
  "obsdate": "2026-03-01",
  "source": "iNaturalist",
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

`obsdate` and `locality` are local-only and are never touched by
`fetch-inat`.

### `metadata/assets.jsonl`

One row per file and per processing state.

```json
{"file":"source/raw/IMG_1234.CR2","role":"original","status":"kept","selected":true}
{"file":"source/raw/IMG_0001.CR2","role":"original","status":"rejected","selected":false}
{"file":"review/IMG_1234.jpg","role":"review","status":"generated"}
{"file":"export/inat/IMG_1234.jpg","role":"upload","status":"ready"}
{"file":"export/stock/IMG_1234.jpg","role":"stock","status":"ready"}
```

### `metadata/<stem>.exif.json`

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

### `metadata/<stem>.xmp`

Sidecar mirror for apps that prefer XMP, one per raw.

## Minimal command set

```text
bio ingest <card-or-folder> --obsdate YYYY-MM-DD --slug "prairie-warbler"
bio cull <obs-folder>
bio fetch-inat <obs-folder> --observation-id 123456789 [--force]
bio write-sidecars <obs-folder>
bio render-review <obs-folder>
bio export-inat <obs-folder>
bio export-stock <obs-folder>
bio build-keywords <obs-folder>
```

`bio cull` already deletes rejected raws in place (see **Workflow** step 2);
there is no separate `prune` command.

`--force` on `fetch-inat` is only needed to relink a folder that is already
associated with a different `--observation-id`; see **Workflow** step 3.

## Minimal functions

- `make_observation_path(obsdate, slug) -> Path`
- `load_observation(folder) -> Observation` / `save_observation(folder, data: Observation)`
- `scan_assets(folder) -> list[Asset]`
- `save_assets(folder, assets: list[Asset])`
- `set_asset_status(folder, file_names, status, selected) -> list[Asset]`
- `delete_rejected_raws(folder)`
- `fetch_inat_observation(observation_id) -> dict` (returns the iNat-owned
  subset of fields — `observation_id`, `source`, `taxon`, `place_guess`,
  `license`, `quality_grade`, `keywords_seed` — reshaped into
  bio-automation's own field names, not iNat's raw API response)
- `merge_inat_fields(existing: Observation, fetched: dict) -> Observation`
  (overwrites only the iNat-owned fields, unions `keywords_seed`, leaves
  `obsdate`/`locality`/everything else untouched; pure, no I/O)
- `extract_exif(raw_path) -> dict`
- `write_sidecars(folder) -> list[Path]` (writes `metadata/<stem>.exif.json`
  + `metadata/<stem>.xmp` for every kept raw)
- `render_review_jpeg(raw_path, dest_path) -> Path` (stub pending an
  image-rendering dependency — see Implementation notes)
- `export_inat_jpeg(raw_path, dest_path) -> Path` (stub, same dependency)
- `export_stock_jpeg(raw_path, dest_path) -> Path` (stub, same dependency)
- `build_keywords(observation: Observation) -> {title, caption, keywords, source_taxon, source_observation_id}`
  (one record per observation folder, not per asset — `keywords.json` is a
  single shared file)

## Implementation notes

- No GUI dependency in v1; every action must be runnable from the CLI.
- Use read-only iNat metadata fetches only.
- Keep raw retention strict: only selected raws survive long term.
- Prefer plain JSON, JSONL, and XMP over any catalog database for v1.
- Make the pipeline idempotent so reruns do not duplicate exports or metadata.
- `observation.json`/`assets.jsonl` are the `Observation`/`Asset` dataclasses in
  `models.py`, not ad hoc dicts: writers call `dataclasses.asdict()` before
  handing data to `io.write_json`/`write_jsonl`, and readers reconstruct with
  `Observation(**data)` / `Asset(**row)`. This is v1-only ceremony we accept:
  the pipeline is the sole reader and writer of its own files, so unknown or
  missing keys raise `TypeError` immediately rather than silently drifting.
  If a future step (e.g. real `fetch-inat`) needs to tolerate extra fields,
  extend the dataclasses or filter keys explicitly at that boundary instead of
  reverting to dicts.
- `Observation.source` is a two-state lifecycle, not a free-form string:
  `"local-ingest"` (set by `bio ingest`) and `"iNaturalist"` (set by
  `bio fetch-inat` on a successful merge). The `"inat-readonly"` value that
  used to appear here described the API *access policy* (read-only), not
  per-record data provenance, and has been retired.
- `bio write-sidecars`/`render-review`/`export-inat`/`export-stock` and their
  underlying `render.py`/`exif.py` functions are implemented (2026-10-19).
  review/inat/stock JPEGs are sourced from the raw's embedded preview
  (`exiftool -b -PreviewImage`, falling back to `-JpgFromRaw`), resized and
  re-encoded with Pillow (now a real `pyproject.toml` dependency), not a full
  raw development. `export-stock` is therefore capped by whatever
  resolution/color the camera embedded in-frame -- a raw-decode library
  (e.g. `rawpy`/LibRaw, already used elsewhere in this codebase by
  `preclean.py`) would be needed to go beyond that, if ever needed for stock
  buyers. All four operate only on raws at `status == "kept"` (not merely
  ingested) -- a raw that cull never explicitly kept is skipped, not treated
  as implicitly kept.

## What to leave manual at first

- Auto-tagging from pixels.
- Auto-cropping.
- Aggressive denoise on every frame.
