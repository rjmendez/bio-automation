# Folder and metadata schema

Goal: keep the library filesystem-first, iNaturalist-friendly, and easy to open source later.

## Principles

- Originals stay immutable.
- Every derived file should be reproducible from the source plus metadata.
- Folder names should be human-readable first, machine-readable second.
- Metadata should live in plain text sidecars or a small local index, not a proprietary catalog.

## Recommended layout

```text
F:\Bio\
  2026\
    2026-03-01\
      obs-20260301-001-prairie-warbler\
        source\
          IMG_1234.CR2
          IMG_1235.CR2
        work\
          IMG_1234.dng
        review\
          IMG_1234.jpg
        export\
          inat\
            IMG_1234_upload.jpg
        metadata\
          observation.json
          assets.jsonl
```

## Folder meaning

| Folder | Purpose |
|---|---|
| `source/` | Camera originals and untouched imports |
| `work/` | Temporary or intermediate processing outputs |
| `review/` | Human-review copies for culling or comparison |
| `export/inat/` | Upload-ready images for iNaturalist |
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
```

## Processing states

- `ingested`
- `selected`
- `processed`
- `exported`
- `uploaded`

## What to automate first

1. Ingest into `source/`.
2. Create the observation folder and `observation.json`.
3. Rename files by capture time if needed.
4. Generate a review JPEG.
5. Export an iNat-sized upload copy.

## What to leave manual at first

- Auto-tagging from pixels.
- Auto-cropping.
- Aggressive denoise on every frame.
