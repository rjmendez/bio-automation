# bio-automation

Tools and workflows for managing a personal bio/photo library, including iNaturalist-oriented processing.

## Usage

```powershell
bio ingest F:\Bio\2025\2025-10-09\inat-LR --obsdate 2025-10-09 --slug prairie-warbler
bio reject F:\Bio\2025\2025-10-09\obs-20251009-prairie-warbler DSC_0001.jpg DSC_0002.jpg
bio cull F:\Bio\2025\2025-10-09\obs-20251009-prairie-warbler
```

`ingest` copies source files into `source/raw/` and seeds metadata.
`reject` marks files for deletion.
`cull` deletes rejected raw files.

## Design

- [Folder and metadata schema](docs/folder-metadata-schema.md)

## Disk guard

`bio_automation.disk_guard` pauses a worker (instead of failing mid-write) while C: or the data drive is under its floor, writes `disk_pause` / `disk_resume` / `disk_timeout` events (the social audit log, or `settings.DISK_GUARD_LOG` for the pipeline) and resumes by itself at floor + margin. Thresholds: `BIO_DISK_GUARD_SYSTEM_MIN_GB` (15), `BIO_DISK_GUARD_DATA_MIN_GB` (20), `BIO_DISK_GUARD_POLL_S`, `BIO_DISK_GUARD_RESUME_MARGIN_GB`, `BIO_DISK_GUARD_MAX_WAIT_S` (0 = wait as long as it takes); social-prep and the live post run have their own (`BIO_SOCIAL_PREP_*`, `BIO_SOCIAL_POST_*`). Scripts outside this package (the ops repo's `nas_pipeline.py`, `pipeline_upkeep`) call `bio disk-guard --wait --who <name> && <step>` or `disk_guard.guard_for_write(who)` before each write step. Temp files go to `BIO_SCRATCH_DIR` (default `D:\bio-cache\tmp`), never `%TEMP%`.
