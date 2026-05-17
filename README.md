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
