# EXIF -> Observation JSON / XMP Field Mapping

**Bio-automation pipeline (loci_ingest) reference**

---

## Pipeline stages

1. **ingest**: Copy raw files, create `observation.json` with source = local-ingest
2. **fetch-inat**: Merge iNat observation fields; overwrites source and some keys
3. **write-sidecars**: Extract EXIF, normalize to `metadata/<STEM>.exif.json`, render `metadata/<STEM>.xmp`
4. **build-keywords**: Compute stock keywords from observation record
5. **export**: Render JPEGs (iNat-sized, stock-sized)

---

## 1. Camera EXIF → Normalized Sidecar JSON

**Source:** exiftool -j on raw file (CR2/NEF/RAF/etc)
**Output:** `metadata/<STEM>.exif.json` (one per raw)

| EXIF Tag | JSON Field | Type | Format | Immutable | Notes |
|---------|------------|------|--------|------------|-------|
| DateTimeOriginal | datetime_original | string | ISO8601 (YYYY-MM-DDTHH:MM:SS) | Yes | Capture timestamp |
| Make | camera_make | string | Free text | Yes | Manufacturer |
| Model | camera_model | string | Free text | Yes | Camera body |
| LensModel | lens_model | string \| null | Free text | Yes | Lens name (null if unknown) |
| GPSLatitude | gps.lat | float | Decimal degrees [-90, 90] | Yes | WGS84 lat; null if absent |
| GPSLongitude | gps.lon | float | Decimal degrees [-180, 180] | Yes | WGS84 lon; null if absent |
| GPSAltitude | gps.alt | float \| null | Meters above sea level | Yes | Altitude; null if absent |
| (source path) | file | string | Relative path within repo | Yes | e.g., "source/raw/IMG_1234.CR2" |

### Normalization rules

- Timestamps: format as ISO8601 (use T separator; Z for UTC if applicable)
- GPS: convert to decimal degrees; missing → `null` (never 0 or empty string)
- All other fields required; missing keys → `null`
- Do not fill defaults; keep null to indicate missing data

### Example exif.json

```json
{
  "file": "source/raw/IMG_1234.CR2",
  "datetime_original": "2026-03-01T14:32:45",
  "camera_make": "Canon",
  "camera_model": "Canon EOS R6",
  "lens_model": "RF 100-500mm F4.5-7.1 L IS USM",
  "gps": {
    "lat": 41.1234,
    "lon": -72.5678,
    "alt": 145.2
  }
}
```

---

## 2. Observation Lifecycle: Local Ingest → iNat Merge

**File:** `metadata/observation.json`
**Dataclass:** `models.Observation`

### Phase 1: Local Ingest

Set immediately by `ingest`; only mutable by `fetch-inat`:

| Field | Source | Type | Default | Behavior | Overwritable |
|-------|--------|------|---------|----------|--------------|
| observation_id | `--observation-id` (fetch-inat only) | int \| null | null | Always null until iNat link | By fetch-inat |
| obsdate | `--obsdate YYYY-MM-DD` (ingest) | string | required | Baked into folder path; immutable | Never |
| source | System | enum | local-ingest | Changes to iNaturalist after fetch-inat | By fetch-inat only |
| taxon | iNat API (Phase 2) | dict | {} | Empty until iNat fetch | By fetch-inat |
| place_guess | Inferred from folder parent / user | string | folder path | Geographic reference; iNat can override | By fetch-inat |
| locality | User notes (local-only) | string \| null | null | Local-only; iNat does not overwrite | Never |
| license | iNat API (Phase 2) | string \| null | null | License code from iNat | By fetch-inat |
| quality_grade | iNat API (Phase 2) | string \| null | null | research / needs_id / casual | By fetch-inat |
| keywords_seed | iNat keywords (unioned) | list[str] | [] | Unioned with iNat keywords | By fetch-inat |

### Phase 2: iNaturalist Fetch

Merge behavior (`pipeline.merge_inat_fields`):

| Field | iNat API -> Output | Merge Rule | Result After Merge |
|-------|-------------------|-----------|-------------------|
| observation_id | response.id | Overwrite | 123456789 (example) |
| source | N/A | Force-set | iNaturalist |
| taxon | response.taxon | Overwrite | {id, name, rank} |
| place_guess | response.place_guess | Overwrite | iNat's location string |
| license | response.license | Overwrite | CC BY-NC (example) |
| quality_grade | response.quality_grade | Overwrite | research/needs_id/casual |
| keywords_seed | (derived) | Union | Append iNat keywords (deduped) |
| obsdate | N/A | Never touched | Still from ingest |
| locality | N/A | Never touched | Stays null or user-set value |

### Observers

- `obsdate` is baked into folder path; never rewritten from iNat's observed_on
- `locality` is local-only; iNat's place_guess does not overwrite
- `keywords_seed` is unioned (appended) not replaced
- `source` tracks provenance (local-ingest vs iNaturalist)

---

## 3. EXIF Sidecar → XMP Output Binding

| exif.json Field | XMP Namespace/Element | Type | Omitted if Null |
|-----------------|-----------------------|------|-----------------|
| datetime_original | exif:DateTimeOriginal | string | Yes |
| camera_make | exif:Make | string | Yes |
| camera_model | exif:Model | string | Yes |
| lens_model | exif:LensModel | string | Yes |
| gps.lat | exif:GPSLatitude | float | Yes |
| gps.lon | exif:GPSLongitude | float | Yes |
| gps.alt | exif:GPSAltitude | float | Yes |

**XMP Template (minimal v1)**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:exif="http://ns.adobe.com/exif/1.0/">
  <rdf:Description rdf:about=""
                   exif:DateTimeOriginal="2026-03-01T14:32:45"
                   exif:Make="Canon"
                   exif:Model="Canon EOS R6"
                   exif:LensModel="RF 100-500mm F4.5-7.1 L IS USM">
    <exif:GPSLatitude>41.1234</exif:GPSLatitude>
    <exif:GPSLongitude>-72.5678</exif:GPSLongitude>
    <exif:GPSAltitude>145.2</exif:GPSAltitude>
  </rdf:Description>
</rdf:RDF>
```

**Binding rules**

- All non-null EXIF fields → matching XMP attribute/element
- GPS fields split: `gps.{lat, lon, alt}` → separate `exif:GPS{Latitude, Longitude, Altitude}` elements
- Null values omitted from XMP
- Namespace: http://ns.adobe.com/exif/1.0/
- UTF-8 encoding declared

---

## 4. Observation JSON → iNat Submit Payload

From subagent-1 we have the authoritative spec, summarized:

- Required fields: `species_guess` OR `taxon_id`, `observed_on`/`observed_on_string`, `latitude`, `longitude`, `license` (optional), `local_photos` at least 1 (for Research Grade)
- Geo-privacy: `geoprivacy` (open/obscured/private), `positional_accuracy` (meters)
- Quality-grade (post-submission): `quality_grade` (research/needs_id/casual), `captive_cultivated` (boolean)

The pipeline will materialize this from `metadata/observation.json`.

---

## 5. Observation JSON → Adobe Stock Asset

- Title: `taxon.name + place_guess` (max 200 chars, ideal 70)
- Keywords: from `observation.keywords_seed` ordered by relevance; 5–50 (ideal 15–35)
- Description: optional; links taxon, location, date
- IPTC/XMP: use `metadata/<STEM>.xmp` as source; add IPTC Core/CommonPlaces if needed

---

## 6. Validation Rules

- **EXIF → JSON Normalization**
  - All exiftool fields must appear in normalized output
  - Timestamps ISO8601 (YYYY-MM-DDTHH:MM:SS)
  - GPS floats, decimal degrees; missing → null
  - No defaults; preserve nulls

- **XMP binding**
  - 1:1 mapping for each field; GPS split into three elements
  - Null values omitted from XMP
  - Valid XML, proper namespaces, UTF-8

- **Observation immutability**
  - `obsdate` never changes after ingest
  - `locality` never overwritten by iNat
  - `source` correctly tracks provenance

- **Keyword union**
  - `keywords_seed` superset of input (or equal if no iNat keywords added)
  - Deduped; correctly ordered for stock

- **Stock keyword hygiene**
  - Singular nouns; factual adjectives only
  - No brands, people names, model specs
  - No keyword spammings

---

## 7. Generating Schema Files

- `bio_pipeline_schema.json` – complete JSON schema for validation
  - observation.json constraints (type, enum, nullability)
  - exif.json constraints
  - assets.jsonl and keywords.json as needed
- `EXIF_to_Observation_Mapping.md` – this document (the full mapping)

All schema files should be checked into `docs/` for reproducible ingestion/validation pipelines.

---

## 8. Link to Reference Schemas

- iNaturalist Observation JSON: docs/iNat_AdobeStock_submission.md (field-level spec)
- Adobe Stock IPTC/XMP: docs/iNat_AdobeStock_submission.md (section Adobe Stock Asset: IPTC/XMP)
- Local sidecar and meta`bio_pipeline_schema.json`, `EXIF_to_Observation_Mapping.md`