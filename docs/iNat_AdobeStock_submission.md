## iNaturalist & Adobe Stock: Submission Schemas & Metadata Reference

**RESEARCH mode – grounded in authoritative documentation**

### iNaturalist Observation JSON Schema (POST /observations)

**Core Required Fields:**

| Field | Type | Description | Source |
|-------|------|-------------|--------|
| `species_guess` | string | Common/scientific name ("Monarch butterfly") | https://inaturalist.org/pages/api+reference |
| `taxon_id` | integer | iNaturalist taxon ID (enables faster identification) | https://api.inaturalist.org/v2/docs/ |
| `observed_on` / `observed_on_string` | ISO 8601 date | Date/time observation was made | https://pyinaturalist.readthedocs.io/en/stable/modules/pyinaturalist.v2.observations.html |
| `latitude` | float | WGS84 latitude [-90, 90] | https://inaturalist.org/pages/api+reference |
| `longitude` | float | WGS84 longitude [-180, 180] | https://inaturalist.org/pages/api+reference |
| `location_is_exact` | boolean | Whether coordinates are exact or approximate | https://pyinaturalist.readthedocs.io/en/stable/modules/pyinaturalist.v2.observations.html |
| `license` | string | CC0, CC-BY, CC-BY-NC, CC-BY-ND, CC-BY-SA, CC-BY-NC-ND, CC-BY-NC-SA, or null | https://pyinaturalist.readthedocs.io/en/stable/modules/pyinaturalist.v2.observations.html |

**Geo-Privacy Fields:**

| Field | Values | Description |
|-------|--------|-------------|
| `geoprivacy` | open, obscured, obscured_private, private | Location privacy level. Auto-set for endangered species. |
| `positional_accuracy` | integer (meters) | Accuracy estimate (7798 for ~5 miles; 44 for high precision) |

**Verifiability & Quality (Post-Submission):**

| Field | Values | Description |
|-------|--------|-------------|
| `quality_grade` | casual, needs_id, research | Research Grade = community consensus on species-level ID + photo + date + coordinates + wild/free-living organism |
| `captive_cultivated` | boolean | Mark as captive/cultivated (prevents Research Grade unless released) |

---

### Adobe Stock Asset: IPTC/XMP Metadata Schema

**Mandatory Adobe Stock Fields:**

| Field | Type | Limit | Description |
|-------|------|-------|-------------|
| **Title** | string | 200 chars (ideal: 70) | Concise, searchable. Becomes URL. No brand/product/people names. |
| **Keywords** | array | 5–50 (ideal: 15–35) | Ordered by relevance. First 10 weighted heaviest for search. Singular nouns, factual adjectives. No brands, no people names. |
| **Description** | string | N/A | Free-text narrative. Optional but recommended. |

**IPTC Core 1.5 (XMP Namespace: `Iptc4xmpCore`):**

| Field | XMP Property | Type | Use |
|-------|--------------|------|-----|
| Title | `iptc4xmpCore:Title` | Text (lang-alt) | Headline asset title |
| Description | `iptc4xmpCore:Description` | Text (lang-alt) | Narrative description |
| Keywords | `iptc4xmpCore:Keywords` | Text array | Ordered keyword list |
| Creator | `iptc4xmpCore:Creator` | Text array | Photographer/artist name |
| Copyright Notice | `iptc4xmpCore:CopyrightNotice` | Text (lang-alt) | © notice |
| Usage Terms | `iptc4xmpCore:UsageTerms` | Text (lang-alt) | License terms (e.g., "CC-BY-NC") |
| Credit Line | `iptc4xmpCore:CreditLine` | Text | Attribution requirement |

**IPTC Extension 1.8 (XMP Namespace: `Iptc4xmpExt`):**

| Field | XMP Property | Nature Photography Use |
|-------|--------------|------------------------|
| Location Created | `Iptc4xmpExt:LocationCreated` | Geographic metadata: Country, State, City, GPS (WGS84). Critical for stock discovery. |
| Location Created: Country | `Iptc4xmpExt:LocationCreated[1]/Iptc4xmpExt:CountryName` | "United States", "Canada" |
| Location Created: State/Province | `Iptc4xmpExt:LocationCreated[1]/Iptc4xmpExt:ProvinceState` | "Colorado", "Ontario" |
| Location Created: City | `Iptc4xmpExt:LocationCreated[1]/Iptc4xmpExt:City` | "Denver", "Rocky Mountain National Park" |
| Location Created: GPS | `Iptc4xmpExt:LocationCreated[1]/exif:GPSLatitude`, `exif:GPSLongitude` | WGS84 coordinates. Altitude as `altitude/1` (fraction). |
| Artwork/Species | `Iptc4xmpExt:ArtworkOrObject` | Structured: Scientific name, Common name, Copyright, Source. **Essential for species tagging.** |
| Species Scientific Name | `Iptc4xmpExt:ArtworkOrObject[1]/Iptc4xmpExt:Title` | Formal taxon: "Danaus plexippus" |
| Event | `Iptc4xmpExt:Event` | "fall migration", "mating display", "foraging" |
| Date Created | `Iptc4xmpExt:DateCreated` | ISO 8601 date of photograph ("2023-08-15") |

**PLUS Metadata (Rights, Namespace: `plus`):**

| Field | XMP Property | Description |
|-------|--------------|-------------|
| Copyright Owner | `plus:CopyrightOwner` | Creator/rights holder name |
| Credit Line Required | `plus:CreditLineRequired` | Boolean: attribution mandatory |
| Web Statement of Rights | `plus:WebStatementOfRights` | URI to full license terms (https://creativecommons.org/licenses/by-nc/4.0/) |
| Copyright Status | `plus:CopyrightStatus` | copyrighted, public domain, unknown |

**XMP Rights (Namespace: `xmpRights`):**

| Field | XMP Property | Description |
|-------|--------------|-------------|
| Copyright | `xmpRights:Copyright` | "© 2023 Jane Photographer" |
| Marked | `xmpRights:Marked` | Boolean: copyright-marked |
| Usage Terms | `xmpRights:UsageTerms` | "CC-BY-NC 4.0" or license summary |
| Web Statement | `xmpRights:WebStatement` | URL to full rights document |

---

See notes section in ingestion plan for privacy, license, and submission best practices.
