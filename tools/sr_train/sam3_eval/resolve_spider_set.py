"""Resolve a varied set of real, on-disk spider (Arachnida) photos for the SAM3-vs-Topaz
mask comparison. Uses the pipeline's own authoritative manifests (manifest.csv,
manifest_nas.csv -- obs_id -> output_jpeg, already-rendered and disk-verified by the
pipeline itself) cross-referenced against inat_observations.json's taxon.iconic ==
"Arachnida", instead of reconstructing paths from filenames -- the DSC_#### frame number
collides across shoot days (the exact trap that produced a wrong test image earlier this
session), and these manifests already carry the disambiguated, correct path per obs_id.
"""
import csv
import json
import random
from pathlib import Path

INAT_OBS = Path(r"F:\bio-automation-data\inat_observations.json")
MANIFESTS = [
    Path(r"F:\Bio-export\adobe-stock\manifest.csv"),
    Path(r"F:\Bio-export\adobe-stock\manifest_nas.csv"),
]
OUT_LIST = Path(r"D:\dev\bio-sr-train\tools\sr_train\sam3_eval\spider_set.json")


def main(target_n=50):
    obs_data = json.loads(INAT_OBS.read_text(encoding="utf-8"))["observations"]
    obs_by_id = {o["id"]: o for o in obs_data}

    resolved = []
    unresolved = []

    for manifest_path in MANIFESTS:
        with open(manifest_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("status") != "ok":
                    continue
                raw_ids = (row.get("obs_ids") or "").replace(";", ",")
                ids = [s.strip() for s in raw_ids.split(",") if s.strip()]
                if not ids:
                    continue
                # One row can cover several obs_ids (burst frames); any Arachnida id counts.
                taxa_hit = None
                for id_str in ids:
                    try:
                        obs_id = int(id_str)
                    except ValueError:
                        continue
                    obs = obs_by_id.get(obs_id)
                    if obs and (obs.get("taxon") or {}).get("iconic") == "Arachnida":
                        taxa_hit = (obs_id, obs["taxon"])
                        break
                if taxa_hit is None:
                    continue

                obs_id, taxon = taxa_hit
                out_jpeg = Path(row["output_jpeg"])
                if out_jpeg.exists():
                    resolved.append({
                        "obs_id": obs_id,
                        "taxon": taxon.get("name"),
                        "common": taxon.get("common"),
                        "observed_on": obs_by_id[obs_id].get("observed_on"),
                        "path": str(out_jpeg),
                        "title": row.get("title"),
                        "manifest": manifest_path.name,
                    })
                else:
                    unresolved.append({"obs_id": obs_id, "reason": "manifest path missing on disk",
                                        "tried": str(out_jpeg)})

    # Dedup by obs_id (a burst can produce several manifest rows for the same observation).
    by_obs = {}
    for r in resolved:
        by_obs.setdefault(r["obs_id"], r)
    resolved = list(by_obs.values())

    # True random sample across ALL distinct taxa (previous version alphabetically
    # truncated after sorting by taxon name, which silently biased the set toward
    # genus names starting A-F -- caught before reporting any conclusion from it).
    by_taxon = {}
    for r in resolved:
        by_taxon.setdefault(r["taxon"], []).append(r)
    rng = random.Random(20261023)
    taxa_order = list(by_taxon.keys())
    rng.shuffle(taxa_order)
    final = [by_taxon[t][0] for t in taxa_order[:target_n]]

    OUT_LIST.write_text(json.dumps(final, indent=2))
    print(f"resolved={len(resolved)} unresolved={len(unresolved)} distinct_taxa={len(by_taxon)} "
          f"-> wrote {len(final)} to {OUT_LIST}")
    print("distinct taxa in final set:", sorted(set(r['taxon'] for r in final)))


if __name__ == "__main__":
    main()
