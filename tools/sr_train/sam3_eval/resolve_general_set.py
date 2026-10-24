"""Resolve a cross-taxon random sample (NOT limited to Arachnida) to test whether the
SAM3-vs-Topaz masking result generalizes across the pipeline's real subject distribution
(Plantae, Fungi, Insecta, Mammalia, Aves, Mollusca, Reptilia, Amphibia, ...), using the
same authoritative manifest.csv/manifest_nas.csv (obs_id -> output_jpeg) +
inat_observations.json (obs_id -> taxon.iconic, taxon.common) approach as
resolve_spider_set.py. Stratified: up to PER_GROUP photos per iconic group, randomly
chosen within each group (seeded), so small groups aren't drowned out by Plantae/Fungi's
much larger counts.
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
OUT_LIST = Path(r"D:\dev\bio-sr-train\tools\sr_train\sam3_eval\general_set.json")
PER_GROUP = 8
MIN_GROUP_SIZE = 3
SKIP_GROUPS = {"Chromista"}  # too few observations to be meaningful (n=4 in the library)


def main():
    obs_data = json.loads(INAT_OBS.read_text(encoding="utf-8"))["observations"]
    obs_by_id = {o["id"]: o for o in obs_data}

    resolved_by_group = {}
    for manifest_path in MANIFESTS:
        with open(manifest_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("status") != "ok":
                    continue
                raw_ids = (row.get("obs_ids") or "").replace(";", ",")
                ids = [s.strip() for s in raw_ids.split(",") if s.strip()]
                hit = None
                for id_str in ids:
                    try:
                        obs_id = int(id_str)
                    except ValueError:
                        continue
                    obs = obs_by_id.get(obs_id)
                    if obs is None:
                        continue
                    iconic = (obs.get("taxon") or {}).get("iconic")
                    if iconic and iconic not in SKIP_GROUPS:
                        hit = (obs_id, obs, iconic)
                        break
                if hit is None:
                    continue
                obs_id, obs, iconic = hit
                out_jpeg = Path(row["output_jpeg"])
                if not out_jpeg.exists():
                    continue
                entry = {
                    "obs_id": obs_id,
                    "iconic": iconic,
                    "taxon": (obs.get("taxon") or {}).get("name"),
                    "common": (obs.get("taxon") or {}).get("common"),
                    "observed_on": obs.get("observed_on"),
                    "path": str(out_jpeg),
                    "title": row.get("title"),
                }
                resolved_by_group.setdefault(iconic, {})[obs_id] = entry  # dedup by obs_id

    rng = random.Random(20261023)
    final = []
    group_counts = {}
    for group, by_obs in resolved_by_group.items():
        items = list(by_obs.values())
        if len(items) < MIN_GROUP_SIZE:
            continue
        rng.shuffle(items)
        picked = items[:PER_GROUP]
        final.extend(picked)
        group_counts[group] = (len(picked), len(items))

    OUT_LIST.write_text(json.dumps(final, indent=2))
    print(f"wrote {len(final)} photos across {len(group_counts)} iconic groups to {OUT_LIST}")
    for g, (picked, total) in sorted(group_counts.items(), key=lambda kv: -kv[1][1]):
        print(f"  {g:16s} picked={picked:3d} available={total}")


if __name__ == "__main__":
    main()
