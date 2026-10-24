"""Resolve a training pool for the 3 weak categories found this session (Fungi, Plantae,
Protozoa), EXCLUDING the 94-photo general_set.json eval sample so that set stays a clean
held-out comparison for before/after fine-tune measurement.
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
EVAL_SET = Path(r"D:\dev\bio-sr-train\tools\sr_train\sam3_eval\general_set.json")
OUT = Path(r"D:\dev\bio-sr-train\tools\sr_train\sam3_eval\weak_train_pool.json")
GROUPS = {"Fungi", "Plantae", "Protozoa"}
# Protozoa's real ceiling in this library is 123 available (measured); Fungi/Plantae have
# thousands. Scaled up from the 50/group smoke-test pool per the owner's go-ahead to use a
# real GPU's headroom -- Protozoa capped near its actual supply, not artificially matched
# to Fungi/Plantae's much larger counts.
PER_GROUP = {"Fungi": 5000, "Plantae": 5000, "Protozoa": 120}  # 5000 = take everything
# available for Fungi/Plantae (actual supply: 2295/2449, well under this cap); Protozoa
# capped at its real ceiling (123 total available, measured).


def main():
    obs_data = json.loads(INAT_OBS.read_text(encoding="utf-8"))["observations"]
    obs_by_id = {o["id"]: o for o in obs_data}

    eval_obs_ids = {r["obs_id"] for r in json.loads(EVAL_SET.read_text(encoding="utf-8"))}

    by_group = {}
    for manifest_path in MANIFESTS:
        with open(manifest_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("status") != "ok":
                    continue
                raw_ids = (row.get("obs_ids") or "").replace(";", ",")
                ids = [s.strip() for s in raw_ids.split(",") if s.strip()]
                if any(s.strip().isdigit() and int(s.strip()) in eval_obs_ids for s in ids):
                    continue  # skip rows overlapping the held-out eval set entirely
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
                    if iconic in GROUPS:
                        hit = (obs_id, obs, iconic)
                        break
                if hit is None:
                    continue
                obs_id, obs, iconic = hit
                out_jpeg = Path(row["output_jpeg"])
                if not out_jpeg.exists():
                    continue
                entry = {
                    "obs_id": obs_id, "iconic": iconic,
                    "taxon": (obs.get("taxon") or {}).get("name"),
                    "common": (obs.get("taxon") or {}).get("common"),
                    "path": str(out_jpeg),
                }
                by_group.setdefault(iconic, {})[obs_id] = entry

    rng = random.Random(20261023)
    final = []
    for g in GROUPS:
        items = list(by_group.get(g, {}).values())
        rng.shuffle(items)
        picked = items[:PER_GROUP[g]]
        final.extend(picked)
        print(f"{g}: picked {len(picked)} of {len(items)} available (eval-set excluded)")

    OUT.write_text(json.dumps(final, indent=2))
    print(f"\nwrote {len(final)} training-pool photos to {OUT}")


if __name__ == "__main__":
    main()
