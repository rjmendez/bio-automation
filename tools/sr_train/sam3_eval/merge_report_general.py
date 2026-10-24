"""Merge the Topaz and SAM3 cross-taxon general-set results into a comparison table,
broken down by iconic group."""
import json
from pathlib import Path
from collections import defaultdict

BASE = Path(r"D:\dev\bio-sr-train\tools\sr_train\sam3_eval")
topaz = json.loads((BASE / "results_general/topaz/_summary.json").read_text(encoding="utf-8"))
sam3 = json.loads((BASE / "results_general/sam3/_summary.json").read_text(encoding="utf-8"))

topaz_by_obs = {r["obs_id"]: r for r in topaz}
sam3_by_obs = {r["obs_id"]: r for r in sam3}

rows = []
for obs_id, t in topaz_by_obs.items():
    s = sam3_by_obs.get(obs_id)
    if s is None:
        continue
    topaz_frac = t.get("mask_area_frac")
    topaz_degenerate = topaz_frac is not None and topaz_frac > 0.85
    inf = s["prompts"].get("informed", {})
    gen = s["prompts"].get("generic", {})
    sam3_hit = (inf.get("n_instances", 0) > 0) or (gen.get("n_instances", 0) > 0)
    rows.append({
        "obs_id": obs_id, "iconic": t["iconic"], "taxon": t["taxon"], "common": t.get("common"),
        "topaz_mask_area_frac": topaz_frac, "topaz_degenerate": topaz_degenerate,
        "topaz_error": t.get("error"),
        "informed_prompt": s.get("informed_prompt"), "informed_score": inf.get("top_score"),
        "generic_prompt": s.get("generic_prompt"), "generic_score": gen.get("top_score"),
        "sam3_hit": sam3_hit, "sam3_error": s.get("error"),
    })

(BASE / "results_general" / "merged.json").write_text(json.dumps(rows, indent=2))

by_group = defaultdict(list)
for r in rows:
    by_group[r["iconic"]].append(r)

lines = [f"# SAM3 vs Topaz subject mask -- {len(rows)}-photo cross-taxon batch (2026-10-23)\n",
         "Two SAM3 prompts per photo: 'informed' (the photo's own iNat common name -- the "
         "realistic production case once a species is known) and 'generic' (a fixed "
         "per-group noun phrase, e.g. 'plant'/'mushroom'/'bird' -- the pre-ID triage case).\n"]

n = len(rows)
n_topaz_bad = sum(1 for r in rows if r["topaz_degenerate"] or r["topaz_error"])
n_sam3_hit = sum(1 for r in rows if r["sam3_hit"])
lines.append(f"**Overall**: Topaz degenerate/error {n_topaz_bad}/{n} ({100*n_topaz_bad/n:.0f}%) | "
             f"SAM3 hit (informed or generic) {n_sam3_hit}/{n} ({100*n_sam3_hit/n:.0f}%)\n")

lines.append("| group | n | Topaz degenerate | SAM3 hit (any) | SAM3 informed avg score | SAM3 generic avg score |")
lines.append("|---|---|---|---|---|---|")
for g, grp in sorted(by_group.items(), key=lambda kv: -len(kv[1])):
    gn = len(grp)
    gbad = sum(1 for r in grp if r["topaz_degenerate"] or r["topaz_error"])
    ghit = sum(1 for r in grp if r["sam3_hit"])
    inf_scores = [r["informed_score"] for r in grp if r["informed_score"] is not None]
    gen_scores = [r["generic_score"] for r in grp if r["generic_score"] is not None]
    inf_avg = sum(inf_scores) / len(inf_scores) if inf_scores else 0
    gen_avg = sum(gen_scores) / len(gen_scores) if gen_scores else 0
    lines.append(f"| {g} | {gn} | {gbad}/{gn} | {ghit}/{gn} | {inf_avg:.3f} | {gen_avg:.3f} |")

lines.append("\n## Full per-photo table\n")
lines.append("| iconic | taxon | common | Topaz area | SAM3 informed (prompt->score) | SAM3 generic (prompt->score) |")
lines.append("|---|---|---|---|---|---|")
for r in sorted(rows, key=lambda r: (r["iconic"], -(r["informed_score"] or 0))):
    ta = f"{r['topaz_mask_area_frac']:.3f}" if r["topaz_mask_area_frac"] is not None else "ERR"
    inf = f"{r['informed_prompt']!r}->{r['informed_score']:.3f}" if r["informed_score"] is not None else "-"
    gen = f"{r['generic_prompt']!r}->{r['generic_score']:.3f}" if r["generic_score"] is not None else "-"
    lines.append(f"| {r['iconic']} | {r['taxon']} | {r['common'] or ''} | {ta} | {inf} | {gen} |")

report = "\n".join(lines)
(BASE / "results_general" / "REPORT.md").write_text(report)
print("\n".join(lines[:30]))
