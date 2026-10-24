"""Merge the Topaz and SAM3 batch results into one comparison table + markdown report."""
import json
from pathlib import Path

BASE = Path(r"D:\dev\bio-sr-train\tools\sr_train\sam3_eval")
topaz = json.loads((BASE / "results/topaz/_summary.json").read_text(encoding="utf-8"))
sam3 = json.loads((BASE / "results/sam3/_summary.json").read_text(encoding="utf-8"))

topaz_by_obs = {r["obs_id"]: r for r in topaz}
sam3_by_obs = {r["obs_id"]: r for r in sam3}

TRUE_SPIDER_GENERA_HINTS = None  # decided per-row by iconic/taxon rank heuristics below

rows = []
for obs_id, t in topaz_by_obs.items():
    s = sam3_by_obs.get(obs_id)
    if s is None:
        continue
    topaz_frac = t.get("mask_area_frac")
    topaz_degenerate = topaz_frac is not None and topaz_frac > 0.85
    sp = s["prompts"].get("spider", {})
    ar = s["prompts"].get("arachnid", {})
    rows.append({
        "obs_id": obs_id,
        "taxon": t["taxon"],
        "common": t.get("common"),
        "topaz_mask_area_frac": topaz_frac,
        "topaz_degenerate": topaz_degenerate,
        "topaz_error": t.get("error"),
        "sam3_spider_score": sp.get("top_score"),
        "sam3_spider_area_frac": sp.get("mask_area_frac"),
        "sam3_arachnid_score": ar.get("top_score"),
        "sam3_any_hit": (sp.get("n_instances", 0) > 0) or (ar.get("n_instances", 0) > 0),
        "sam3_error": s.get("error"),
    })

n = len(rows)
n_topaz_degenerate = sum(1 for r in rows if r["topaz_degenerate"])
n_topaz_error = sum(1 for r in rows if r["topaz_error"])
n_sam3_hit = sum(1 for r in rows if r["sam3_any_hit"])
n_sam3_error = sum(1 for r in rows if r["sam3_error"])
n_both_good = sum(1 for r in rows if r["sam3_any_hit"] and not r["topaz_degenerate"] and not r["topaz_error"])
n_sam3_wins = sum(1 for r in rows if r["sam3_any_hit"] and (r["topaz_degenerate"] or r["topaz_error"]))
n_topaz_wins = sum(1 for r in rows if not r["sam3_any_hit"] and not r["topaz_degenerate"] and not r["topaz_error"])
n_both_fail = sum(1 for r in rows if not r["sam3_any_hit"] and (r["topaz_degenerate"] or r["topaz_error"]))

(BASE / "results" / "merged.json").write_text(json.dumps(rows, indent=2))

lines = []
lines.append(f"# SAM3 vs Topaz subject mask -- {n}-photo spider/arachnid batch (2026-10-23)\n")
lines.append(f"- Topaz degenerate (mask covers >85% of frame, i.e. failed to isolate subject): "
              f"**{n_topaz_degenerate}/{n}** ({100*n_topaz_degenerate/n:.0f}%)")
lines.append(f"- Topaz hard errors: {n_topaz_error}/{n}")
lines.append(f"- SAM3 found the subject (\"spider\" or \"arachnid\" prompt, confidence>0.5): "
              f"**{n_sam3_hit}/{n}** ({100*n_sam3_hit/n:.0f}%)")
lines.append(f"- SAM3 hard errors: {n_sam3_error}/{n}")
lines.append(f"- Both worked: {n_both_good}  |  SAM3 worked where Topaz failed: **{n_sam3_wins}**  |  "
              f"Topaz worked, SAM3 missed: {n_topaz_wins}  |  Both failed: {n_both_fail}\n")

lines.append("| obs_id | taxon | common | Topaz mask area | SAM3 spider score | SAM3 arachnid score | verdict |")
lines.append("|---|---|---|---|---|---|---|")
for r in sorted(rows, key=lambda r: (r["topaz_degenerate"] is not True, -(r["sam3_spider_score"] or 0))):
    topaz_bad = r["topaz_degenerate"] or r["topaz_error"]
    verdict = "SAM3 wins" if (r["sam3_any_hit"] and topaz_bad) else \
              "both ok" if (r["sam3_any_hit"] and not topaz_bad) else \
              "both failed" if (not r["sam3_any_hit"] and topaz_bad) else \
              "Topaz ok, SAM3 missed"
    ta = f"{r['topaz_mask_area_frac']:.3f}" if r["topaz_mask_area_frac"] is not None else "ERR"
    sp = f"{r['sam3_spider_score']:.3f}" if r["sam3_spider_score"] is not None else "-"
    ar = f"{r['sam3_arachnid_score']:.3f}" if r["sam3_arachnid_score"] is not None else "-"
    lines.append(f"| {r['obs_id']} | {r['taxon']} | {r['common'] or ''} | {ta} | {sp} | {ar} | {verdict} |")

report = "\n".join(lines)
(BASE / "results" / "REPORT.md").write_text(report)
print(report)
