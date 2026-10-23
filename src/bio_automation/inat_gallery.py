"""Show the proposed (not yet submitted) iNat photos in the PiGallery2 gallery, grouped by what the BioCLIP 2 suggestions say they are.

Layout, under one top-level folder that is NOT a ``<year>/<year-month>`` folder (the Bluesky poster draws from those and must never see iNat's
2048 px upload copies): ``inat-pending/<predicted iconic group>/g<group>[_<Species>_a<accuracy %>]/<file>``. The folder names carry the
suggestion, so no file is modified. A species is named only when its MEASURED accuracy reaches ``min_accuracy`` (default 0.6), and that accuracy is
part of the name. It comes from whichever source is better calibrated for this group: the nearest-neighbour vote over the owner's own photos
(suggest_bins.json) or the whole-taxonomy zero-shot probability (zeroshot_bins.json, per observation). The iconic folder is the named species' group;
without a named species it is the nearest-neighbour iconic vote when it reaches ``min_share``, else ``Unsure``.

Files are HARDLINKS of the upload copies (no extra disk, the copies untouched); hardlinks need the same volume as the copies. Nothing already in
the gallery is replaced, and only links this module made (its manifest) are ever pruned. A change publishes ONE ``gallery.reindex`` message
naming every folder it touched, which the gallery worker turns into PiGallery2 indexing.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

ROOT_NAME = "inat-pending"
UNSURE = "Unsure"


class GalleryLinkError(RuntimeError):
    pass


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-")


def best_species(suggestion: dict, min_accuracy: float) -> dict | None:
    """The better-calibrated top species of the two sources (the nearest-neighbour vote in "species", the whole-taxonomy zero-shot in "zero_shot"),
    when its MEASURED accuracy reaches *min_accuracy*; a tie goes to the nearest-neighbour vote. None when neither qualifies."""
    cands = []
    for key in ("species", "zero_shot"):
        top = (suggestion.get(key) or [None])[0]
        if top and top.get("measured_accuracy") is not None and top["measured_accuracy"] >= min_accuracy:
            cands.append(top)
    return max(cands, key=lambda c: c["measured_accuracy"]) if cands else None


def group_folder(group, suggestion: dict | None, min_share: float = 0.5, min_accuracy: float = 0.6) -> str:
    """'inat-pending/<Iconic or Unsure>/g<group>[_<Species>_a<accuracy%>]' for one proposed observation. The species is named when its measured accuracy
    reaches min_accuracy and the iconic folder is that species' group; otherwise the iconic folder is the nearest-neighbour iconic vote when it reaches
    min_share, else Unsure."""
    iconic, species = UNSURE, ""
    if suggestion:
        best = best_species(suggestion, min_accuracy)
        if best:
            species = f"_{_slug(best['taxon'])}_a{round(best['measured_accuracy'] * 100)}"
            iconic = _slug(best.get("iconic") or "") or UNSURE
        if iconic == UNSURE:
            top_i = (suggestion.get("iconic") or [None])[0]
            if top_i and top_i["vote_share"] >= min_share:
                iconic = _slug(top_i["iconic"]) or UNSURE
    return f"{ROOT_NAME}/{iconic}/g{int(group):04d}{species}"


def plan(queue: list[dict], suggestions: dict, min_share: float = 0.5, min_accuracy: float = 0.6) -> tuple[dict[str, str], list[dict]]:
    """({gallery-relative path: source copy path}, [queue entries skipped because their copy is not on disk]).
    Two copies with the same file name in one group folder is an error: the second would silently shadow the first."""
    links: dict[str, str] = {}
    skipped: list[dict] = []
    for item in queue:
        copy = item.get("copy")
        if not copy or not os.path.exists(copy):
            skipped.append(item)
            continue
        rel = f"{group_folder(item['group'], suggestions.get(str(item['group'])), min_share, min_accuracy)}/{Path(copy).name}"
        if rel in links and not os.path.samefile(links[rel], copy):
            raise GalleryLinkError(f"two different upload copies would both be {rel}: {links[rel]} and {copy}")
        links[rel] = copy
    return links, skipped


def _folders_of(rels) -> list[str]:
    out = {ROOT_NAME}
    for rel in rels:
        parts = rel.split("/")[:-1]
        for i in range(2, len(parts) + 1):
            out.add("/".join(parts[:i]))
    return sorted(out)


def apply(links: dict[str, str], gallery_dir: Path, manifest_path: Path, *, queue=None, prune: bool = True) -> dict:
    """Hardlink every planned file into the gallery; with *prune*, remove the links an earlier run made that the plan no longer contains
    (a regrouped observation moves, a submitted one leaves). Returns counts plus the refused paths; publishes one gallery.reindex message when
    anything changed and *queue* is given."""
    gallery_dir, manifest_path = Path(gallery_dir), Path(manifest_path)
    old = json.load(open(manifest_path, encoding="utf-8")) if manifest_path.exists() else {}
    made: list[str] = []
    refused: dict[str, str] = {}
    for rel, src in links.items():
        dst = gallery_dir / rel
        if dst.exists():
            if not os.path.samefile(src, dst):
                refused[rel] = f"the gallery already holds a different file here ({dst.stat().st_size} bytes); not replaced"
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src, dst)
        except OSError as e:
            raise GalleryLinkError(f"cannot hardlink {src} -> {dst}: {e} (a hardlink needs the same volume as the upload copy)") from e
        made.append(rel)
    removed: list[str] = []
    kept_old: dict[str, str] = {}
    for rel, src in old.items():
        if rel in links:
            continue
        dst = gallery_dir / rel
        if not prune:
            kept_old[rel] = src
        elif dst.exists() and (not os.path.exists(src) or os.path.samefile(src, dst)):     # ours: still the link of that copy (or its copy is gone)
            dst.unlink()
            removed.append(rel)
            for parent in list(dst.parents)[:2]:                   # the group folder, then its iconic folder, when they are empty
                if parent != gallery_dir / ROOT_NAME and parent.is_dir() and not any(parent.iterdir()):
                    parent.rmdir()
    new_manifest = {**kept_old, **{rel: src for rel, src in links.items() if rel not in refused}}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    tmp.write_text(json.dumps(new_manifest, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, manifest_path)
    published = None
    if queue is not None and (made or removed):
        folders = [f for f in _folders_of(made + removed) if (gallery_dir / f).is_dir()]    # a folder just pruned away cannot be indexed
        published = queue.publish("gallery.reindex", {"gallery_dir": str(gallery_dir), "folders": folders})
    return {"linked": len(made), "already_there": len(links) - len(made) - len(refused), "removed": len(removed),
            "refused": refused, "reindex_message": published}
