"""
Classify rendered stock photos via DA-CLIP degradation embeddings.

Reads triage_selection.json, finds each rendered JPEG in BIO_RENDER_OUT,
runs DACLIP to get content + degradation vectors, clusters degradation
vectors into 4 groups (clean/lens_blur/noise/mixed), and writes:

  daclip_triage.json  — per-image {id, path, content_vec, degrad_vec, cluster, route}

where route is one of: skip | sharpen | denoise_sharpen | sharpen_strong

Env vars:
  BIO_RENDER_OUT   base path of rendered JPEGs (default: F:\\Bio-export\\adobe-stock\\rendered_new)
  BIO_DACLIP_K     number of clusters (default: 4)
  BIO_DACLIP_OUT   output JSON path (default: {ROOT}/daclip_triage.json)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUT_BASE = os.environ.get("BIO_RENDER_OUT", r"F:\Bio-export\adobe-stock\rendered_new")
K = int(os.environ.get("BIO_DACLIP_K", "4"))
TRIAGE_OUT = os.environ.get("BIO_DACLIP_OUT", os.path.join(ROOT, "daclip_triage.json"))
LOG_PATH = os.path.join(ROOT, "daclip_triage.log")


def log(msg: str) -> None:
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _route_from_norm(norm: float) -> str:
    """Map degradation-vector L2-norm to a routing label.

    Thresholds are intentionally conservative; save full vectors so they
    can be re-calibrated once real labelled data is available.
    """
    if norm < 0.3:
        return "skip"
    if norm <= 0.6:
        return "sharpen"
    return "sharpen_strong"


def _kmeans_numpy(X: np.ndarray, k: int, n_init: int = 5, max_iter: int = 100) -> np.ndarray:
    """Simple k-means returning cluster labels for each row in X."""
    rng = np.random.default_rng(42)
    best_labels = None
    best_inertia = float("inf")
    for _ in range(n_init):
        idx = rng.choice(len(X), k, replace=False)
        centroids = X[idx].copy()
        for _ in range(max_iter):
            diffs = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]  # [N,k,D]
            labels = np.argmin(np.sum(diffs ** 2, axis=-1), axis=1)
            new_centroids = np.array(
                [X[labels == c].mean(axis=0) if (labels == c).any() else centroids[c] for c in range(k)]
            )
            if np.allclose(new_centroids, centroids):
                break
            centroids = new_centroids
        inertia = sum(np.sum((X[labels == c] - centroids[c]) ** 2) for c in range(k) if (labels == c).any())
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
    return best_labels


def main() -> None:
    from bio_automation import daclip

    selection_path = os.path.join(os.path.dirname(ROOT), "triage_selection.json")
    if not os.path.exists(selection_path):
        # fall back to ROOT level (pipeline parent may vary)
        selection_path = os.path.join(ROOT, "triage_selection.json")

    log(f"Loading triage selection from {selection_path}")
    todo = json.load(open(selection_path, encoding="utf-8"))
    log(f"{len(todo)} entries in triage_selection.json")

    log("Loading DA-CLIP session …")
    session = daclip.load_session()
    log("DA-CLIP session ready")

    results: list[dict] = []
    skipped = 0
    for i, r in enumerate(todo, 1):
        fid = r["id"]
        safe = "".join(c if c not in '<>:"/\\|?*' else "_" for c in r["stem"])
        jpeg_path = os.path.join(OUT_BASE, r["day"][:4], r["day"], f"{safe}_{fid}.jpg")
        if not os.path.exists(jpeg_path):
            log(f"  SKIP {fid}: rendered JPEG not found at {jpeg_path}")
            skipped += 1
            continue
        try:
            content_vec, degrad_vec = daclip.embed_path(Path(jpeg_path), session)
        except Exception as exc:
            log(f"  ERROR {fid}: {type(exc).__name__} {exc}")
            skipped += 1
            continue
        norm = float(np.linalg.norm(degrad_vec))
        results.append({
            "id": fid,
            "path": jpeg_path,
            "content_vec": content_vec.tolist(),
            "degrad_vec": degrad_vec.tolist(),
            "degrad_norm": norm,
            "cluster": -1,
            "route": _route_from_norm(norm),
        })
        if i % 20 == 0:
            log(f"  {i}/{len(todo)} processed, {skipped} skipped so far")

    log(f"Embedded {len(results)} images, skipped {skipped}")

    # K-means clustering on degradation vectors (if enough data)
    if len(results) >= K:
        degrad_matrix = np.array([r["degrad_vec"] for r in results], dtype=np.float32)
        try:
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=K, n_init=5, random_state=42)
            labels = km.fit_predict(degrad_matrix).tolist()
        except ImportError:
            log("sklearn not available, falling back to numpy k-means")
            labels = _kmeans_numpy(degrad_matrix, K).tolist()
        for rec, lbl in zip(results, labels):
            rec["cluster"] = int(lbl)
        log(f"K-means (k={K}) assigned cluster labels")
    else:
        log(f"Too few images ({len(results)}) for k={K} clustering; cluster=-1 for all")

    with open(TRIAGE_OUT, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    log(f"Wrote {len(results)} records to {TRIAGE_OUT}")


if __name__ == "__main__":
    main()
