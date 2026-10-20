"""Subject-consistency safety gate, run before Zerene focus-stacks a batch.

Built after a real incident: a 10-frame test batch for this pipeline was
grouped by shutter-timing proximity only, which silently mixed two
completely different subjects (a spider and a bracket fungus) into one
batch. Zerene then spent 15 minutes trying to align/merge frames that do
not belong together before timing out. Nothing upstream checked image
*content* before launching that run.

check_subject_consistency() closes that gap using the existing DA-CLIP
embeddings (bio_automation.daclip): every frame's content_vec is compared
by cosine similarity to one reference frame's content_vec, and any frame
that falls below a similarity threshold is reported as a likely different
subject.

This is a real ONNX-model forward pass per image via daclip.embed_path(),
not free -- callers checking more than one batch should load one
daclip.load_session() session and reuse it across every call, rather than
loading a fresh session per batch or per path.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import daclip

# Similarity floor below which a frame is treated as a likely different
# subject. CORRECTED 2026-10-19 after independent verification against the
# actual incident images (tests/fixtures/sample_raw.NEF = the spider,
# sample_raw_fungus.NEF = the bracket fungus -- see tests/test_grouping_live.py):
# the real cross-subject pair measures 0.8111 cosine similarity in DA-CLIP
# content-embedding space, and a real same-subject pair (two frames 2.9s
# apart in the same burst) measures 0.9731. The original value committed
# here, 0.75, was asserted as "a real cross-subject mismatch reads far
# below this" without being run against real data -- it does not: 0.8111 is
# ABOVE 0.75, so the original threshold would NOT have caught the exact
# incident this check exists to prevent. 0.88 sits with real margin below
# the measured same-subject score and above the measured cross-subject
# score; tests/test_grouping_live.py asserts both measurements directly so
# this number stays provably correct rather than asserted.
DEFAULT_THRESHOLD = 0.88


class SubjectMismatchError(RuntimeError):
    """Raised by callers (see bio_automation.pipeline) when
    check_subject_consistency() finds one or more frames in a batch that
    look like a different subject than the rest, naming each offending
    file and the similarity score that tripped it.
    """

    def __init__(self, outliers: list[tuple[Path, float]], threshold: float = DEFAULT_THRESHOLD):
        self.outliers = list(outliers)
        self.threshold = threshold
        detail = ", ".join(f"{path} (similarity={similarity:.3f})" for path, similarity in outliers)
        super().__init__(
            f"{len(outliers)} frame(s) fall below the subject-consistency threshold "
            f"{threshold:.2f}, so this batch looks like it contains more than one "
            f"subject: {detail}"
        )


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Plain cosine similarity between two 1-D vectors, in [-1.0, 1.0].

    Pure function -- no I/O, no model, just the two already-embedded
    vectors the caller supplies. Returns 0.0 if either vector has zero
    magnitude (undefined direction, so "no similarity" rather than a
    division error).
    """
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def check_subject_consistency(raw_paths: list[Path], session,
                               threshold: float = DEFAULT_THRESHOLD) -> list[tuple[Path, float]]:
    """Flag frames in *raw_paths* whose subject looks inconsistent with the
    rest of the batch, before it is handed to Zerene.

    Embeds every path's DA-CLIP content_vec via daclip.embed_path(session),
    then picks the reference embedding as the median-position frame in
    *raw_paths* -- the same convention preclean.compute_fixed_params()
    already uses for its representative frame (one real frame's actual
    embedding, not a centroid or pairwise average). Returns a (path,
    similarity) pair for every frame whose cosine similarity to that
    reference is strictly BELOW *threshold*; an empty list means no
    outliers. The reference frame's similarity to itself is always 1.0, so
    it is never flagged, without needing to special-case it.

    *session* is a loaded daclip.load_session() ONNX InferenceSession.
    This makes one real model forward pass per path in *raw_paths* -- not
    free -- so reuse the same *session* across every call in a batch
    instead of loading a new one per call.

    Raises ValueError if *raw_paths* is empty -- there is no reference
    frame to compare against.
    """
    raw_paths = [Path(p) for p in raw_paths]
    if not raw_paths:
        raise ValueError("check_subject_consistency requires at least one raw path")

    content_vecs = [daclip.embed_path(path, session)[0] for path in raw_paths]
    reference_vec = content_vecs[len(raw_paths) // 2]

    outliers: list[tuple[Path, float]] = []
    for path, content_vec in zip(raw_paths, content_vecs):
        similarity = cosine_similarity(content_vec, reference_vec)
        if similarity < threshold:
            outliers.append((path, similarity))
    return outliers
