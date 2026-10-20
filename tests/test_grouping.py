"""Unit tests for bio_automation.grouping (hermetic -- the DA-CLIP ONNX
session is mocked the same way tests/test_daclip.py mocks it; no real
model is loaded here)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from bio_automation import grouping


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _write_tiny_png(path: Path, value: int = 128) -> Path:
    """Write a real, tiny, Pillow-openable PNG -- daclip.embed_path() calls
    Image.open() on the path, so the file must be a real image even though
    the ONNX session itself is mocked."""
    Image.new("RGB", (4, 4), color=(value, value, value)).save(path)
    return path


def _mock_session(outputs: list[np.ndarray]) -> MagicMock:
    """Return a mock ort.InferenceSession whose run() yields each array in
    *outputs* in order, one per call -- mirrors test_daclip.py's
    _mock_session, extended to a sequence of calls instead of one fixed
    return value."""
    sess = MagicMock()
    sess.get_inputs.return_value = [MagicMock(name="image")]
    sess.run.side_effect = [[output] for output in outputs]
    return sess


def _vecs(content: np.ndarray) -> np.ndarray:
    """Build a (2, 512) DA-CLIP output row-pair: row 0 = content (the
    vector under test), row 1 = an arbitrary degradation vector that
    check_subject_consistency() must ignore."""
    degrad = np.full(512, 0.5, dtype=np.float32)
    return np.stack([content, degrad]).astype(np.float32)


def _unit(axis: int, dims: int = 512) -> np.ndarray:
    vec = np.zeros(dims, dtype=np.float32)
    vec[axis] = 1.0
    return vec


# ---------------------------------------------------------------------------
# cosine_similarity()
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors_similarity_is_one(self):
        a = _unit(0)
        assert grouping.cosine_similarity(a, a) == 1.0

    def test_orthogonal_vectors_similarity_is_zero(self):
        """Negative/contrasting twin to the identical-vectors case above:
        a different, discriminating fixture must produce a different,
        exact result."""
        a, b = _unit(0), _unit(1)
        assert grouping.cosine_similarity(a, b) == 0.0

    def test_opposite_vectors_similarity_is_minus_one(self):
        a, b = _unit(0), -_unit(0)
        assert grouping.cosine_similarity(a, b) == -1.0

    def test_scale_invariant(self):
        """Positive twin to the identical-vectors case: same direction,
        different magnitude, must still read as perfectly similar."""
        a = _unit(0)
        b = a * 37.0
        assert grouping.cosine_similarity(a, b) == pytest.approx(1.0)

    def test_zero_vector_returns_zero_not_a_crash(self):
        a = np.zeros(512, dtype=np.float32)
        b = _unit(0)
        assert grouping.cosine_similarity(a, b) == 0.0


# ---------------------------------------------------------------------------
# check_subject_consistency()
# ---------------------------------------------------------------------------

class TestCheckSubjectConsistency:
    def test_empty_raw_paths_raises(self):
        sess = _mock_session([])
        with pytest.raises(ValueError, match="at least one raw path"):
            grouping.check_subject_consistency([], sess)

    def test_all_similar_frames_returns_empty_list(self, tmp_path):
        """Positive twin to the divergent-frame case below: three frames
        that all embed to (near-)identical content vectors must produce no
        outliers at all."""
        paths = [_write_tiny_png(tmp_path / f"f{i}.png") for i in range(3)]
        same = _unit(0)
        sess = _mock_session([_vecs(same), _vecs(same), _vecs(same)])

        result = grouping.check_subject_consistency(paths, sess)

        assert result == []
        assert sess.run.call_count == 3

    def test_one_divergent_frame_is_flagged_with_similarity_score(self, tmp_path):
        """Negative twin to the all-similar case above: the median-position
        frame (index 1 of 3) is the reference; frame 0 matches it, frame 2
        is orthogonal to it and must be flagged with the exact similarity
        score computed."""
        p0 = _write_tiny_png(tmp_path / "f0.png")
        p1 = _write_tiny_png(tmp_path / "f1.png")
        p2 = _write_tiny_png(tmp_path / "f2.png")
        reference_vec = _unit(0)
        divergent_vec = _unit(1)
        sess = _mock_session([_vecs(reference_vec), _vecs(reference_vec), _vecs(divergent_vec)])

        result = grouping.check_subject_consistency([p0, p1, p2], sess)

        assert result == [(p2, 0.0)]

    def test_frame_exactly_at_threshold_is_not_flagged(self, tmp_path):
        """Boundary case: a similarity equal to the threshold is not
        'below' it, so it must not be flagged."""
        p0 = _write_tiny_png(tmp_path / "f0.png")
        p1 = _write_tiny_png(tmp_path / "f1.png")
        ref = _unit(0)
        sess = _mock_session([_vecs(ref), _vecs(ref)])

        result = grouping.check_subject_consistency([p0, p1], sess, threshold=1.0)

        assert result == []

    def test_custom_threshold_is_honored(self, tmp_path):
        """Positive/negative pair driven by *threshold* itself rather than
        the embeddings: the same two frames are flagged at a strict
        threshold and not flagged at a lax one."""
        p0 = _write_tiny_png(tmp_path / "f0.png")
        p1 = _write_tiny_png(tmp_path / "f1.png")
        ref = _unit(0)
        other = (_unit(0) * 0.9 + _unit(1) * 0.1)

        # len([p0, p1]) // 2 == 1, so p1's own embedding is the reference;
        # p1 trivially matches itself (similarity 1.0) and is never
        # flagged -- p0 is the one compared against it and can be.
        sess_strict = _mock_session([_vecs(ref), _vecs(other)])
        flagged = grouping.check_subject_consistency([p0, p1], sess_strict, threshold=0.999)
        assert [path for path, _ in flagged] == [p0]

        sess_lax = _mock_session([_vecs(ref), _vecs(other)])
        not_flagged = grouping.check_subject_consistency([p0, p1], sess_lax, threshold=0.5)
        assert not_flagged == []

    def test_reference_is_median_position_not_first(self, tmp_path):
        """Confirms the reference is picked by *position* (index
        len//2), matching preclean.compute_fixed_params()'s convention --
        not the first frame, which this fixture deliberately makes the
        odd one out so the two tests can't accidentally agree."""
        p0 = _write_tiny_png(tmp_path / "f0.png")
        p1 = _write_tiny_png(tmp_path / "f1.png")
        p2 = _write_tiny_png(tmp_path / "f2.png")
        p3 = _write_tiny_png(tmp_path / "f3.png")
        p4 = _write_tiny_png(tmp_path / "f4.png")
        odd_one_out = _unit(0)
        common = _unit(1)
        # index 2 (len(5)//2) is the reference: everyone else matches it.
        sess = _mock_session([_vecs(odd_one_out), _vecs(common), _vecs(common),
                               _vecs(common), _vecs(common)])

        result = grouping.check_subject_consistency([p0, p1, p2, p3, p4], sess)

        assert result == [(p0, 0.0)]

    def test_session_called_once_per_path(self, tmp_path):
        paths = [_write_tiny_png(tmp_path / f"f{i}.png") for i in range(4)]
        vec = _unit(0)
        sess = _mock_session([_vecs(vec)] * 4)

        grouping.check_subject_consistency(paths, sess)

        assert sess.run.call_count == 4


# ---------------------------------------------------------------------------
# SubjectMismatchError
# ---------------------------------------------------------------------------

class TestSubjectMismatchError:
    def test_message_names_offending_files_and_scores(self):
        outliers = [(Path("a.NEF"), 0.1), (Path("b.NEF"), 0.42)]
        exc = grouping.SubjectMismatchError(outliers, threshold=0.75)
        message = str(exc)
        assert "a.NEF" in message
        assert "0.100" in message
        assert "b.NEF" in message
        assert "0.420" in message
        assert "0.75" in message

    def test_stores_outliers_and_threshold_as_attributes(self):
        outliers = [(Path("a.NEF"), 0.1)]
        exc = grouping.SubjectMismatchError(outliers, threshold=0.6)
        assert exc.outliers == outliers
        assert exc.threshold == 0.6
