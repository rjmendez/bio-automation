"""Live, unmocked proof that grouping.DEFAULT_THRESHOLD actually separates a
real cross-subject pair from a real same-subject pair -- not asserted, measured.

Why this file exists: grouping.py's subject-consistency gate was built
specifically because a real 10-frame batch silently mixed a spider and a
bracket fungus (see grouping.py's module docstring). The threshold that
first shipped (0.75) was justified in a code comment as "a real
cross-subject mismatch reads far below this" -- but that claim was never
run against the actual incident images. It was wrong: the real spider/
fungus pair measures 0.8111, which is ABOVE 0.75, so the original
threshold would not have caught the exact incident it exists to prevent.
Mocked unit tests (tests/test_grouping.py) can't catch this class of bug
by construction -- a mock's similarity values are whatever the test author
picked, so a wrong real-world threshold and a right one look identical to
a test built on synthetic vectors. Only real embeddings of the real
incident images can prove the threshold is actually correct.

Deliberately not gated behind an opt-in env var (see AGENTS.md rule 10 and
tests/test_preclean_live.py's docstring for why) -- runs on every default
`pytest` invocation.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from bio_automation import daclip
from bio_automation.grouping import (
    DEFAULT_THRESHOLD,
    check_subject_consistency,
    cosine_similarity,
)

SPIDER = Path(__file__).parent / "fixtures" / "sample_raw.NEF"
FUNGUS = Path(__file__).parent / "fixtures" / "sample_raw_fungus.NEF"


@pytest.fixture(scope="module")
def session():
    return daclip.load_session()


class TestRealThresholdSeparatesRealIncidentImages:
    def test_real_cross_subject_pair_falls_below_threshold(self, session):
        """The actual spider-vs-fungus pair from the real incident this
        check exists to catch."""
        v_spider, _ = daclip.embed_path(SPIDER, session)
        v_fungus, _ = daclip.embed_path(FUNGUS, session)

        similarity = cosine_similarity(v_spider, v_fungus)

        assert similarity < DEFAULT_THRESHOLD, (
            f"real cross-subject similarity {similarity:.4f} is not below "
            f"DEFAULT_THRESHOLD={DEFAULT_THRESHOLD} -- the threshold would "
            f"not catch the real incident this check exists to prevent"
        )

    def test_check_subject_consistency_flags_the_real_fungus_frame(self, session):
        """End-to-end through the real public function, not just the raw
        cosine_similarity math: a 3-frame batch where 2 are the spider and
        1 is the fungus must flag exactly the fungus frame."""
        outliers = check_subject_consistency([SPIDER, SPIDER, FUNGUS], session)

        outlier_paths = [path for path, _ in outliers]
        assert outlier_paths == [FUNGUS]

    def test_all_same_subject_batch_flags_nothing(self, session):
        """Positive twin: a batch of only the spider frame (repeated) must
        not flag itself."""
        outliers = check_subject_consistency([SPIDER, SPIDER, SPIDER], session)
        assert outliers == []
