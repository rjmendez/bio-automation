"""Unit tests for bio_automation.inat_queue (pure functions, no I/O, no
stubbing needed)."""
from __future__ import annotations

from bio_automation.inat_queue import QueueItem, group_photos, presort


def _photo(path, day, t, camera="Nikon Z8", src="F", lat=None, lon=None):
    return {"path": path, "day": day, "t": t, "camera": camera, "src": src, "lat": lat, "lon": lon}


class TestGroupPhotos:
    def test_splits_same_day_same_camera_photos_into_two_bursts_on_a_time_gap(self):
        photos = [
            _photo("a.jpg", "2026-09-01", 1000.0),
            _photo("b.jpg", "2026-09-01", 1050.0),
            _photo("c.jpg", "2026-09-01", 5000.0),
        ]

        groups = group_photos(photos, burst_gap_seconds=120.0)

        assert [g.group for g in groups] == [1, 2]
        assert groups[0].photos == ["a.jpg", "b.jpg"]
        assert groups[1].photos == ["c.jpg"]

    def test_photos_exactly_at_the_gap_boundary_stay_in_the_same_group(self):
        photos = [_photo("a.jpg", "2026-09-01", 1000.0), _photo("b.jpg", "2026-09-01", 1120.0)]

        groups = group_photos(photos, burst_gap_seconds=120.0)

        assert len(groups) == 1
        assert groups[0].photos == ["a.jpg", "b.jpg"]

    def test_different_cameras_on_the_same_day_never_share_a_group(self):
        photos = [
            _photo("a.jpg", "2026-09-01", 1000.0, camera="Nikon Z8"),
            _photo("b.jpg", "2026-09-01", 1001.0, camera="iPhone 15"),
        ]

        groups = group_photos(photos, burst_gap_seconds=120.0)

        assert sorted(g.camera for g in groups) == ["Nikon Z8", "iPhone 15"]
        assert all(len(g.photos) == 1 for g in groups)

    def test_a_large_location_jump_within_the_time_gap_still_splits_the_group(self):
        """Two photos close in time but far apart in space (GPS fix moved,
        or a bad neighbour-location inference) must not be merged into one
        proposed observation."""
        photos = [
            _photo("a.jpg", "2026-09-01", 1000.0, lat=37.9, lon=-122.5),
            _photo("b.jpg", "2026-09-01", 1010.0, lat=40.7, lon=-74.0),
        ]

        groups = group_photos(photos, burst_gap_seconds=120.0, max_location_drift_m=50.0)

        assert len(groups) == 2

    def test_close_locations_within_drift_stay_merged(self):
        """Positive twin of the location-jump test: a small, within-GPS-
        noise difference keeps the burst together."""
        photos = [
            _photo("a.jpg", "2026-09-01", 1000.0, lat=37.90000, lon=-122.50000),
            _photo("b.jpg", "2026-09-01", 1010.0, lat=37.90005, lon=-122.50005),
        ]

        groups = group_photos(photos, burst_gap_seconds=120.0, max_location_drift_m=50.0)

        assert len(groups) == 1
        assert groups[0].photos == ["a.jpg", "b.jpg"]

    def test_group_numbers_run_once_across_the_whole_sorted_queue_not_per_bucket(self):
        photos = [
            _photo("a.jpg", "2026-09-01", 1000.0, camera="Nikon Z8"),
            _photo("b.jpg", "2026-09-01", 1001.0, camera="iPhone 15"),
            _photo("c.jpg", "2026-09-02", 1000.0, camera="Nikon Z8"),
        ]

        groups = group_photos(photos, burst_gap_seconds=120.0)

        assert sorted(g.group for g in groups) == [1, 2, 3]

    def test_empty_photo_list_yields_no_groups(self):
        assert group_photos([], burst_gap_seconds=120.0) == []

    def test_rerunning_on_the_same_input_reproduces_the_same_group_numbers(self):
        photos = [
            _photo("b.jpg", "2026-09-01", 1050.0),
            _photo("a.jpg", "2026-09-01", 1000.0),
        ]

        first = group_photos(photos, burst_gap_seconds=120.0)
        second = group_photos(list(reversed(photos)), burst_gap_seconds=120.0)

        assert [g.group for g in first] == [g.group for g in second]
        assert [g.photos for g in first] == [g.photos for g in second]


class TestPresort:
    def _items(self):
        return [
            QueueItem(group=1, day="2026-09-01", camera="Nikon Z8", src="F", photos=["a.jpg"]),
            QueueItem(group=2, day="2026-09-01", camera="Nikon Z8", src="F", photos=["b.jpg"]),
            QueueItem(group=3, day="2026-09-01", camera="Nikon Z8", src="F", photos=["c.jpg"]),
            QueueItem(group=4, day="2026-09-01", camera="Nikon Z8", src="F", photos=["d.jpg"]),
        ]

    def test_buckets_items_by_their_recorded_decision(self):
        decisions = {1: {"status": "approved"}, 2: {"status": "skipped"}, 3: {"status": "hold"}}

        buckets = presort(self._items(), decisions)

        assert buckets == {"keep": [1], "cull": [2], "hold": [3], "undecided": [4]}

    def test_an_explicit_pending_decision_lands_in_undecided_same_as_no_decision(self):
        """Positive/negative twin on the same fixture: a recorded "pending"
        row and a missing row both mean "no human decision yet"."""
        decisions = {1: {"status": "pending"}}

        buckets = presort(self._items(), decisions)

        assert buckets["undecided"] == [1, 2, 3, 4]

    def test_no_decisions_at_all_puts_everything_in_undecided(self):
        buckets = presort(self._items(), {})

        assert buckets["undecided"] == [1, 2, 3, 4]
        assert buckets["keep"] == []
        assert buckets["cull"] == []
        assert buckets["hold"] == []

    def test_presort_never_calls_into_decision_recording(self):
        """presort must only ever read; it has no business writing a
        decision. There is no decision-writer to stub here (inat_ledger's
        record_decision is a separate module presort does not import), so
        this instead proves presort is pure: the same decisions dict
        passed in comes back unmodified."""
        decisions = {1: {"status": "approved"}}
        decisions_copy = dict(decisions)

        presort(self._items(), decisions)

        assert decisions == decisions_copy
