"""Unit tests for bio_automation.inat_post (hermetic -- create_fn/attach_fn
are always a recording/configured stub here; the real urllib transport is
exercised separately in TestDefaultTransport with only
urllib.request.urlopen itself stubbed)."""
from __future__ import annotations

import urllib.error

import pytest

from bio_automation.inat import RateLimiter
from bio_automation.inat_post import (
    DEFAULT_PACE_SECONDS,
    AlreadyPostedError,
    InatAuthError,
    MissingCredentialError,
    build_observation_body,
    post_approved,
    post_one,
)
from bio_automation.inat_queue import QueueItem


DECISION = {"group": 1, "status": "approved", "observed_on_string": "2026-09-01 08:00:00",
            "time_zone": "Pacific Time (US & Canada)", "lat": 38.5, "lon": -122.25,
            "geoprivacy": "obscured", "ts": 1.0}


def _item(group=1, photos=("a.jpg", "b.jpg")):
    return QueueItem(group=group, day="2026-09-01", camera="Nikon Z8", src="F", photos=list(photos))


class RecordingCreateFn:
    def __init__(self, observation_ids=None, to_raise=None):
        self.calls: list[QueueItem] = []
        self.observation_ids = observation_ids or {}
        self.to_raise = to_raise

    def __call__(self, item, jwt):
        self.calls.append(item)
        if self.to_raise is not None:
            raise self.to_raise
        return {"observation_id": self.observation_ids[item.group], "uuid": f"uuid-{item.group}"}


class RecordingAttachFn:
    def __init__(self, fail_after: int | None = None, to_raise=None):
        self.calls: list[tuple[int, str]] = []
        self.fail_after = fail_after
        self.to_raise = to_raise or RuntimeError("upload failed")

    def __call__(self, observation_id, path, jwt):
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            self.calls.append((observation_id, path))
            raise self.to_raise
        self.calls.append((observation_id, path))


class TestPostOne:
    def test_dry_run_default_never_calls_create_or_attach(self):
        create_fn, attach_fn = RecordingCreateFn(), RecordingAttachFn()

        result = post_one(_item(), jwt="fake-jwt", posted={}, create_fn=create_fn, attach_fn=attach_fn)

        assert result == {"group": 1, "status": "dry_run", "observation_id": None, "uuid": None,
                           "photos": 0, "error": None}
        assert create_fn.calls == []
        assert attach_fn.calls == []

    def test_full_success_attaches_every_photo_and_reports_posted(self):
        create_fn = RecordingCreateFn(observation_ids={1: 4242})
        attach_fn = RecordingAttachFn()

        result = post_one(_item(), jwt="fake-jwt", posted={}, create_fn=create_fn, attach_fn=attach_fn,
                           dry_run=False)

        assert result == {"group": 1, "status": "posted", "observation_id": 4242, "uuid": "uuid-1",
                           "photos": 2, "error": None}
        assert attach_fn.calls == [(4242, "a.jpg"), (4242, "b.jpg")]

    def test_photo_attach_failure_after_observation_created_is_partial_not_failed(self):
        create_fn = RecordingCreateFn(observation_ids={1: 4242})
        attach_fn = RecordingAttachFn(fail_after=1)

        result = post_one(_item(), jwt="fake-jwt", posted={}, create_fn=create_fn, attach_fn=attach_fn,
                           dry_run=False)

        assert result["status"] == "partial"
        assert result["observation_id"] == 4242
        assert result["photos"] == 1
        assert "upload failed" in result["error"]

    def test_observation_creation_failure_is_failed_with_no_observation_id(self):
        """Positive twin of the partial test above, on the same fixture
        shape: the earlier failure point (create, not attach) yields the
        other status."""
        create_fn = RecordingCreateFn(to_raise=RuntimeError("API down"))
        attach_fn = RecordingAttachFn()

        result = post_one(_item(), jwt="fake-jwt", posted={}, create_fn=create_fn, attach_fn=attach_fn,
                           dry_run=False)

        assert result["status"] == "failed"
        assert result["observation_id"] is None
        assert result["photos"] == 0
        assert attach_fn.calls == []

    def test_already_posted_raises_without_calling_create_fn(self):
        create_fn, attach_fn = RecordingCreateFn(observation_ids={1: 4242}), RecordingAttachFn()
        posted = {1: {"group": 1, "status": "posted", "observation_id": 4242}}

        with pytest.raises(AlreadyPostedError, match="group 1"):
            post_one(_item(), jwt="fake-jwt", posted=posted, create_fn=create_fn, attach_fn=attach_fn,
                     dry_run=False)

        assert create_fn.calls == []

    def test_already_partial_also_raises(self):
        """Positive twin of the "already posted" test: "partial" is just
        as done as "posted" and must also refuse a direct re-post."""
        create_fn = RecordingCreateFn(observation_ids={1: 4242})
        posted = {1: {"group": 1, "status": "partial", "observation_id": 4242}}

        with pytest.raises(AlreadyPostedError, match="group 1"):
            post_one(_item(), jwt="fake-jwt", posted=posted, create_fn=create_fn, dry_run=False)

    def test_a_failed_group_is_not_treated_as_already_posted(self):
        """Negative twin: "failed" means no observation was ever created,
        so it must NOT raise AlreadyPostedError -- it's safe to retry."""
        create_fn = RecordingCreateFn(observation_ids={1: 4242})
        attach_fn = RecordingAttachFn()
        posted = {1: {"group": 1, "status": "failed", "observation_id": None}}

        result = post_one(_item(), jwt="fake-jwt", posted=posted, create_fn=create_fn, attach_fn=attach_fn,
                           dry_run=False)

        assert result["status"] == "posted"
        assert len(create_fn.calls) == 1

    def test_an_auth_error_from_create_fn_is_caught_and_recorded_not_raised(self):
        create_fn = RecordingCreateFn(to_raise=InatAuthError("iNaturalist rejected credentials (HTTP 401)"))

        result = post_one(_item(), jwt="expired-jwt", posted={}, create_fn=create_fn, dry_run=False)

        assert result["status"] == "failed"
        assert "401" in result["error"]


class TestPostApproved:
    def test_posts_only_approved_items_in_order(self):
        items = [_item(1), _item(2), _item(3)]
        decisions = {1: {"status": "approved"}, 2: {"status": "skipped"}, 3: {"status": "hold"}}
        create_fn, attach_fn = RecordingCreateFn(observation_ids={1: 1001}), RecordingAttachFn()

        results = post_approved(items, decisions, posted={}, jwt="fake-jwt", create_fn=create_fn,
                                 attach_fn=attach_fn, dry_run=False, rate_limiter=RateLimiter(min_interval=0))

        assert [c.group for c in create_fn.calls] == [1]
        assert [r["status"] for r in results] == ["posted"]

    def test_an_already_done_approved_item_is_skipped_without_calling_create_fn(self):
        items = [_item(1)]
        decisions = {1: {"status": "approved"}}
        posted = {1: {"group": 1, "status": "posted", "observation_id": 1001}}
        create_fn = RecordingCreateFn()

        results = post_approved(items, decisions, posted, jwt="fake-jwt", create_fn=create_fn,
                                 dry_run=False, rate_limiter=RateLimiter(min_interval=0))

        assert create_fn.calls == []
        assert results == []

    def test_stops_after_the_first_problem_and_does_not_attempt_later_items(self):
        items = [_item(1), _item(2)]
        decisions = {1: {"status": "approved"}, 2: {"status": "approved"}}
        create_fn = RecordingCreateFn(to_raise=InatAuthError("iNaturalist rejected credentials (HTTP 401)"))

        results = post_approved(items, decisions, posted={}, jwt="expired-jwt", create_fn=create_fn,
                                 dry_run=False, rate_limiter=RateLimiter(min_interval=0))

        assert [c.group for c in create_fn.calls] == [1]
        assert len(results) == 1
        assert results[0]["status"] == "failed"

    def test_missing_jwt_raises_before_attempting_anything(self):
        items = [_item(1)]
        decisions = {1: {"status": "approved"}}
        create_fn = RecordingCreateFn(observation_ids={1: 1001})

        with pytest.raises(MissingCredentialError, match="INAT_JWT"):
            post_approved(items, decisions, posted={}, jwt=None, create_fn=create_fn, dry_run=False)

        assert create_fn.calls == []

    def test_dry_run_default_needs_no_jwt_and_calls_nothing(self):
        """Positive twin of the missing-jwt test: dry_run (the default)
        never needs credentials at all."""
        items = [_item(1)]
        decisions = {1: {"status": "approved"}}
        create_fn = RecordingCreateFn(observation_ids={1: 1001})

        results = post_approved(items, decisions, posted={}, jwt=None, create_fn=create_fn,
                                 rate_limiter=RateLimiter(min_interval=0))

        assert results == [{"group": 1, "status": "dry_run", "observation_id": None, "uuid": None,
                             "photos": 0, "error": None}]
        assert create_fn.calls == []

    def test_a_successful_post_is_recorded_into_the_posted_dict_in_place(self):
        items = [_item(1)]
        decisions = {1: {"status": "approved"}}
        posted: dict = {}
        create_fn = RecordingCreateFn(observation_ids={1: 1001})
        attach_fn = RecordingAttachFn()

        post_approved(items, decisions, posted, jwt="fake-jwt", create_fn=create_fn, attach_fn=attach_fn,
                      dry_run=False, rate_limiter=RateLimiter(min_interval=0))

        assert posted[1]["status"] == "posted"
        assert posted[1]["observation_id"] == 1001

    def test_dry_run_does_not_mutate_the_posted_dict(self):
        items = [_item(1)]
        decisions = {1: {"status": "approved"}}
        posted: dict = {}
        create_fn = RecordingCreateFn(observation_ids={1: 1001})

        post_approved(items, decisions, posted, jwt="fake-jwt", create_fn=create_fn, dry_run=True,
                      rate_limiter=RateLimiter(min_interval=0))

        assert posted == {}
        assert create_fn.calls == []

    def test_respects_the_limit_parameter(self):
        items = [_item(1), _item(2), _item(3)]
        decisions = {g: {"status": "approved"} for g in (1, 2, 3)}
        create_fn = RecordingCreateFn(observation_ids={1: 1, 2: 2, 3: 3})
        attach_fn = RecordingAttachFn()

        results = post_approved(items, decisions, posted={}, jwt="fake-jwt", create_fn=create_fn,
                                 attach_fn=attach_fn, dry_run=False, limit=2,
                                 rate_limiter=RateLimiter(min_interval=0))

        assert [c.group for c in create_fn.calls] == [1, 2]
        assert len(results) == 2

    def test_paces_consecutive_posts_through_the_rate_limiter(self):
        items = [_item(1), _item(2)]
        decisions = {1: {"status": "approved"}, 2: {"status": "approved"}}
        create_fn = RecordingCreateFn(observation_ids={1: 1, 2: 2})
        attach_fn = RecordingAttachFn()
        clock = iter([100.0, 100.0])
        slept: list[float] = []
        limiter = RateLimiter(min_interval=1.0, time_fn=lambda: next(clock), sleep_fn=slept.append)

        post_approved(items, decisions, posted={}, jwt="fake-jwt", create_fn=create_fn, attach_fn=attach_fn,
                      dry_run=False, rate_limiter=limiter)

        assert slept == [1.0]

    def test_default_rate_limiter_paces_at_the_legacy_1_1_second_interval(self):
        assert DEFAULT_PACE_SECONDS == 1.1


class TestDefaultTransport:
    """The real urllib transport, with only urllib.request.urlopen (the
    actual network dependency) stubbed -- not the function under test."""

    def test_401_on_create_raises_auth_error(self, monkeypatch):
        import bio_automation.inat_post as inat_post_module

        def fake_urlopen(request, timeout=120):
            raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {}, None)

        monkeypatch.setattr(inat_post_module.urllib.request, "urlopen", fake_urlopen)

        with pytest.raises(InatAuthError, match="401"):
            inat_post_module._default_create_observation(_item(), "expired-jwt", DECISION)

    def test_500_on_create_is_not_wrapped_as_an_auth_error(self, monkeypatch):
        import bio_automation.inat_post as inat_post_module

        def fake_urlopen(request, timeout=120):
            raise urllib.error.HTTPError(request.full_url, 500, "Server Error", {}, None)

        monkeypatch.setattr(inat_post_module.urllib.request, "urlopen", fake_urlopen)

        with pytest.raises(urllib.error.HTTPError) as exc_info:
            inat_post_module._default_create_observation(_item(), "fake-jwt", DECISION)
        assert exc_info.value.code == 500


class TestObservationBody:
    def test_body_carries_every_approved_field_with_inat_api_names(self):
        assert build_observation_body(DECISION) == {"observation": {
            "observed_on_string": "2026-09-01 08:00:00",
            "time_zone": "Pacific Time (US & Canada)",
            "latitude": 38.5, "longitude": -122.25, "geoprivacy": "obscured",
        }}

    def test_optional_taxon_fields_are_included_only_when_present(self):
        body = build_observation_body({**DECISION, "species_guess": "Monarch", "taxon_id": 48662})
        assert body["observation"]["species_guess"] == "Monarch"
        assert body["observation"]["taxon_id"] == 48662

    def test_missing_required_field_raises_instead_of_posting_an_empty_observation(self):
        bad = {k: v for k, v in DECISION.items() if k not in ("lat", "geoprivacy")}
        with pytest.raises(ValueError, match="approved decision is missing: lat, geoprivacy"):
            build_observation_body(bad)

    def test_default_transport_sends_that_exact_body(self, monkeypatch):
        import io
        import json

        import bio_automation.inat_post as inat_post_module

        sent = []

        def fake_urlopen(request, timeout=120):
            sent.append((request.full_url, json.loads(request.data)))
            return io.BytesIO(json.dumps({"results": [{"id": 77, "uuid": "u-77"}]}).encode())

        monkeypatch.setattr(inat_post_module.urllib.request, "urlopen", fake_urlopen)
        attached = []

        row = post_one(_item(photos=("a.jpg",)), jwt="jwt", posted={}, dry_run=False, decision=DECISION,
                       attach_fn=lambda oid, path, jwt: attached.append((oid, path)))

        assert sent == [("https://api.inaturalist.org/v1/observations",
                         build_observation_body(DECISION))]
        assert row["status"] == "posted" and row["observation_id"] == 77
        assert attached == [(77, "a.jpg")]

    def test_post_approved_passes_each_items_own_decision(self, monkeypatch):
        import io
        import json

        import bio_automation.inat_post as inat_post_module

        bodies = []

        def fake_urlopen(request, timeout=120):
            bodies.append(json.loads(request.data))
            return io.BytesIO(json.dumps({"results": [{"id": 5, "uuid": "u"}]}).encode())

        monkeypatch.setattr(inat_post_module.urllib.request, "urlopen", fake_urlopen)
        decisions = {1: DECISION}

        results = post_approved([_item(photos=())], decisions, {}, jwt="jwt", dry_run=False,
                                rate_limiter=RateLimiter(min_interval=0))

        assert [r["status"] for r in results] == ["posted"]
        assert bodies == [build_observation_body(DECISION)]

    def test_default_transport_without_a_decision_fails_the_row_not_an_empty_post(self, monkeypatch):
        import bio_automation.inat_post as inat_post_module

        monkeypatch.setattr(inat_post_module.urllib.request, "urlopen",
                            lambda *a, **k: pytest.fail("must not reach the network"))

        row = post_one(_item(), jwt="jwt", posted={}, dry_run=False)

        assert row["status"] == "failed"
        assert row["error"].startswith("ValueError: approved decision is missing: observed_on_string")
