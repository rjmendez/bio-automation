"""social_gates: the alt-text privacy gate, the image floor and the live recheck before a send, run through the real hook framework with a real
audit log. Stubs: the iNat fetch (a recording fake). Images are real Pillow JPEGs (noise for a sharp photo, a smooth gradient for a blurry one)."""
from __future__ import annotations

import datetime as dt
import socket
import urllib.error
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from bio_automation import social, social_gates, social_hooks
from bio_automation.inat import InatNotFoundError, InatTransientError
from bio_automation.social_gates import Verdict
from bio_automation.social_rules import Audit
from test_social import observation, rules_with


@pytest.fixture
def audit(tmp_path):
    return Audit(tmp_path / "audit.jsonl")


@pytest.fixture
def gates():
    social_gates.register_hooks()


def run_draft(ctx, audit, **rules):
    return social_hooks.run("draft", ctx, rules=rules_with(**rules), audit=audit)


def hidden_obs():
    return observation(place="Berks County, US-PA, US", geoprivacy="private")


# ------------------------------------------------------------------------------------------------------------ alt_text_privacy

class TestAltTextPrivacy:
    def ctx(self, alt, obs=None):
        return {"obs": obs or observation(), "obs_id": 42, "alt": alt}

    @pytest.mark.parametrize("alt,kind", [
        ("A spider at 40.1234, -75.5678 on a leaf.", "coordinates"),
        ("Photographed behind 1234 Old Mill Rd in the evening.", "street address"),
        ("A moth near 7 Oak Street.", "street address"),
    ])
    def test_coordinates_and_addresses_are_rejected_with_their_kind_only(self, alt, kind):
        v = social_gates.alt_text_privacy(self.ctx(alt))
        assert v == Verdict("reject", f"alt text leaks {kind}")

    @pytest.mark.parametrize("alt", [
        "A green orbweaver spider in a web, about 2.5, 3.1 cm across.",
        "The spider is 12 mm long on a long drive of grass.",
        "Three ways to see it, two courts away from nothing.",
        "A lattice orbweaver on a leaf.",
    ])
    def test_ordinary_numbers_and_words_pass(self, alt):
        assert social_gates.alt_text_privacy(self.ctx(alt)) is None

    def test_a_hidden_places_name_is_rejected(self):
        v = social_gates.alt_text_privacy(self.ctx("A spider found in Berks woodland.", hidden_obs()))
        assert v == Verdict("reject", "alt text leaks place finer than the county")

    def test_the_same_place_name_is_fine_when_the_location_is_shown(self):
        assert social_gates.alt_text_privacy(self.ctx("A spider found in Berks woodland.", observation(place="Berks County, US-PA, US"))) is None

    def test_an_obscured_taxon_may_name_the_county_and_state_but_nothing_finer(self):
        obs = observation(place="Hawk Mountain, Berks County, PA, US")
        obs["taxon_geoprivacy"] = "obscured"
        assert social_gates.alt_text_privacy(self.ctx("Seen at Hawk Mountain.", obs)) == Verdict("reject", "alt text leaks place finer than the county")
        assert social_gates.alt_text_privacy(self.ctx("Seen in Berks County, Pennsylvania.", obs)) is None

    def test_a_state_level_place_leaks_nothing_because_nothing_finer_than_the_county_is_in_it(self):
        obs = observation(place="Pennsylvania, US", geoprivacy="obscured")
        assert social_gates.forbidden_tokens(obs) == []
        assert social_gates.alt_text_privacy(self.ctx("Seen in Pennsylvania.", obs)) is None

    def test_a_lowercase_word_that_is_also_a_place_token_is_not_a_leak(self):
        obs = observation(place="Black Rock Road, Berks County, PA", geoprivacy="private")
        assert social_gates.alt_text_privacy(self.ctx("A black spider on a rock.", obs)) is None
        assert social_gates.alt_text_privacy(self.ctx("A spider on Black Rock.", obs)) == Verdict("reject", "alt text leaks place finer than the county")

    def test_generic_place_words_and_short_tokens_are_not_leaks(self):
        obs = observation(place="Ono County, PA, USA", geoprivacy="private")
        assert social_gates.place_tokens(obs["place_guess"]) == []
        assert social_gates.alt_text_privacy(self.ctx("A spider in the County Park, USA.", obs)) is None

    def test_place_tokens_split_on_commas_and_spaces(self):
        assert social_gates.place_tokens("Lake Nockamixon, Bucks County, US-PA, US") == ["Lake", "Nockamixon", "Bucks"]
        assert social_gates.place_tokens(None) == []

    def test_several_leaks_are_all_named(self):
        v = social_gates.alt_text_privacy(self.ctx("Berks spider at 40.1234, -75.5678 behind 9 Elm Street.", hidden_obs()))
        assert v.reason == "alt text leaks coordinates; street address; place finer than the county"

    def test_through_the_framework_the_rule_switches_the_gate_and_the_reject_is_audited(self, gates, audit):
        ctx = {"obs": hidden_obs(), "obs_id": 42, "alt": "In Berks woods.", "image": Path("unused")}
        social_hooks._hooks[:] = [h for h in social_hooks._hooks if h.name == "alt_text_privacy"]
        off = run_draft(ctx, audit, alt_text_privacy=False)
        assert [(h.name, v.action) for h, v in off] == []
        on = run_draft(ctx, audit, alt_text_privacy=True)
        assert [(h.name, v) for h, v in on] == [("alt_text_privacy", Verdict("reject", "alt text leaks place finer than the county"))]
        rec = [r for r in audit.records() if r["event"] == "hook"]
        assert [(r["name"], r["stage"], r["action"], r["reason"], r["obs_id"]) for r in rec] == [
            ("alt_text_privacy", "draft", "reject", "alt text leaks place finer than the county", 42)]


# ----------------------------------------------------------------------------------------------------------------- image_floor

def photo(path: Path, mean=110, size=(640, 480), seed=0, sigma=40) -> Path:
    """A sharp photo-like JPEG: seeded noise around `mean` (lots of edges everywhere)."""
    rng = np.random.RandomState(seed)
    arr = np.clip(rng.normal(mean, sigma, (size[1], size[0], 3)), 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path, "JPEG", quality=95)
    return path


def gradient(path: Path, size=(640, 480), lo=60, hi=190) -> Path:
    row = np.linspace(lo, hi, size[0]).astype(np.uint8)
    arr = np.tile(row[None, :, None], (size[1], 1, 3))
    Image.fromarray(arr).save(path, "JPEG", quality=95)
    return path


class TestImageFloor:
    def verdict(self, path):
        return social_gates.image_floor({"image": path})

    def test_a_sharp_normally_exposed_photo_passes(self, tmp_path):
        assert self.verdict(photo(tmp_path / "a.jpg")) is None

    def test_a_black_frame_is_rejected(self, tmp_path):
        Image.new("RGB", (640, 480), (0, 0, 0)).save(tmp_path / "b.jpg", "JPEG")
        v = self.verdict(tmp_path / "b.jpg")
        assert v == Verdict("reject", "image is black (median luma 0.0 < 2.0)")

    def test_under_exposed_but_not_black_is_rejected_as_under_exposed(self, tmp_path):
        v = self.verdict(photo(tmp_path / "u.jpg", mean=7, sigma=3))
        assert v.action == "reject" and v.reason.startswith("image is grossly under-exposed (median luma ")

    def test_blown_out_is_rejected(self, tmp_path):
        v = self.verdict(photo(tmp_path / "o.jpg", mean=252, sigma=3))
        assert v.action == "reject" and v.reason.startswith("image is grossly over-exposed (median luma ")

    def test_a_smooth_gradient_is_rejected_as_blurry_while_the_same_exposure_in_noise_passes(self, tmp_path):
        v = self.verdict(gradient(tmp_path / "g.jpg"))
        assert v.action == "reject" and v.reason.startswith("image is far too blurry (sharpness 0.")
        assert self.verdict(photo(tmp_path / "n.jpg", mean=125)) is None

    def test_a_thumbnail_sized_picture_is_rejected_for_size(self, tmp_path):
        v = self.verdict(photo(tmp_path / "s.jpg", size=(200, 150)))
        assert v == Verdict("reject", "image is too small (200x150, long edge < 256)")

    def test_measurements_are_exact_for_a_flat_picture(self, tmp_path):
        Image.new("RGB", (300, 300), (100, 100, 100)).save(tmp_path / "f.png")
        assert social_gates.measure_image(tmp_path / "f.png") == {"width": 300, "height": 300, "luma": pytest.approx(100.0, abs=0.01), "sharp": 0.0}

    def test_the_floor_values_sit_between_failures_and_the_calibrated_good_shots(self):
        """Calibration on 153 real shots: luma 19.4..177.5 and sharpness 17.4.. (see the module constants); blurred copies measured 2.2-4.1."""
        assert (social_gates.BLACK_LUMA, social_gates.UNDER_LUMA, social_gates.OVER_LUMA, social_gates.BLUR_SHARP) == (2.0, 14.0, 247.0, 8.0)

    def test_through_the_framework_a_black_frame_skips_only_while_the_rule_is_on(self, gates, audit, tmp_path):
        Image.new("RGB", (640, 480), (0, 0, 0)).save(tmp_path / "b.jpg", "JPEG")
        social_hooks._hooks[:] = [h for h in social_hooks._hooks if h.name == "image_floor"]
        ctx = {"obs": observation(), "obs_id": 42, "alt": "x", "image": tmp_path / "b.jpg"}
        assert run_draft(ctx, audit, image_floor=False) == []
        on = run_draft(ctx, audit, image_floor=True)
        assert [(h.name, v.action) for h, v in on] == [("image_floor", "reject")]


# ------------------------------------------------------------------------------------------------------------------ live_recheck

class LiveFetch:
    """Recording fake of the iNat fetch: answers from a dict; an Exception value is raised."""

    def __init__(self, answers):
        self.answers, self.calls = answers, []

    def __call__(self, obs_id):
        self.calls.append(obs_id)
        a = self.answers[obs_id]
        if isinstance(a, Exception):
            raise a
        return a


def single_draft(obs=None):
    obs = obs or observation()
    return {"draft_id": "2024-10-obs42", "obs_id": 42, "observation": obs, "text": social.text_segments(obs)}


def themed_draft(obs_by_id):
    return {"draft_id": "2026-10-24-spiders", "obs_id": min(obs_by_id), "observations": {str(k): v for k, v in obs_by_id.items()},
            "text": [["text", "Spider Saturday"]]}


def recheck(draft, answers, audit, **rules):
    fetch = LiveFetch(answers)
    ctx = {"draft_id": draft["draft_id"], "draft": draft, "fetch": fetch, "audit": audit, "now": dt.datetime(2026, 10, 24, 8), "live": True, "manual": False}
    return social_gates.live_recheck(ctx), fetch


class TestLiveRecheck:
    def test_an_unchanged_observation_passes_and_is_fetched_once(self, audit):
        v, fetch = recheck(single_draft(), {42: observation()}, audit)
        assert v is None and fetch.calls == [42]

    def test_a_deleted_observation_withdraws(self, audit):
        v, _ = recheck(single_draft(), {42: InatNotFoundError("gone")}, audit)
        assert v == Verdict("withdraw", "observation 42 no longer exists on iNat")

    def test_a_now_casual_observation_withdraws(self, audit):
        v, _ = recheck(single_draft(), {42: observation(grade="casual")}, audit)
        assert v == Verdict("withdraw", "observation 42 is now casual")

    def test_a_taxon_less_observation_withdraws(self, audit):
        now = observation()
        now["taxon"] = None
        v, _ = recheck(single_draft(), {42: now}, audit)
        assert v == Verdict("withdraw", "observation 42 now has no taxon")

    def test_a_different_taxon_name_withdraws(self, audit):
        v, _ = recheck(single_draft(), {42: observation(name="Araneus diadematus")}, audit)
        assert v == Verdict("withdraw", "observation 42 was re-identified: Araneus thaddeus -> Araneus diadematus")

    def test_a_location_that_became_hidden_withdraws_when_the_text_shows_the_place(self, audit):
        v, _ = recheck(single_draft(), {42: observation(geoprivacy="private")}, audit)
        assert v == Verdict("withdraw", "observation 42 now hides its location but the post still shows its place")

    def test_a_hidden_location_passes_when_the_draft_never_showed_a_place(self, audit):
        then = observation(place="")
        v, _ = recheck(single_draft(then), {42: observation(place="", geoprivacy="private")}, audit)
        assert v is None

    def test_an_unexpected_error_is_not_swallowed_so_the_framework_can_block(self, audit):
        with pytest.raises(RuntimeError, match="boom"):
            recheck(single_draft(), {42: RuntimeError("boom")}, audit)
        assert audit.records() == []

    def test_through_the_framework_an_unexpected_error_blocks_the_send(self, gates, audit):
        social_hooks._hooks[:] = [h for h in social_hooks._hooks if h.name == "live_recheck"]
        ctx = {"draft_id": "2024-10-obs42", "draft": single_draft(), "fetch": LiveFetch({42: ValueError("bad json")}), "now": None, "live": True,
               "manual": False}
        on = social_hooks.run("pre_send", ctx, rules=rules_with(live_recheck=True), audit=audit)
        assert [(h.name, v.action) for h, v in on] == [("live_recheck", "block")]

    @pytest.mark.parametrize("error", [urllib.error.URLError("down"), ConnectionResetError("reset"), socket.timeout("slow"),
                                       socket.gaierror("no dns")])
    def test_network_and_socket_errors_fail_open(self, audit, error):
        v, fetch = recheck(single_draft(), {42: error}, audit)
        assert v is None and fetch.calls == [42]
        assert [r["event"] for r in audit.records()] == ["live_recheck_unavailable"]

    @pytest.mark.parametrize("now_kwargs,then_kwargs,expected", [
        ({"geoprivacy": "obscured", "place": "Pennsylvania, US"}, {}, "observation 42 now hides its location but the post still shows its place"),
        ({"geoprivacy": "obscured-private", "place": ""}, {}, "observation 42 now hides its location but the post still shows its place"),
        ({"place": "Lehigh County, PA, USA"}, {}, "observation 42 has a different place on iNat than the one the post shows"),
        ({"geoprivacy": "obscured"}, {"geoprivacy": "obscured"}, None),         # already obscured when drafted: unchanged
        ({}, {}, None),
    ])
    def test_a_newly_obscured_or_moved_place_withdraws_but_an_unchanged_one_passes(self, audit, now_kwargs, then_kwargs, expected):
        v, _ = recheck(single_draft(observation(**then_kwargs)), {42: observation(**now_kwargs)}, audit)
        assert (v.reason if v else None) == expected

    @pytest.mark.parametrize("level", ["obscured", "private"])
    def test_a_taxon_geoprivacy_that_appeared_since_the_draft_withdraws(self, audit, level):
        now = {**observation(), "taxon_geoprivacy": level}
        v, _ = recheck(single_draft(), {42: now}, audit)
        assert v == Verdict("withdraw", "observation 42 now hides its location but the post still shows its place")

    def test_a_taxon_geoprivacy_recorded_unchanged_at_draft_time_passes(self, audit):
        both = {**observation(), "taxon_geoprivacy": "obscured"}
        v, _ = recheck(single_draft(both), {42: dict(both)}, audit)
        assert v is None

    def test_a_transient_error_fails_open_and_is_audited(self, audit):
        v, fetch = recheck(single_draft(), {42: InatTransientError("503")}, audit)
        assert v is None and fetch.calls == [42]
        assert [(r["event"], r["draft_id"], r["obs_id"], r["error"]) for r in audit.records()] == [
            ("live_recheck_unavailable", "2024-10-obs42", 42, "InatTransientError: 503")]

    def test_a_network_error_also_fails_open(self, audit):
        v, _ = recheck(single_draft(), {42: TimeoutError("timed out")}, audit)
        assert v is None
        assert [r["error"] for r in audit.records()] == ["TimeoutError: timed out"]

    def test_a_themed_draft_checks_every_observation_and_names_the_failing_one(self, audit):
        draft = themed_draft({10: observation(10), 11: observation(11), 12: observation(12)})
        v, fetch = recheck(draft, {10: observation(10), 11: observation(11, grade="casual"), 12: observation(12)}, audit)
        assert v == Verdict("withdraw", "observation 11 is now casual") and fetch.calls == [10, 11]

    def test_a_themed_draft_with_every_observation_fine_fetches_each_once(self, audit):
        draft = themed_draft({10: observation(10), 11: observation(11)})
        v, fetch = recheck(draft, {10: observation(10), 11: observation(11)}, audit)
        assert v is None and fetch.calls == [10, 11]

    def test_after_the_first_unavailable_answer_the_rest_of_the_draft_is_not_asked_but_recorded(self, audit):
        draft = themed_draft({10: observation(10), 11: observation(11)})
        v, fetch = recheck(draft, {10: InatTransientError("503"), 11: observation(11, grade="casual")}, audit)
        assert v is None and fetch.calls == [10]
        assert [(r["obs_id"], r["error"], r.get("skipped")) for r in audit.records()] == [(10, "InatTransientError: 503", None),
                                                                                           (11, "InatTransientError: 503", True)]

    def test_positive_twin_with_iNat_up_every_observation_is_asked(self, audit):
        draft = themed_draft({10: observation(10), 11: observation(11)})
        v, fetch = recheck(draft, {10: observation(10), 11: observation(11, grade="casual")}, audit)
        assert v == Verdict("withdraw", "observation 11 is now casual") and fetch.calls == [10, 11]

    def test_three_unavailable_observations_in_a_run_block_the_send(self, audit):
        draft = themed_draft({10: observation(10), 11: observation(11), 12: observation(12)})
        v, fetch = recheck(draft, {10: InatTransientError("503")}, audit)
        assert v == Verdict("block", "iNat could not be asked about 3 observations in this run (limit 3); the live recheck cannot vouch for the draft, nothing is sent")
        assert fetch.calls == [10] and len(audit.records()) == 3

    def test_two_unavailable_observations_still_fail_open(self, audit):
        draft = themed_draft({10: observation(10), 11: observation(11)})
        v, _ = recheck(draft, {10: InatTransientError("503")}, audit)
        assert v is None and len(audit.records()) == 2

    def test_the_unavailable_count_is_per_run(self, tmp_path):
        first = Audit(tmp_path / "audit.jsonl", run="r1")
        recheck(themed_draft({10: observation(10), 11: observation(11)}), {10: InatTransientError("503")}, first)
        second = Audit(tmp_path / "audit.jsonl", run="r2")
        v, _ = recheck(single_draft(), {42: InatTransientError("503")}, second)
        assert v is None                                              # 2 earlier unavailable records belong to another run: this run has 1

    def test_the_runs_audit_from_the_context_receives_the_record(self, tmp_path):
        mine = Audit(tmp_path / "mine.jsonl", run="run-xyz")
        recheck(single_draft(), {42: InatTransientError("503")}, mine)
        assert [r["run"] for r in mine.records()] == ["run-xyz"]

    def alt_draft(self, alt):
        then = observation(place="Hickory Run State Park, PA, USA")
        draft = single_draft(then)
        draft["text"] = [["text", "A lattice orbweaver"]]                 # the post text shows no place: only the alt can leak it
        draft["images"] = [{"file": "1.jpg", "alt": alt}]
        return draft

    def test_an_alt_naming_the_place_is_withdrawn_when_the_observation_turned_private_since(self, audit):
        draft = self.alt_draft("A lattice orbweaver in the Hickory woods.")
        v, _ = recheck(draft, {42: observation(place="", geoprivacy="private")}, audit)
        assert v == Verdict("withdraw", "alt text of image 1 leaks place finer than the county now that observation 42 hides its location")

    def test_the_same_alt_is_withdrawn_while_the_observation_is_still_public_because_the_place_is_finer_than_the_county(self, audit):
        v, _ = recheck(self.alt_draft("A lattice orbweaver in the Hickory woods."), {42: observation(place="Hickory Run State Park, PA, USA")}, audit)
        assert v == Verdict("withdraw", "alt text of image 1 leaks place finer than the county")        # no "now hides its location": it does not

    def test_an_alt_naming_only_the_county_passes_while_the_observation_is_still_public(self, audit):
        then = observation(place="Berks County, PA, USA")
        draft = single_draft(then)
        draft["text"] = [["text", "A lattice orbweaver"]]
        draft["images"] = [{"file": "1.jpg", "alt": "A lattice orbweaver in Berks County woods."}]
        v, _ = recheck(draft, {42: observation(place="Berks County, PA, USA")}, audit)
        assert v is None

    def test_an_alt_without_the_place_passes_when_the_observation_turned_private(self, audit):
        v, _ = recheck(self.alt_draft("A lattice orbweaver in the woods."), {42: observation(place="", geoprivacy="private")}, audit)
        assert v is None

    def test_a_themed_drafts_alt_is_checked_against_its_own_observation(self, audit):
        then10, then11 = observation(10, place="Hickory Run State Park, PA, USA"), observation(11)
        draft = themed_draft({10: then10, 11: then11})
        draft["images"] = [{"file": "1.jpg", "obs_id": 11, "alt": "A spider by the Hickory trail."}, {"file": "2.jpg", "obs_id": 10, "alt": "A spider on a leaf."}]
        v, _ = recheck(draft, {10: observation(10, place="", geoprivacy="private"), 11: observation(11)}, audit)
        assert v is None                                              # the Hickory alt belongs to observation 11, which is still public and never had that place
        draft["images"][1]["alt"] = "A spider on a Hickory leaf."
        v, _ = recheck(draft, {10: observation(10, place="", geoprivacy="private"), 11: observation(11)}, audit)
        assert v == Verdict("withdraw", "alt text of image 2 leaks place finer than the county now that observation 10 hides its location")

    def test_through_the_framework_the_withdraw_is_audited_and_the_rule_can_switch_it_off(self, gates, audit):
        social_hooks._hooks[:] = [h for h in social_hooks._hooks if h.name == "live_recheck"]
        fetch = LiveFetch({42: observation(grade="casual")})
        ctx = {"draft_id": "2024-10-obs42", "draft": single_draft(), "fetch": fetch, "now": None, "live": True, "manual": False}
        assert social_hooks.run("pre_send", ctx, rules=rules_with(live_recheck=False), audit=audit) == [] and fetch.calls == []
        on = social_hooks.run("pre_send", ctx, rules=rules_with(live_recheck=True), audit=audit)
        assert [(h.name, v) for h, v in on] == [("live_recheck", Verdict("withdraw", "observation 42 is now casual"))]
        assert [(r["event"], r["action"], r["draft_id"]) for r in audit.records()] == [("hook", "withdraw", "2024-10-obs42")]


# --------------------------------------------------------------------------------------------------------------- registration

def test_register_hooks_registers_the_gates_with_their_rules_and_priorities():
    social_gates.register_hooks()
    got = sorted((h.stage, h.priority, h.name, h.rule) for stage in ("draft", "pre_send") for h in social_hooks.hooks(stage))
    assert got == [("draft", 10, "alt_text_privacy", "alt_text_privacy"), ("draft", 30, "image_floor", "image_floor"),
                   ("pre_send", 20, "live_recheck", "live_recheck"), ("pre_send", 25, "gate_rescreen_image", "image_floor"),
                   ("pre_send", 25, "gate_rescreen_privacy", "alt_text_privacy")]


def test_every_gate_is_registered_critical_where_the_framework_supports_it(monkeypatch):
    calls = []

    def recording_register(name, stage, fn, *, rule=None, priority=50, critical=False):
        calls.append((name, stage, rule, priority, critical))

    monkeypatch.setattr(social_hooks, "register", recording_register)
    social_gates.register_hooks()
    assert calls == [("alt_text_privacy", "draft", "alt_text_privacy", 10, True), ("image_floor", "draft", "image_floor", 30, True),
                     ("live_recheck", "pre_send", "live_recheck", 20, True), ("gate_rescreen_privacy", "pre_send", "alt_text_privacy", 25, True),
                     ("gate_rescreen_image", "pre_send", "image_floor", 25, True)]


# ------------------------------------------------------------------------------------------------------ privacy gate, hardened

class TestCoordinateForms:
    @pytest.mark.parametrize("alt", [
        "Brambles photographed at 40.4123, -75.9876.",                     # sentence-final pair
        "Brambles at 40.4123, -75.9876, near a log.",
        "Brambles at 40.41234 -75.98765 on a log.",                        # no comma
        "Brambles at +40.4123;-75.9876 on a log.",                         # plus sign, semicolon
        "Brambles at 40.4123,-75.9876",                                    # no space, end of text
        "Brambles at 40.4123, −75.9876 on a log.",                    # Unicode minus
        "Brambles at 40,4123 -75,9876 on a log.",                          # comma decimals
        "Brambles at 40.4123 N 75.9876 W on a log.",                       # hemisphere suffixes
        "Brambles at 40.4123° N, 75.9876° W.",
        "Brambles at N 40.4123 W 75.9876.",
        "Brambles at 40°24'44\"N 75°59'15\"W.",                  # degrees, minutes, seconds
        "Brambles at 40° 24.7′ N.",
        "Brambles, lat 40.4 on a log.",                                    # labels
        "Brambles, Longitude: -75.9.",
        "Brambles at lng=-75.98.",
        "Brambles at 40.4123 N.",
    ])
    def test_coordinates_in_every_form_are_found(self, alt):
        assert social_gates.has_coordinates(alt) is True
        assert social_gates.alt_text_privacy({"obs": observation(), "alt": alt}) == Verdict("reject", "alt text leaks coordinates")

    @pytest.mark.parametrize("alt", [
        "A moth, 4000 px across, photographed in 2025.",
        "Shot at f/2.8 and 1/250 s, ISO 1600.",
        "The wing is 10.5 mm long, 2.5, 3.1 cm in the frame.",
        "About 12.5 cm tall, 3.75 cm wide.",
        "Photographed in October 2025, 40 mm lens.",
        "A bee in a long meadow, a latch on a gate, a longhorn beetle.",
        "Seen on the north side, facing east at 5 S 4 mm.",
    ])
    def test_ordinary_numbers_and_words_are_not_coordinates(self, alt):
        assert social_gates.has_coordinates(alt) is False
        assert social_gates.alt_text_privacy({"obs": observation(), "alt": alt}) is None


def stock_label(place="Hawk Mountain Sanctuary"):
    return {"title": f"Lattice Orbweaver (Araneus thaddeus) — {place}",
            "caption": f"Lattice Orbweaver (Araneus thaddeus) photographed at {place} on 2024-10-01.",
            "keywords": ["spider", "araneus", "sgl220", "pennsylvania"]}


class TestLabelPlace:
    def ctx(self, alt, obs, label=None):
        return {"obs": obs, "obs_id": 42, "alt": alt, "label": label}

    def test_label_places_reads_the_caption_and_the_title_context(self):
        assert social_gates.label_places(stock_label()) == ["Hawk Mountain Sanctuary", "Hawk Mountain Sanctuary"]
        assert social_gates.label_places({"title": "Spider — Autumn", "caption": "Spider photographed on 2024-10-01."}) == []
        assert social_gates.label_places(None) == []

    def test_a_private_observation_cannot_name_the_place_the_label_carries(self):
        obs = observation(place=None, geoprivacy="private")
        v = social_gates.alt_text_privacy(self.ctx("A spider near the Hawk Mountain Sanctuary visitor lot.", obs, stock_label()))
        assert v == Verdict("reject", "alt text leaks place finer than the county")

    def test_the_label_place_is_a_leak_for_a_public_observation_too_but_the_county_is_not(self):
        hidden = observation(place=None, geoprivacy="private")
        assert social_gates.alt_text_privacy(self.ctx("A spider on a leaf near a parking lot.", hidden, stock_label())) is None
        shown = observation()
        assert social_gates.alt_text_privacy(self.ctx("A spider near the Hawk Mountain Sanctuary visitor lot.", shown, stock_label())) == Verdict(
            "reject", "alt text leaks place finer than the county")
        assert social_gates.alt_text_privacy(self.ctx("A spider in Bucks County woods.", shown, stock_label())) is None

    def test_an_obscured_observation_may_name_its_state_but_not_the_labels_finer_places(self):
        obs = observation(place="Pennsylvania, US", geoprivacy="obscured")
        assert social_gates.alt_text_privacy(self.ctx("Brambles at Hawk Mountain Sanctuary in Kempton.", obs, stock_label())).reason == (
            "alt text leaks place finer than the county")
        assert social_gates.alt_text_privacy(self.ctx("A spider somewhere in Pennsylvania.", obs, stock_label())) is None
        assert social_gates.alt_text_privacy(self.ctx("A spider on a leaf.", obs, stock_label())) is None

    def test_a_token_with_digits_from_the_caption_is_found(self):
        obs = observation(place=None, geoprivacy="private")
        label = stock_label("SGL220 , PA, USA")
        assert social_gates.forbidden_tokens(obs, label) == ["SGL220"]
        assert social_gates.alt_text_privacy(self.ctx("A spider by the sgl220 gate.", obs, label)).reason == "alt text leaks place finer than the county"

    @pytest.mark.parametrize("alt", ["Seen in KEMPTON.", "seen in kempton.", "Seen in Kempton!"])
    def test_place_names_match_in_any_case(self, alt):
        obs = observation(place="Kempton, PA, USA", geoprivacy="private")
        assert social_gates.alt_leaks(alt, obs) == ["place finer than the county"]

    @pytest.mark.parametrize("alt", ["Near Río Grande.", "Near RIO GRANDE.", "Near rio grande."])
    def test_place_names_match_across_accents(self, alt):
        obs = observation(place="Río Grande, TX, USA", geoprivacy="private")
        assert social_gates.alt_leaks(alt, obs) == ["place finer than the county"]

    def test_an_ordinary_word_in_a_place_name_leaks_only_when_capitalised(self):
        obs = observation(place="Black Rock Road, PA", geoprivacy="private")
        assert social_gates.alt_leaks("A black spider on a rock.", obs) == []
        assert social_gates.alt_leaks("A spider on Black Rock.", obs) == ["place finer than the county"]

    def test_label_without_place_strips_the_title_context_caption_and_place_keywords(self):
        obs = observation(place=None, geoprivacy="private")
        stripped = social_gates.label_without_place({**stock_label("SGL220 , PA, USA"), "keywords": ["spider", "sgl220", "araneus"]}, obs)
        assert stripped == {"title": "Lattice Orbweaver (Araneus thaddeus)", "caption": None, "keywords": ["spider", "araneus"]}

    def test_a_season_after_the_dash_is_kept(self):
        obs = observation(place=None, geoprivacy="private")
        label = {"title": "Spider — Autumn", "caption": "Spider photographed on 2024-10-01.", "keywords": ["autumn"]}
        assert social_gates.label_without_place(label, obs) == {"title": "Spider — Autumn", "caption": None, "keywords": ["autumn"]}


class TestMakeDraftHidesLabelPlace:
    def draft(self, tmp_path, obs, **rules):
        from test_social import RecordingFetch, jpeg

        g = tmp_path / "gallery"
        rels = ["2024/2024-10/2024-10-01_DSC_1.jpg"]
        jpeg(g / rels[0], size=(300, 200))
        seen = []

        def describe(image, context, *, model):
            seen.append({k: context[k] for k in ("place", "title", "caption", "keywords")})
            return "A spider on a leaf."

        social.make_draft(42, rels, month="2026-10", gallery_dir=g, drafts_dir=tmp_path / "drafts", model="vis:1",
                          fetch=RecordingFetch({42: obs}), read_label=lambda p: stock_label("SGL220 , PA, USA"), describe_fn=describe,
                          rules=rules_with(**rules), audit=Audit(tmp_path / "a.jsonl"), now=dt.datetime(2026, 10, 21, 9, 0))
        return seen

    def test_a_hidden_location_keeps_the_label_place_from_the_vision_model(self, tmp_path):
        seen = self.draft(tmp_path, observation(place=None, geoprivacy="private"), alt_text_privacy=True)
        assert seen == [{"place": None, "title": "Lattice Orbweaver (Araneus thaddeus)", "caption": None,
                         "keywords": ["spider", "araneus", "pennsylvania"]}]

    def test_an_obscured_location_gives_the_model_no_place_either_and_keeps_only_the_state_word(self, tmp_path):
        seen = self.draft(tmp_path, observation(place="Pennsylvania, US", geoprivacy="obscured"), alt_text_privacy=True)
        assert seen == [{"place": None, "title": "Lattice Orbweaver (Araneus thaddeus)", "caption": None,
                         "keywords": ["spider", "araneus", "pennsylvania"]}]       # "pennsylvania" is the state, allowed; only SGL220 is finer

    def test_a_shown_location_gives_the_model_the_county_but_not_the_labels_finer_place(self, tmp_path):
        seen = self.draft(tmp_path, observation(), alt_text_privacy=True)
        assert seen == [{"place": "Bucks County, PA, USA", "title": "Lattice Orbweaver (Araneus thaddeus)", "caption": None,
                         "keywords": ["spider", "araneus", "pennsylvania"]}]

    def test_with_the_rule_off_the_label_goes_through_unchanged(self, tmp_path):
        seen = self.draft(tmp_path, observation(place=None, geoprivacy="private"), alt_text_privacy=False)
        label = stock_label("SGL220 , PA, USA")
        assert seen == [{"place": None, "title": label["title"], "caption": label["caption"], "keywords": label["keywords"]}]


# ----------------------------------------------------------------------------------------------------------- gate_rescreen

def stored_draft(tmp_path, alts, obs=None, black=()):
    d = tmp_path / "drafts" / "2026-10-obs42"
    d.mkdir(parents=True)
    images = []
    for i, alt in enumerate(alts, 1):
        if i in black:
            Image.new("RGB", (640, 480), (0, 0, 0)).save(d / f"{i}.jpg", "JPEG")
        else:
            photo(d / f"{i}.jpg", seed=i)
        images.append({"file": f"{i}.jpg", "alt": alt})
    obs = obs or observation()
    return d, {"draft_id": "2026-10-obs42", "obs_id": 42, "observation": obs, "images": images, "text": social.text_segments(obs)}


def rescreen(hook, d, draft, audit):
    return hook({"draft_id": draft["draft_id"], "draft": draft, "draft_dir": d, "audit": audit})


class TestGateRescreen:
    def test_every_alt_text_of_the_stored_draft_is_screened_and_the_reason_names_the_kind_only(self, tmp_path, audit):
        obs = observation(place="Kempton, PA, USA", geoprivacy="private")
        d, draft = stored_draft(tmp_path, ["A spider on a leaf.", "A spider in Kempton at 40.4123, -75.9876."], obs)
        v = rescreen(social_gates.rescreen_privacy, d, draft, audit)
        assert v == Verdict("withdraw", "alt text of image 2 leaks coordinates; place finer than the county")
        assert [(r["event"], r["draft_id"], r["gate"], r["reason"]) for r in audit.records()] == [
            ("gate_rescreen", "2026-10-obs42", "alt_text_privacy", "alt text of image 2 leaks coordinates; place finer than the county")]

    def test_a_clean_stored_draft_passes_and_audits_nothing(self, tmp_path, audit):
        obs = observation(place="Kempton, PA, USA", geoprivacy="private")
        d, draft = stored_draft(tmp_path, ["A spider on a leaf.", "A spider on a web."], obs)
        assert rescreen(social_gates.rescreen_privacy, d, draft, audit) is None and audit.records() == []

    def test_a_themed_draft_screens_each_image_against_its_own_observation(self, tmp_path, audit):
        d, draft = stored_draft(tmp_path, ["A spider in Kempton.", "A spider in Kempton."])
        draft["images"][0]["obs_id"], draft["images"][1]["obs_id"] = 10, 11
        draft["observations"] = {"10": observation(10), "11": observation(11, place="Kempton, PA, USA", geoprivacy="private")}
        assert rescreen(social_gates.rescreen_privacy, d, draft, audit) == Verdict("withdraw", "alt text of image 2 leaks place finer than the county")

    def test_every_image_file_is_measured_and_the_black_one_withdraws(self, tmp_path, audit):
        d, draft = stored_draft(tmp_path, ["a", "b"], black=(2,))
        v = rescreen(social_gates.rescreen_image, d, draft, audit)
        assert v == Verdict("withdraw", "image 2 (2.jpg): image is black (median luma 0.0 < 2.0)")
        assert [(r["event"], r["gate"]) for r in audit.records()] == [("gate_rescreen", "image_floor")]

    def test_good_images_pass(self, tmp_path, audit):
        d, draft = stored_draft(tmp_path, ["a", "b"])
        assert rescreen(social_gates.rescreen_image, d, draft, audit) is None and audit.records() == []

    def test_a_missing_image_file_withdraws(self, tmp_path, audit):
        d, draft = stored_draft(tmp_path, ["a"])
        (d / "1.jpg").unlink()
        assert rescreen(social_gates.rescreen_image, d, draft, audit) == Verdict("withdraw", "image 1 (1.jpg): image file is missing or unreadable")

    def test_through_the_framework_the_rules_switch_each_rescreen(self, gates, tmp_path, audit):
        social_hooks._hooks[:] = [h for h in social_hooks._hooks if h.name.startswith("gate_rescreen")]
        obs = observation(place="Kempton, PA, USA", geoprivacy="private")
        d, draft = stored_draft(tmp_path, ["In Kempton."], obs, black=(1,))
        ctx = {"draft_id": draft["draft_id"], "draft": draft, "draft_dir": d, "audit": audit}
        both_off = social_hooks.run("pre_send", ctx, rules=rules_with(alt_text_privacy=False, image_floor=False), audit=audit)
        assert both_off == []
        only_image = social_hooks.run("pre_send", ctx, rules=rules_with(alt_text_privacy=False, image_floor=True), audit=audit)
        assert [(h.name, v.action) for h, v in only_image] == [("gate_rescreen_image", "withdraw")]
        only_alt = social_hooks.run("pre_send", ctx, rules=rules_with(alt_text_privacy=True, image_floor=False), audit=audit)
        assert [(h.name, v.action) for h, v in only_alt] == [("gate_rescreen_privacy", "withdraw")]


def test_the_suite_isolates_the_downloaded_photo_cache(tmp_path):
    from bio_automation import settings

    assert settings.INAT_BEST_PHOTO_DIR == tmp_path / "no-prints" / "inat_best"
