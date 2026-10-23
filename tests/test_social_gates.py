"""social_gates: the alt-text privacy gate, the image floor and the live recheck before a send, run through the real hook framework with a real
audit log. Stubs: the iNat fetch (a recording fake). Images are real Pillow JPEGs (noise for a sharp photo, a smooth gradient for a blurry one)."""
from __future__ import annotations

import datetime as dt
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

    @pytest.mark.parametrize("alt,kind,snippet", [
        ("A spider at 40.1234, -75.5678 on a leaf.", "coordinates", "'40.1234, -75.5678'"),
        ("Photographed behind 1234 Old Mill Rd in the evening.", "street address", "'1234 Old Mill Rd'"),
        ("A moth near 7 Oak Street.", "street address", "'7 Oak Street'"),
    ])
    def test_coordinates_and_addresses_are_rejected_with_their_kind(self, alt, kind, snippet):
        v = social_gates.alt_text_privacy(self.ctx(alt))
        assert v == Verdict("reject", f"alt text leaks {kind} ({snippet})")

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
        assert v == Verdict("reject", "alt text leaks hidden place name (Berks)")

    def test_the_same_place_name_is_fine_when_the_location_is_shown(self):
        assert social_gates.alt_text_privacy(self.ctx("A spider found in Berks woodland.", observation(place="Berks County, US-PA, US"))) is None

    def test_taxon_geoprivacy_hides_the_place_too(self):
        obs = observation(place="Pennsylvania, US")
        obs["taxon_geoprivacy"] = "obscured"
        assert social_gates.alt_text_privacy(self.ctx("Seen in Pennsylvania.", obs)) == Verdict("reject", "alt text leaks hidden place name (Pennsylvania)")

    def test_a_lowercase_word_that_is_also_a_place_token_is_not_a_leak(self):
        obs = observation(place="Black Rock Road, Berks County, PA", geoprivacy="private")
        assert social_gates.alt_text_privacy(self.ctx("A black spider on a rock.", obs)) is None
        assert social_gates.alt_text_privacy(self.ctx("A spider on Black Rock.", obs)) == Verdict("reject", "alt text leaks hidden place name (Black, Rock)")

    def test_generic_place_words_and_short_tokens_are_not_leaks(self):
        obs = observation(place="Ono County, PA, USA", geoprivacy="private")
        assert social_gates.place_tokens(obs["place_guess"]) == []
        assert social_gates.alt_text_privacy(self.ctx("A spider in the County Park, USA.", obs)) is None

    def test_place_tokens_split_on_commas_and_spaces(self):
        assert social_gates.place_tokens("Lake Nockamixon, Bucks County, US-PA, US") == ["Lake", "Nockamixon", "Bucks"]
        assert social_gates.place_tokens(None) == []

    def test_several_leaks_are_all_named(self):
        v = social_gates.alt_text_privacy(self.ctx("Berks spider at 40.1234, -75.5678 behind 9 Elm Street.", hidden_obs()))
        assert v.reason == ("alt text leaks coordinates ('40.1234, -75.5678'); street address ('9 Elm Street'); hidden place name (Berks)")

    def test_through_the_framework_the_rule_switches_the_gate_and_the_reject_is_audited(self, gates, audit):
        ctx = {"obs": hidden_obs(), "obs_id": 42, "alt": "In Berks woods.", "image": Path("unused")}
        social_hooks._hooks[:] = [h for h in social_hooks._hooks if h.name == "alt_text_privacy"]
        off = run_draft(ctx, audit, alt_text_privacy=False)
        assert [(h.name, v.action) for h, v in off] == []
        on = run_draft(ctx, audit, alt_text_privacy=True)
        assert [(h.name, v) for h, v in on] == [("alt_text_privacy", Verdict("reject", "alt text leaks hidden place name (Berks)"))]
        rec = [r for r in audit.records() if r["event"] == "hook"]
        assert [(r["name"], r["stage"], r["action"], r["reason"], r["obs_id"]) for r in rec] == [
            ("alt_text_privacy", "draft", "reject", "alt text leaks hidden place name (Berks)", 42)]


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
        assert v == Verdict("withdraw", "observation 42 now hides its location but the post still shows 'Bucks County, PA, USA'")

    def test_a_hidden_location_passes_when_the_draft_never_showed_a_place(self, audit):
        then = observation(place="")
        v, _ = recheck(single_draft(then), {42: observation(place="", geoprivacy="private")}, audit)
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

    def test_a_themed_draft_continues_past_an_unavailable_fetch_to_check_the_rest(self, audit):
        draft = themed_draft({10: observation(10), 11: observation(11)})
        v, fetch = recheck(draft, {10: InatTransientError("503"), 11: observation(11, grade="casual")}, audit)
        assert v == Verdict("withdraw", "observation 11 is now casual") and fetch.calls == [10, 11]

    def test_through_the_framework_the_withdraw_is_audited_and_the_rule_can_switch_it_off(self, gates, audit):
        social_hooks._hooks[:] = [h for h in social_hooks._hooks if h.name == "live_recheck"]
        fetch = LiveFetch({42: observation(grade="casual")})
        ctx = {"draft_id": "2024-10-obs42", "draft": single_draft(), "fetch": fetch, "now": None, "live": True, "manual": False}
        assert social_hooks.run("pre_send", ctx, rules=rules_with(live_recheck=False), audit=audit) == [] and fetch.calls == []
        on = social_hooks.run("pre_send", ctx, rules=rules_with(live_recheck=True), audit=audit)
        assert [(h.name, v) for h, v in on] == [("live_recheck", Verdict("withdraw", "observation 42 is now casual"))]
        assert [(r["event"], r["action"], r["draft_id"]) for r in audit.records()] == [("hook", "withdraw", "2024-10-obs42")]


# --------------------------------------------------------------------------------------------------------------- registration

def test_register_hooks_registers_the_three_gates_with_their_rules_and_priorities():
    social_gates.register_hooks()
    got = sorted((h.stage, h.priority, h.name, h.rule) for stage in ("draft", "pre_send") for h in social_hooks.hooks(stage))
    assert got == [("draft", 10, "alt_text_privacy", "alt_text_privacy"), ("draft", 30, "image_floor", "image_floor"),
                   ("pre_send", 20, "live_recheck", "live_recheck")]
