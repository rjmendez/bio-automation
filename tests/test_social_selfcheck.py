"""social-selfcheck: every check has an ok twin and a warn or fail twin, built from synthetic files in tmp_path; the exit codes, the --json shape
and the audit record are checked on the command itself. Only the disk-space reading and the two online probes are replaced (recording stubs)."""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import types
from pathlib import Path

import pytest
from test_social import NOW, drafted, rules_with, write_draft

from bio_automation import cli, settings, social, social_hooks, social_selfcheck, sync_marker
from bio_automation.social import Ledger
from bio_automation.social_rules import Audit
from bio_automation.social_selfcheck import FAIL, OK, WARN, GB

NAMES = ["rules", "extensions", "themes", "ledger", "audit", "secret-scan", "bluesky-session", "gallery-index", "qdrant-sync", "taxonomy", "approved-drafts", "pending-quality", "pause", "script-drift",
         "disk-system", "disk-data"]


class World:
    """A complete, healthy posting setup under tmp_path; a test breaks one thing and looks at that check."""

    def __init__(self, tmp_path, monkeypatch):
        self.tmp, self.mp = tmp_path, monkeypatch
        self.clock = NOW
        self.ledger_path = tmp_path / "l.jsonl"
        self.drafts = tmp_path / "drafts"
        self.index = tmp_path / "index.json"
        self.taxonomy = tmp_path / "taxonomy.json"
        self.oauth = tmp_path / "oauth"
        self.audit_path = tmp_path / "audit.jsonl"
        for name, value in (("SOCIAL_DIR", tmp_path), ("SOCIAL_LEDGER", self.ledger_path), ("SOCIAL_AUDIT_LOG", self.audit_path),
                            ("GALLERY_OBS_INDEX", self.index), ("OBS_TAXONOMY_FILE", self.taxonomy), ("BSKY_OAUTH_DIR", self.oauth),
                            ("SOCIAL_PAUSE_FILE", tmp_path / "PAUSED"), ("CACHE_DIR", tmp_path / "cache")):
            monkeypatch.setattr(settings, name, value)
        monkeypatch.setenv("BIO_SOCIAL_RULE_LOWER_QUALITY_PHOTOS", "200")        # the synthetic drafts have 300 px photos
        self.scripts = tmp_path / "deployed-scripts"                 # a healthy deployed folder: byte copies of the repo's scheduled-task scripts
        self.scripts.mkdir()
        for f in social_selfcheck.repo_scripts_dir().glob("*.ps1"):
            (self.scripts / f.name).write_bytes(f.read_bytes())
        monkeypatch.setattr(settings, "SOCIAL_SCRIPTS_DIR", self.scripts)
        self.oauth.mkdir()
        (self.oauth / "account.json").write_text(json.dumps({"did": "did:plc:x", "handle": "me.example"}), encoding="utf-8")
        self.set_index({"a/1.jpg": 1, "a/2.jpg": 2, "a/3.jpg": 3}, age_days=1)
        self.taxonomy.write_text(json.dumps({"1": {}, "2": {}}), encoding="utf-8")
        drafted_ledger, _ = drafted(tmp_path, 2)
        self.ledger = drafted_ledger
        monkeypatch.setattr(settings, "QDRANT_SYNC_MARKER", tmp_path / "qdrant_sync_marker.json")
        for sync in sync_marker.NAMES:
            sync_marker.record(sync, now=NOW)

    def set_index(self, files, age_days):
        self.index.write_text(json.dumps({"files": files, "unresolved": {}}), encoding="utf-8")
        t = self.clock.timestamp() - age_days * 86400
        os.utime(self.index, (t, t))

    def run(self, **kw):
        kw.setdefault("now", NOW)
        kw.setdefault("drives", {"system": Path("C:/sys"), "data": Path("D:/data")})
        kw.setdefault("free", lambda p: 100 * GB)
        return {c.name: c for c in social_selfcheck.run_checks(**kw)}


@pytest.fixture
def world(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


def no_deep_integrity(monkeypatch):
    monkeypatch.setitem(sys.modules, "bio_automation.social_state", None)


# ----------------------------------------------------------------------------------------------------------------------- healthy

class TestHealthy:
    def test_every_check_is_ok_on_a_healthy_setup(self, world):
        checks = world.run()
        assert list(checks) == NAMES
        assert {n: c.level for n, c in checks.items()} == {n: OK for n in NAMES}
        assert social_selfcheck.worst(list(checks.values())) == OK

    def test_worst_is_the_most_severe_level(self):
        c = social_selfcheck.Check
        assert social_selfcheck.worst([c("a", OK, ""), c("b", WARN, ""), c("c", OK, "")]) == WARN
        assert social_selfcheck.worst([c("a", WARN, ""), c("b", FAIL, ""), c("c", OK, "")]) == FAIL
        assert social_selfcheck.worst([]) == OK


# ----------------------------------------------------------------------------------------------------------------------- rules, themes

class TestExtensionsCheck:
    def test_all_modules_and_critical_hooks_present_is_ok(self, world):
        c = world.run()["extensions"]
        assert (c.level, c.message) == (OK, f"{len(social_hooks.EXTENSIONS)} social modules loaded, 5 critical hooks registered")

    def test_a_deleted_extension_module_fails(self, world, monkeypatch):
        monkeypatch.setattr(social_hooks, "EXTENSIONS", (*social_hooks.EXTENSIONS, "social_deleted_gate"))
        c = world.run()["extensions"]
        assert c.level == FAIL
        assert "social_deleted_gate (ModuleNotFoundError: No module named 'bio_automation.social_deleted_gate')" in c.message

    def test_a_critical_hook_that_is_not_registered_fails_and_names_it(self, world, monkeypatch):
        real = social_hooks.load_all
        def load_without_image_floor():
            mods = real()
            social_hooks._hooks[:] = [h for h in social_hooks._hooks if h.name != "image_floor"]
            return mods
        monkeypatch.setattr(social_hooks, "load_all", load_without_image_floor)
        c = world.run()["extensions"]
        assert (c.level, c.message) == (FAIL, "critical hook(s) not registered: image_floor (draft)")


class TestRulesAndThemes:
    def test_a_bad_rule_in_the_environment_fails(self, world, monkeypatch):
        monkeypatch.setenv("BIO_SOCIAL_RULE_DAILY_CAP", "banana")
        c = world.run()["rules"]
        assert c.level == FAIL
        assert "BIO_SOCIAL_RULE_DAILY_CAP" in c.message and "banana" in c.message

    def test_a_bad_rules_file_fails_and_a_good_one_loads(self, world, monkeypatch):
        rules_file = world.tmp / "rules.toml"
        monkeypatch.setattr(settings, "SOCIAL_RULES_FILE", rules_file)
        rules_file.write_text("[rules.nonsense]\nenabled = true\n", encoding="utf-8")
        bad = world.run()["rules"]
        assert bad.level == FAIL and "unknown rule 'nonsense'" in bad.message
        rules_file.write_text("[rules.skip_casual]\nenabled = false\n", encoding="utf-8")
        good = world.run()["rules"]
        assert good.level == OK
        assert good.message == f"rules load ({DEFAULT_ON - 1} on, {len(social_selfcheck_rules()) - DEFAULT_ON + 1} off)"

    def test_a_failing_rules_load_still_lets_the_other_checks_run(self, world, monkeypatch):
        monkeypatch.setenv("BIO_SOCIAL_RULE_DAILY_CAP", "banana")
        assert list(world.run()) == NAMES

    def test_a_bad_themes_file_fails(self, world, monkeypatch):
        themes_file = world.tmp / "themes.toml"
        monkeypatch.setattr(settings, "SOCIAL_THEMES_FILE", themes_file)
        themes_file.write_text("[[theme]]\nkey = 'x'\n", encoding="utf-8")
        c = world.run()["themes"]
        assert c.level == FAIL and "missing name" in c.message

    def test_themes_load_from_the_file_or_the_built_in_defaults(self, world, monkeypatch):
        themes_file = world.tmp / "themes.toml"
        monkeypatch.setattr(settings, "SOCIAL_THEMES_FILE", themes_file)
        assert world.run()["themes"].message.endswith("(built-in defaults)")
        themes_file.write_text("[[theme]]\nkey='m'\nname='Moth'\nweekday='monday'\n[theme.include]\niconic=['Insecta']\n", encoding="utf-8")
        c = world.run()["themes"]
        assert (c.level, c.message) == (OK, "1 themes (file)")


def social_selfcheck_rules():
    from bio_automation.social_rules import RULES

    return RULES


DEFAULT_ON = sum(1 for r in social_selfcheck_rules() if r.default)


# ----------------------------------------------------------------------------------------------------------------------- ledger

class TestLedger:
    def test_a_clean_ledger_is_ok_and_says_deep_integrity_was_skipped(self, world, monkeypatch):
        no_deep_integrity(monkeypatch)
        c = world.run()["ledger"]
        assert c.level == OK
        assert c.message == "ledger parses (2 records); deep integrity check skipped (social_state.check_integrity is not available)"

    def test_a_torn_last_line_fails(self, world, monkeypatch):
        no_deep_integrity(monkeypatch)
        with open(world.ledger_path, "ab") as f:
            f.write(b'{"event": "approved", "draft_id": "2026-10-ob')
        c = world.run()["ledger"]
        assert (c.level, c.message) == (FAIL, "ledger last line is torn (JSONDecodeError)")

    def test_a_valid_last_line_without_a_newline_fails_because_the_next_append_would_glue_to_it(self, world, monkeypatch):
        no_deep_integrity(monkeypatch)
        text = world.ledger_path.read_bytes().rstrip(b"\n")
        world.ledger_path.write_bytes(text)
        c = world.run()["ledger"]
        assert (c.level, c.message) == (FAIL, "ledger last line has no newline (the next record would be glued to it)")

    def test_a_corrupt_middle_line_fails_and_names_the_line(self, world, monkeypatch):
        no_deep_integrity(monkeypatch)
        lines = world.ledger_path.read_bytes().split(b"\n")
        world.ledger_path.write_bytes(b"\n".join([lines[0], b"{broken", *lines[1:]]))
        c = world.run()["ledger"]
        assert (c.level, c.message) == (FAIL, "ledger line 2 is corrupt (JSONDecodeError)")
        got = world.run()
        assert got["approved-drafts"].message == "skipped: the ledger cannot be read (SocialError)"
        assert got["pending-quality"].message == "skipped: the ledger cannot be read (SocialError)"
        assert list(got) == NAMES

    def test_a_line_that_is_json_but_not_a_ledger_record_fails(self, world, monkeypatch):
        no_deep_integrity(monkeypatch)
        world.ledger_path.write_bytes(world.ledger_path.read_bytes() + b"[1, 2]\n")
        assert world.run()["ledger"].message == "ledger line 3 is corrupt (ValueError)"

    def test_a_missing_ledger_is_a_warning(self, world, monkeypatch):
        no_deep_integrity(monkeypatch)
        world.ledger_path.unlink()
        c = world.run()["ledger"]
        assert c.level == WARN and c.message.startswith("no ledger at ")

    def test_the_deep_integrity_check_is_used_when_the_state_module_has_one(self, world, monkeypatch):
        calls = []
        mod = types.ModuleType("bio_automation.social_state")
        mod.check_integrity = lambda path: calls.append(path) or []
        monkeypatch.setitem(sys.modules, "bio_automation.social_state", mod)
        c = world.run()["ledger"]
        assert (c.level, c.message) == (OK, "ledger parses (2 records); integrity check clean")
        assert calls == [world.ledger_path]

    def test_problems_from_the_deep_integrity_check_fail(self, world, monkeypatch):
        mod = types.ModuleType("bio_automation.social_state")
        mod.check_integrity = lambda path: ["draft x posted twice", "gap at record 7"]
        monkeypatch.setitem(sys.modules, "bio_automation.social_state", mod)
        c = world.run()["ledger"]
        assert (c.level, c.message) == (FAIL, "ledger parses (2 records); integrity: draft x posted twice; gap at record 7")

    def test_a_deep_check_that_takes_other_arguments_is_skipped_not_called(self, world, monkeypatch):
        calls = []
        mod = types.ModuleType("bio_automation.social_state")
        mod.check_integrity = lambda a, b: calls.append((a, b)) or ["never"]
        monkeypatch.setitem(sys.modules, "bio_automation.social_state", mod)
        c = world.run()["ledger"]
        assert c.level == OK and "skipped" in c.message
        assert calls == []


# ----------------------------------------------------------------------------------------------------------------------- audit, session, pause

class TestAuditSessionPause:
    def test_the_audit_log_is_writable_or_not(self, world):
        assert world.run()["audit"].level == OK
        blocked = world.tmp / "blocked"
        blocked.mkdir()
        world.mp.setattr(settings, "SOCIAL_AUDIT_LOG", blocked)               # a directory where the file should be
        c = world.run()["audit"]
        assert c.level == FAIL and "is not writable" in c.message

    def test_the_session_files_are_required(self, world):
        assert world.run()["bluesky-session"].message == "session files present for me.example"
        (world.oauth / "account.json").unlink()
        c = world.run()["bluesky-session"]
        assert c.level == FAIL and "account.json is missing" in c.message

    def test_an_unreadable_account_file_fails(self, world):
        (world.oauth / "account.json").write_text("{not json", encoding="utf-8")
        c = world.run()["bluesky-session"]
        assert c.level == FAIL and "unreadable (JSONDecodeError)" in c.message
        (world.oauth / "account.json").write_text(json.dumps({"handle": "me.example"}), encoding="utf-8")        # no did
        assert world.run()["bluesky-session"].level == FAIL

    def test_the_pause_file_state_is_reported_as_information(self, world):
        c = world.run()["pause"]
        assert (c.level, c.message) == (OK, "info: posting is not paused")
        (world.tmp / "PAUSED").write_text("", encoding="utf-8")
        c = world.run()["pause"]
        assert c.level == OK and c.message == f"info: posting is PAUSED ({world.tmp / 'PAUSED'} exists)"


    def test_a_remote_pause_marker_is_reported_and_an_unreachable_folder_or_a_disabled_rule_warns(self, world):
        remote = world.tmp / "synced" / "PAUSE-BIO"
        world.mp.setattr(settings, "SOCIAL_PAUSE_SYNC_FILE", remote)
        c = world.run()["pause"]
        assert (c.level, c.message) == (WARN, f"the synced pause folder {remote.parent} is not reachable, so a remote pause cannot be seen")
        remote.parent.mkdir()
        assert world.run()["pause"].message == "info: posting is not paused"                # twin: reachable folder, no marker
        remote.write_text("", encoding="utf-8")
        c = world.run()["pause"]
        assert (c.level, c.message) == (OK, f"info: posting is PAUSED ({remote} exists)")

    def test_a_pause_marker_with_the_rule_off_warns_that_nothing_is_paused(self, world):
        (world.tmp / "PAUSED").write_text("", encoding="utf-8")
        world.mp.setenv("BIO_SOCIAL_RULE_PAUSE_SWITCH", "off")
        c = world.run()["pause"]
        assert c.level == WARN and c.message == f"a pause marker exists ({world.tmp / 'PAUSED'}) but the rule pause_switch is OFF, so posting is NOT paused"


# ----------------------------------------------------------------------------------------------------------------------- caches

class TestIndexAndTaxonomy:
    def test_a_fresh_index_reports_its_file_count(self, world):
        c = world.run()["gallery-index"]
        assert (c.level, c.message) == (OK, "gallery index is 1.0 days old, 3 files")

    @pytest.mark.parametrize("age,level", [(2.9, OK), (3.0, WARN), (10.0, WARN)])
    def test_an_index_of_three_days_or_more_is_stale(self, world, age, level):
        world.set_index({"a/1.jpg": 1}, age_days=age)
        assert world.run()["gallery-index"].level == level

    def test_a_missing_index_warns_and_an_unreadable_one_fails(self, world):
        world.index.unlink()
        c = world.run()["gallery-index"]
        assert c.level == WARN and "run `bio social-index`" in c.message
        world.index.write_text("[]", encoding="utf-8")
        assert world.run()["gallery-index"].level == FAIL

    def test_the_taxonomy_cache_counts_observations(self, world):
        c = world.run()["taxonomy"]
        assert (c.level, c.message) == (OK, "taxonomy cache has 2 observations")

    def test_a_missing_taxonomy_cache_warns_and_a_broken_one_fails(self, world):
        world.taxonomy.write_text("{broken", encoding="utf-8")
        assert world.run()["taxonomy"].level == FAIL
        world.taxonomy.unlink()
        assert world.run()["taxonomy"].level == WARN


# ----------------------------------------------------------------------------------------------------------------------- drafts

def approve_all(world):
    for did in sorted(world.ledger.drafts()):
        social.approve(world.ledger, world.drafts, did, rules=rules_with(), audit=Audit(world.tmp / "approve-audit.jsonl"))


class TestApprovedDrafts:
    def test_approved_drafts_that_still_match_are_ok(self, world):
        approve_all(world)
        c = world.run()["approved-drafts"]
        assert (c.level, c.message) == (OK, "2 approved drafts match their approval hash and have their images")

    def test_no_approved_drafts_is_ok(self, world):
        assert world.run()["approved-drafts"].message == "0 approved drafts match their approval hash and have their images"

    def test_an_edited_draft_no_longer_matches_its_approval(self, world):
        approve_all(world)
        path = world.drafts / "2026-10-obs1" / "draft.json"
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc["images"][0]["alt"] = "edited after approval"
        path.write_text(json.dumps(doc), encoding="utf-8")
        c = world.run()["approved-drafts"]
        assert (c.level, c.message) == (FAIL, "1 of 2 approved drafts are not sendable: 2026-10-obs1 changed since it was approved")

    def test_a_missing_image_fails(self, world):
        approve_all(world)
        (world.drafts / "2026-10-obs2" / "2.jpg").unlink()
        c = world.run()["approved-drafts"]
        assert c.level == FAIL
        assert c.message.startswith("1 of 2 approved drafts are not sendable: 2026-10-obs2 cannot be read (FileNotFoundError")

    def test_an_unapproved_draft_with_a_missing_image_is_not_this_checks_business(self, world):
        (world.drafts / "2026-10-obs2" / "2.jpg").unlink()
        assert world.run()["approved-drafts"].level == OK


class TestPendingQuality:
    def test_photos_at_the_floor_are_ok_and_one_pixel_under_warns(self, world):
        world.mp.setenv("BIO_SOCIAL_RULE_LOWER_QUALITY_PHOTOS", "300")
        c = world.run()["pending-quality"]
        assert (c.level, c.message) == (OK, "all 2 pending drafts have photos of at least 300 px")
        world.mp.setenv("BIO_SOCIAL_RULE_LOWER_QUALITY_PHOTOS", "301")
        c = world.run()["pending-quality"]
        assert c.level == WARN
        assert c.message == "2 of 2 pending drafts have a photo under 301 px and will be withdrawn: 2026-10-obs1 (300 px), 2026-10-obs2 (300 px)"

    def test_the_default_floor_of_1000_px_flags_the_small_synthetic_drafts(self, world):
        world.mp.delenv("BIO_SOCIAL_RULE_LOWER_QUALITY_PHOTOS")
        assert world.run()["pending-quality"].message.startswith("2 of 2 pending drafts have a photo under 1000 px")

    def test_the_rule_off_skips_the_check(self, world):
        world.mp.setenv("BIO_SOCIAL_RULE_LOWER_QUALITY_PHOTOS", "off")
        c = world.run()["pending-quality"]
        assert (c.level, c.message) == (OK, "lower_quality_photos is off: photo size is not checked")

    def test_a_posted_draft_is_not_pending(self, world):
        world.mp.setenv("BIO_SOCIAL_RULE_LOWER_QUALITY_PHOTOS", "1000")
        write_draft(world.drafts, "2026-10-obs3", 3, "2026-10")
        world.ledger.append("drafted", "2026-10-obs3", obs_id=3, month="2026-10", now=NOW)
        world.ledger.append("posted", "2026-10-obs3", obs_id=3, month="2026-10", uri="at://x", now=NOW)
        assert world.run()["pending-quality"].message.startswith("2 of 2 pending drafts")


# ----------------------------------------------------------------------------------------------------------------------- disks

class TestDisks:
    @pytest.mark.parametrize("gb,level", [(14.9, FAIL), (15.0, WARN), (19.9, WARN), (20.0, OK)])
    def test_system_drive_thresholds(self, world, gb, level):
        got = world.run(free=lambda p: int(gb * GB) if p == Path("C:/sys") else 100 * GB)
        assert got["disk-system"].level == level
        assert got["disk-data"].level == OK

    @pytest.mark.parametrize("gb,level", [(1.9, FAIL), (2.0, WARN), (9.9, WARN), (10.0, OK)])
    def test_data_drive_thresholds(self, world, gb, level):
        got = world.run(free=lambda p: int(gb * GB) if p == Path("D:/data") else 100 * GB)
        assert got["disk-data"].level == level
        assert got["disk-system"].level == OK

    def test_the_message_names_the_drive_and_the_space(self, world):
        c = world.run(free=lambda p: 12 * GB)["disk-system"]
        assert c.message == f"{Path('C:/sys')} has 12.0 GB free (below 15)"

    def test_the_real_disk_reading_finds_a_folder_that_does_not_exist_yet(self, tmp_path):
        assert social_selfcheck._free_bytes(tmp_path / "not" / "there") > 0


# ----------------------------------------------------------------------------------------------------------------------- online

class Recorder:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.calls = result, error, 0

    def __call__(self):
        self.calls += 1
        if self.error:
            raise self.error
        return self.result


class TestOnline:
    def test_nothing_online_is_called_without_the_flag(self, world):
        who, inat = Recorder({"handle": "me.example"}), Recorder(200)
        got = world.run(whoami=who, inat_probe=inat)
        assert "bluesky-online" not in got and "inat-online" not in got
        assert (who.calls, inat.calls) == (0, 0)

    def test_working_probes_are_ok(self, world):
        who, inat = Recorder({"handle": "me.example", "expiresAt": "2026-11-01T00:00:00Z"}), Recorder(200)
        got = world.run(online=True, whoami=who, inat_probe=inat)
        assert (who.calls, inat.calls) == (1, 1)
        assert (got["bluesky-online"].level, got["bluesky-online"].message) == (OK, "Bluesky session works for me.example (expires 2026-11-01T00:00:00Z)")
        assert (got["inat-online"].level, got["inat-online"].message) == (OK, "iNat API reachable (HTTP 200)")

    def test_a_refused_session_fails(self, world):
        got = world.run(online=True, whoami=Recorder(error=PermissionError("refresh refused")), inat_probe=Recorder(200))
        c = got["bluesky-online"]
        assert (c.level, c.message) == (FAIL, "Bluesky session probe failed: PermissionError: refresh refused")

    def test_a_missing_client_library_fails(self, world):
        got = world.run(online=True, whoami=Recorder(error=ImportError("No module named bsky_oauth")), inat_probe=Recorder(200))
        assert got["bluesky-online"].level == FAIL and "not installed" in got["bluesky-online"].message

    def test_inat_trouble_is_a_warning_not_a_failure(self, world):
        who = Recorder({"handle": "me.example"})
        down = world.run(online=True, whoami=who, inat_probe=Recorder(error=TimeoutError("timed out")))["inat-online"]
        assert (down.level, down.message) == (WARN, "iNat is not reachable: TimeoutError: timed out")
        busy = world.run(online=True, whoami=who, inat_probe=Recorder(503))["inat-online"]
        assert (busy.level, busy.message) == (WARN, "iNat answered HTTP 503")


# ----------------------------------------------------------------------------------------------------------------------- the command

@pytest.fixture
def plenty_of_disk(world, monkeypatch):
    """The command reads the real clock and the real disks: give it a day-old index relative to now and 100 GB free everywhere."""
    monkeypatch.setattr(social_selfcheck, "_free_bytes", lambda p: 100 * GB)
    world.clock = dt.datetime.now(settings.LOCAL_TZ)
    world.set_index({"a/1.jpg": 1, "a/2.jpg": 2, "a/3.jpg": 3}, age_days=1)


def command(monkeypatch, capsys, *argv):
    monkeypatch.setattr(sys, "argv", ["bio", "social-selfcheck", *argv])
    code = cli.main()
    return code, capsys.readouterr()


class TestCommand:
    def test_exit_0_when_everything_is_ok(self, world, plenty_of_disk, monkeypatch, capsys):
        code, out = command(monkeypatch, capsys)
        assert code == 0
        lines = out.out.splitlines()
        assert lines[-1] == "selfcheck: ok"
        assert lines[0] == f"[ok  ] rules: rules load ({DEFAULT_ON} on, {len(social_selfcheck_rules()) - DEFAULT_ON} off)"
        assert len(lines) == len(NAMES) + 1

    def test_exit_1_on_a_warning(self, world, plenty_of_disk, monkeypatch, capsys):
        world.set_index({"a/1.jpg": 1}, age_days=30)
        code, out = command(monkeypatch, capsys)
        assert code == 1
        assert out.out.splitlines()[-1] == "selfcheck: warn"

    def test_exit_2_on_a_failure_even_with_warnings_around(self, world, plenty_of_disk, monkeypatch, capsys):
        world.set_index({"a/1.jpg": 1}, age_days=30)
        (world.oauth / "account.json").unlink()
        code, out = command(monkeypatch, capsys)
        assert code == 2
        assert out.out.splitlines()[-1] == "selfcheck: fail"

    def test_json_output_has_the_documented_shape(self, world, plenty_of_disk, monkeypatch, capsys):
        world.set_index({"a/1.jpg": 1}, age_days=30)
        code, out = command(monkeypatch, capsys, "--json")
        doc = json.loads(out.out)
        assert code == 1
        assert list(doc) == ["level", "exit", "online", "checks"]
        assert (doc["level"], doc["exit"], doc["online"]) == ("warn", 1, False)
        assert [c["name"] for c in doc["checks"]] == NAMES
        assert all(list(c) == ["name", "level", "message"] for c in doc["checks"])
        assert {c["name"]: c["level"] for c in doc["checks"] if c["level"] != OK} == {"gallery-index": WARN}

    def test_the_run_is_audited_with_its_level_and_problems(self, world, plenty_of_disk, monkeypatch, capsys):
        world.set_index({"a/1.jpg": 1}, age_days=30)
        command(monkeypatch, capsys)
        recs = [r for r in Audit(world.audit_path).records() if r["event"] == "selfcheck"]
        assert len(recs) == 1
        assert recs[0]["level"] == "warn" and recs[0]["online"] is False
        assert recs[0]["problems"] == ["gallery-index: gallery index is 30.0 days old (limit 3), 1 files"]

    def test_a_clean_run_audits_no_problems(self, world, plenty_of_disk, monkeypatch, capsys):
        command(monkeypatch, capsys)
        rec = [r for r in Audit(world.audit_path).records() if r["event"] == "selfcheck"][0]
        assert (rec["level"], rec["problems"]) == ("ok", [])

    def test_online_calls_the_probes_through_the_command(self, world, plenty_of_disk, monkeypatch, capsys):
        who, inat = Recorder({"handle": "me.example"}), Recorder(200)
        monkeypatch.setattr(social_selfcheck, "default_whoami", who)
        monkeypatch.setattr(social_selfcheck, "default_inat_probe", inat)
        code, out = command(monkeypatch, capsys, "--online", "--json")
        doc = json.loads(out.out)
        assert (code, doc["online"], who.calls, inat.calls) == (0, True, 1, 1)
        assert [c["name"] for c in doc["checks"]][-2:] == ["bluesky-online", "inat-online"]


# ----------------------------------------------------------------------------------------------------------------------- crashes and --deploy

class TestCrash:
    def break_a_draft(self, world):
        (world.drafts / "2026-10-obs1" / "draft.json").write_text("[]", encoding="utf-8")       # valid JSON, not a draft: the pending-quality check crashes

    def test_a_crash_is_exit_2_with_a_final_fail_line_and_no_traceback(self, world, plenty_of_disk, monkeypatch, capsys):
        self.break_a_draft(world)
        code, out = command(monkeypatch, capsys)
        assert code == 2
        assert out.out.splitlines()[-1] == "selfcheck: fail (TypeError)"
        assert out.err.startswith("selfcheck crashed: TypeError: list indices must be integers or slices, not str")
        assert "Traceback" not in out.out + out.err

    def test_a_crash_in_deploy_mode_is_a_fail_too(self, world, plenty_of_disk, monkeypatch, capsys):
        self.break_a_draft(world)
        code, out = command(monkeypatch, capsys, "--deploy")
        assert (code, out.out.splitlines()[-1]) == (2, "selfcheck: fail (TypeError)")

    def test_without_a_broken_draft_the_same_command_exits_0(self, world, plenty_of_disk, monkeypatch, capsys):
        code, out = command(monkeypatch, capsys, "--deploy")
        assert (code, out.out.splitlines()[-1]) == (0, "selfcheck: ok")


class TestDeployMode:
    def detail(self, world, **kw):
        return {c.name: c for c in social_selfcheck.run_checks(now=NOW, drives=kw.pop("drives", {"system": Path("C:/sys"), "data": Path("D:/data")}),
                                                               free=kw.pop("free", lambda p: 100 * GB))}

    def test_low_system_disk_is_a_fail_normally_and_a_warning_in_deploy_mode(self, world, monkeypatch):
        low = lambda p: 1 * GB if str(p).startswith("C:") else 100 * GB       # noqa: E731
        normal = social_selfcheck.selfcheck(drives={"system": Path("C:/sys"), "data": Path("D:/data")}, free=low, now=NOW, audit=Audit(world.audit_path))
        deploy = social_selfcheck.selfcheck(drives={"system": Path("C:/sys"), "data": Path("D:/data")}, free=low, now=NOW, deploy=True,
                                            audit=Audit(world.audit_path))
        by = lambda out: {c["name"]: (c["level"], c["message"]) for c in out["checks"]}["disk-system"]           # noqa: E731
        assert by(normal) == (FAIL, "C:\\sys has 1.0 GB free (below 15)")
        assert by(deploy) == (WARN, "C:\\sys has 1.0 GB free (below 15) [environmental: a warning in --deploy mode]")
        assert (normal["exit"], deploy["exit"], deploy["level"]) == (2, 1, "warn")

    def test_a_missing_bluesky_session_is_a_warning_in_deploy_mode(self, world):
        (world.oauth / "account.json").unlink()
        normal = social_selfcheck.selfcheck(drives={"system": Path("C:/sys"), "data": Path("D:/data")}, free=lambda p: 100 * GB, now=NOW,
                                            audit=Audit(world.audit_path))
        deploy = social_selfcheck.selfcheck(drives={"system": Path("C:/sys"), "data": Path("D:/data")}, free=lambda p: 100 * GB, now=NOW, deploy=True,
                                            audit=Audit(world.audit_path))
        levels = lambda out: {c["name"]: c["level"] for c in out["checks"]}["bluesky-session"]                  # noqa: E731
        assert (levels(normal), normal["exit"], levels(deploy), deploy["exit"]) == (FAIL, 2, WARN, 1)

    def test_an_approved_draft_whose_hash_changed_is_a_warning_in_deploy_mode(self, world):
        approve_all(world)
        path = world.drafts / "2026-10-obs1" / "draft.json"
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc["images"][0]["alt"] = "edited after approval"
        path.write_text(json.dumps(doc), encoding="utf-8")
        kw = dict(drives={"system": Path("C:/sys"), "data": Path("D:/data")}, free=lambda p: 100 * GB, now=NOW, audit=Audit(world.audit_path))
        normal, deploy = social_selfcheck.selfcheck(**kw), social_selfcheck.selfcheck(deploy=True, **kw)
        level = lambda out: {c["name"]: c["level"] for c in out["checks"]}["approved-drafts"]                   # noqa: E731
        assert (level(normal), level(deploy), deploy["exit"]) == (FAIL, WARN, 1)

    KW = dict(drives={"system": Path("C:/sys"), "data": Path("D:/data")}, free=lambda p: 100 * GB, now=NOW)

    def both(self, world):
        """((normal levels, exit), (deploy levels, exit)) on the world as the test left it."""
        normal = social_selfcheck.selfcheck(audit=Audit(world.tmp / "sc-audit.jsonl"), **self.KW)
        deploy = social_selfcheck.selfcheck(deploy=True, audit=Audit(world.tmp / "sc-audit.jsonl"), **self.KW)
        return ({c["name"]: c["level"] for c in normal["checks"]}, normal["exit"]), ({c["name"]: c["level"] for c in deploy["checks"]}, deploy["exit"])

    def test_a_corrupt_ledger_is_a_fail_normally_and_a_warning_in_deploy_mode(self, world):
        lines = world.ledger_path.read_bytes().split(b"\n")
        world.ledger_path.write_bytes(b"\n".join([lines[0], b"{broken", *lines[1:]]))
        (normal, n_exit), (deploy, d_exit) = self.both(world)
        assert (normal["ledger"], n_exit, deploy["ledger"], d_exit) == (FAIL, 2, WARN, 1)

    def test_an_unwritable_audit_log_a_bad_rules_file_and_a_bad_themes_file_are_warnings_in_deploy_mode(self, world, monkeypatch):
        world.audit_path.mkdir()                                       # a directory where the audit log should be: not writable
        rules_file, themes_file = world.tmp / "rules.toml", world.tmp / "themes.toml"
        rules_file.write_text("[[[ not toml", encoding="utf-8")
        themes_file.write_text("[[[ not toml", encoding="utf-8")
        monkeypatch.setattr(settings, "SOCIAL_RULES_FILE", rules_file)
        monkeypatch.setattr(settings, "SOCIAL_THEMES_FILE", themes_file)
        (normal, n_exit), (deploy, d_exit) = self.both(world)
        assert (normal["audit"], normal["rules"], normal["themes"], n_exit) == (FAIL, FAIL, FAIL, 2)
        assert (deploy["audit"], deploy["rules"], deploy["rules-file"], deploy["themes"], deploy["themes-file"], d_exit) == (WARN, OK, WARN, OK, WARN, 1)

    def test_an_unreadable_gallery_index_and_taxonomy_are_warnings_in_deploy_mode(self, world):
        world.index.write_text("{", encoding="utf-8")
        world.taxonomy.write_text("{", encoding="utf-8")
        (normal, n_exit), (deploy, d_exit) = self.both(world)
        assert (normal["gallery-index"], normal["taxonomy"], n_exit) == (FAIL, FAIL, 2)
        assert (deploy["gallery-index"], deploy["taxonomy"], d_exit) == (WARN, WARN, 1)

    def test_a_missing_extension_still_fails_in_deploy_mode(self, world, monkeypatch):
        monkeypatch.setattr(social_hooks, "EXTENSIONS", (*social_hooks.EXTENSIONS, "social_deleted_gate"))
        (normal, n_exit), (deploy, d_exit) = self.both(world)
        assert (normal["extensions"], n_exit, deploy["extensions"], d_exit) == (FAIL, 2, FAIL, 2)

    def test_a_cli_that_does_not_import_fails_in_deploy_mode_and_a_working_one_is_ok(self, world, monkeypatch):
        (_, _), (deploy, d_exit) = self.both(world)
        assert (deploy["cli"], d_exit) == (OK, 0)
        monkeypatch.setitem(sys.modules, "bio_automation.cli", None)
        (_, _), (broken, b_exit) = self.both(world)
        assert (broken["cli"], b_exit) == (FAIL, 2)

    def test_the_deploy_checks_are_the_normal_ones_plus_cli_and_the_two_file_checks(self, world):
        (normal, _), (deploy, _) = self.both(world)
        assert list(normal) == NAMES
        assert list(deploy) == ["rules", "extensions", "cli", "themes", "rules-file", "themes-file", *[n for n in NAMES[3:] if n != "script-drift"]]

    def test_the_command_prints_the_summary_line_last_and_exits_1_for_environmental_trouble(self, world, plenty_of_disk, monkeypatch, capsys):
        (world.oauth / "account.json").unlink()
        code, out = command(monkeypatch, capsys, "--deploy")
        lines = out.out.splitlines()
        assert (code, lines[-1]) == (1, "selfcheck: warn")
        assert "[warn] bluesky-session: no Bluesky session" in out.out
        code, out = command(monkeypatch, capsys)
        assert (code, out.out.splitlines()[-1]) == (2, "selfcheck: fail")

    def test_deploy_with_json_still_ends_with_the_summary_line(self, world, plenty_of_disk, monkeypatch, capsys):
        code, out = command(monkeypatch, capsys, "--deploy", "--json")
        assert (code, out.out.splitlines()[-1]) == (0, "selfcheck: ok")


# ----------------------------------------------------------------------------------------------------------------------- script drift

class TestScriptDrift:
    def repo_names(self):
        return sorted(f.name for f in social_selfcheck.repo_scripts_dir().glob("*.ps1"))

    def test_the_repo_has_the_scheduled_task_scripts_the_check_compares(self):
        assert self.repo_names() == ["notify-to-inbox.ps1", "register-social-tasks.ps1", "run-qdrant-sync.ps1", "run-social-post.ps1", "run-social-prep.ps1", "run-social-watch.ps1"]

    def test_identical_copies_are_ok_and_the_message_counts_them(self, world):
        c = world.run()["script-drift"]
        assert (c.level, c.message) == (OK, f"6 scheduled-task scripts in {world.scripts} match the repo (sha256)")

    def test_a_changed_copy_warns_and_names_the_file(self, world):
        with open(world.scripts / "run-social-post.ps1", "ab") as f:
            f.write(b"# local edit\n")
        c = world.run()["script-drift"]
        assert c.level == WARN
        assert c.message == (f"scheduled-task scripts drifted (differ from the repo: run-social-post.ps1); run scripts\\deploy_prod.ps1 or copy them")

    def test_a_missing_copy_warns_and_names_the_file(self, world):
        (world.scripts / "register-social-tasks.ps1").unlink()
        c = world.run()["script-drift"]
        assert c.level == WARN
        assert c.message == (f"scheduled-task scripts drifted (missing in {world.scripts}: register-social-tasks.ps1); run scripts\\deploy_prod.ps1 or copy them")

    def test_missing_and_changed_are_both_reported(self, world):
        (world.scripts / "notify-to-inbox.ps1").unlink()
        (world.scripts / "run-social-watch.ps1").write_bytes(b"x")
        c = world.run()["script-drift"]
        assert c.message.startswith(f"scheduled-task scripts drifted (missing in {world.scripts}: notify-to-inbox.ps1; differ from the repo: run-social-watch.ps1)")

    @pytest.mark.skipif(sys.platform != "win32", reason="exclusive file handles are a Windows thing; the check's OSError branch is the same everywhere")
    def test_a_locked_copy_warns_naming_the_file_instead_of_crashing_and_the_unlocked_twin_is_ok(self, world):
        import ctypes
        from ctypes import wintypes
        create = ctypes.windll.kernel32.CreateFileW
        create.restype = wintypes.HANDLE
        create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        handle = create(str(world.scripts / "run-social-post.ps1"), 0x80000000, 0, None, 3, 0, None)       # GENERIC_READ, share mode 0, OPEN_EXISTING
        assert handle not in (None, wintypes.HANDLE(-1).value)
        try:
            c = world.run()["script-drift"]
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
        assert (c.level, c.message) == (WARN, "scheduled-task scripts drifted (unreadable (locked or access denied): run-social-post.ps1); run scripts\\deploy_prod.ps1 or copy them")
        assert world.run()["script-drift"].level == OK

    def test_the_rollback_script_and_unknown_files_are_ignored(self, world):
        (world.scripts / "run-social.ps1").write_text("the old single script, edited by hand", encoding="utf-8")
        (world.scripts / "other.ps1").write_text("not ours", encoding="utf-8")
        assert world.run()["script-drift"].level == OK

    def test_a_repo_copy_named_run_social_is_not_demanded_in_the_deployed_folder(self, tmp_path):
        repo, dst = tmp_path / "repo", tmp_path / "dst"
        repo.mkdir(), dst.mkdir()
        (repo / "run-social.ps1").write_text("a", encoding="utf-8")
        (repo / "a.ps1").write_text("b", encoding="utf-8")
        (dst / "a.ps1").write_text("b", encoding="utf-8")
        assert social_selfcheck.check_script_drift(repo, dst) == social_selfcheck.Check("script-drift", OK, f"1 scheduled-task scripts in {dst} match the repo (sha256)")

    def test_a_missing_deployed_folder_warns(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "a.ps1").write_text("b", encoding="utf-8")
        c = social_selfcheck.check_script_drift(repo, tmp_path / "nowhere")
        assert (c.level, c.message) == (WARN, f"scheduled-task script folder {tmp_path / 'nowhere'} does not exist (1 repo scripts not deployed)")

    def test_a_missing_repo_folder_warns(self, tmp_path):
        c = social_selfcheck.check_script_drift(tmp_path / "norepo", tmp_path)
        assert c.level == WARN and c.message == f"cannot compare the scheduled-task scripts: no {tmp_path / 'norepo'} next to this package"

    def test_drift_is_never_a_failure_so_the_exit_code_is_1(self, world):
        (world.scripts / "run-social-post.ps1").write_bytes(b"x")
        out = social_selfcheck.selfcheck(drives={"system": Path("C:/sys"), "data": Path("D:/data")}, free=lambda p: 100 * GB, now=NOW,
                                         audit=Audit(world.tmp / "sc-audit.jsonl"))
        assert (out["level"], out["exit"]) == ("warn", 1)
        assert [c["name"] for c in out["checks"] if c["level"] != OK] == ["script-drift"]

    def test_deploy_mode_does_not_run_the_check(self, world):
        (world.scripts / "run-social-post.ps1").write_bytes(b"x")
        out = social_selfcheck.selfcheck(drives={"system": Path("C:/sys"), "data": Path("D:/data")}, free=lambda p: 100 * GB, now=NOW, deploy=True,
                                         audit=Audit(world.tmp / "sc-audit.jsonl"))
        assert "script-drift" not in [c["name"] for c in out["checks"]]
