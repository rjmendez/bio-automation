"""Backup retention (size cap), the off-drive copy hook, drafts_hashes.json and `bio social-restore-check` (restore the newest backup into a
scratch folder and validate it). Every file lives under tmp_path; the off-drive "other drive" is a folder with an injected drive_of."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import pytest
from test_social import NOW, write_draft
from test_social_state import CONTENT, T0, logged, rec, seed_files, write_lines

from bio_automation import cli, settings, social_state
from bio_automation.social import Ledger, SocialError, approve, draft_hash
from bio_automation.social_state import MANIFEST

BIG = b"x" * 600_000


def drives(p) -> str:
    """A fake drive map: anything named backups (or below) is drive A, everything else drive B."""
    return "A" if "backups" in Path(p).parts else "B"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def draft_dir(draft_id: str) -> Path:
    return Path(settings.SOCIAL_DIR) / "drafts" / draft_id


def approved_draft(draft_id="2026-10-obs1", obs=1):
    """A drafted and approved draft in the configured ledger and drafts dir; returns its approval hash."""
    drafts = Path(settings.SOCIAL_DIR) / "drafts"
    write_draft(drafts, draft_id, obs, "2026-10")
    ledger = Ledger()
    ledger.append("drafted", draft_id, obs_id=obs, month="2026-10", now=NOW - dt.timedelta(days=1))
    return approve(ledger, drafts, draft_id)["hash"]


class TestDraftHashes:
    def test_the_backup_records_the_sha256_of_every_draft_folder(self):
        seed_files()
        write_draft(Path(settings.SOCIAL_DIR) / "drafts", "d-a", 1, "2026-10")
        write_draft(Path(settings.SOCIAL_DIR) / "drafts", "d-b", 2, "2026-10")
        folder = social_state.make_backup(now=T0)
        got = json.loads((folder / "drafts_hashes.json").read_text(encoding="utf-8"))
        assert got == {"d-a": {"sha256": draft_hash(draft_dir("d-a"))}, "d-b": {"sha256": draft_hash(draft_dir("d-b"))}}
        info = json.loads((folder / MANIFEST).read_text(encoding="utf-8"))["files"]["drafts_hashes.json"]
        assert info["sha256"] == sha((folder / "drafts_hashes.json").read_bytes())

    def test_a_broken_draft_is_recorded_with_its_error_not_skipped(self):
        seed_files()
        write_draft(Path(settings.SOCIAL_DIR) / "drafts", "d-ok", 1, "2026-10")
        write_draft(Path(settings.SOCIAL_DIR) / "drafts", "d-bad", 2, "2026-10")
        (draft_dir("d-bad") / "1.jpg").unlink()
        got = json.loads((social_state.make_backup(now=T0) / "drafts_hashes.json").read_text(encoding="utf-8"))
        assert got["d-ok"] == {"sha256": draft_hash(draft_dir("d-ok"))}
        assert got["d-bad"]["error"].startswith("FileNotFoundError")

    def test_no_drafts_folder_means_no_drafts_hashes_file(self):
        seed_files()
        folder = social_state.make_backup(now=T0)
        assert not (folder / "drafts_hashes.json").exists()
        assert "drafts_hashes.json" not in json.loads((folder / MANIFEST).read_text(encoding="utf-8"))["files"]

    def test_restore_checks_the_drafts_hash_file_but_never_writes_it_anywhere(self):
        live = seed_files()
        write_draft(Path(settings.SOCIAL_DIR) / "drafts", "d-a", 1, "2026-10")
        folder = social_state.make_backup(now=T0)
        assert sorted(f for f, _ in social_state.restore(folder.name)) == sorted(live)          # drafts_hashes.json is not among the files it would put back
        (folder / "drafts_hashes.json").write_bytes(b"{}")
        with pytest.raises(social_state.IntegrityError, match="drafts_hashes.json does not match its recorded sha256"):
            social_state.restore(folder.name)


class TestSizeCap:
    def make(self, n, *, max_mb, corrupt_from=None):
        live = seed_files()
        names = []
        for i in range(n):
            live["prints.csv"].write_bytes(BIG)
            live["ledger.jsonl"].write_bytes(b"garbage\n" if corrupt_from is not None and i >= corrupt_from else CONTENT["ledger"])
            names.append(social_state.make_backup(now=T0 + dt.timedelta(days=i), max_mb=max_mb).name)
        return names

    def test_over_the_cap_the_oldest_go_first_and_the_newest_stays(self):
        names = self.make(3, max_mb=1)
        assert [f.name for f in social_state.backups()] == [names[-1]]
        assert [r["removed"] for r in logged() if r["event"] == "backup_made"] == [[], [names[0]], [names[1]]]

    def test_under_the_cap_or_with_no_cap_everything_stays(self):
        assert len(self.make(3, max_mb=0)) == len(social_state.backups()) == 3

    def test_a_cap_that_fits_three_keeps_exactly_the_newest_three(self):
        names = self.make(5, max_mb=2)                       # 5 x 600 KB = 3 MB; 3 x 600 KB fits 2 MiB, 4 do not
        assert [f.name for f in social_state.backups()] == names[2:]

    def test_the_newest_clean_backup_survives_the_cap(self):
        names = self.make(4, max_mb=1, corrupt_from=1)
        assert [f.name for f in social_state.backups()] == [names[0], names[3]]


class TestOffsite:
    def test_off_by_default_nothing_is_copied_and_nothing_audited(self, tmp_path):
        seed_files()
        social_state.make_backup(now=T0)
        assert [r["event"] for r in logged()] == ["backup_made"]

    def test_a_backup_is_copied_to_the_other_drive_read_back_and_audited(self, tmp_path):
        seed_files()
        write_draft(Path(settings.SOCIAL_DIR) / "drafts", "d-a", 1, "2026-10")
        dest = tmp_path / "other-drive" / "bio"
        folder = social_state.make_backup(now=T0, offsite=dest, drive_of=drives)
        copy = dest / folder.name
        names = sorted(p.name for p in folder.iterdir())
        assert sorted(p.name for p in copy.iterdir()) == names
        assert all((copy / n).read_bytes() == (folder / n).read_bytes() for n in names)
        events = [r for r in logged() if r["event"] == "backup_offsite"]
        assert len(events) == 1 and events[0]["name"] == folder.name and events[0]["dest"] == str(dest)
        assert [p.name for p in dest.iterdir()] == [folder.name]               # no .tmp folder left behind

    def test_the_destination_is_pruned_to_the_newest_keep(self, tmp_path):
        seed_files()
        dest = tmp_path / "other"
        made = [social_state.make_backup(now=T0 + dt.timedelta(days=i), keep=2, offsite=dest, drive_of=drives).name for i in range(3)]
        assert sorted(p.name for p in dest.iterdir()) == made[1:]

    def test_the_same_drive_is_refused_audited_and_the_backup_itself_still_exists(self, tmp_path):
        seed_files()
        dest = tmp_path / "same-drive"
        folder = social_state.make_backup(now=T0, offsite=dest)               # the real device check: tmp_path is one drive
        assert folder.is_dir() and not dest.exists()
        failed = [r for r in logged() if r["event"] == "backup_offsite_failed"]
        assert len(failed) == 1 and "is on the same drive as the backups" in failed[0]["error"]
        assert not [r for r in logged() if r["event"] == "backup_offsite"]

    def test_the_setting_turns_it_on(self, tmp_path, monkeypatch):
        seed_files()
        monkeypatch.setattr(settings, "SOCIAL_BACKUP_OFFSITE_DIR", str(tmp_path / "via-setting"))
        social_state.make_backup(now=T0)
        assert [r["event"] for r in logged()] == ["backup_made", "backup_offsite_failed"]   # reached the hook; refused only for the same drive

    def test_an_unwritable_destination_is_audited_and_never_raises(self, tmp_path):
        seed_files()
        blocker = tmp_path / "file-not-dir"
        blocker.write_text("x")
        folder = social_state.make_backup(now=T0, offsite=blocker / "sub", drive_of=drives)
        assert folder.is_dir()
        assert [r["event"] for r in logged()] == ["backup_made", "backup_offsite_failed"]

    def test_a_backup_file_that_no_longer_matches_its_manifest_is_not_copied_off_drive(self, tmp_path):
        seed_files()
        folder = social_state.make_backup(now=T0)
        (folder / "ledger.jsonl").write_bytes(b"damaged\n")
        dest = tmp_path / "other"
        assert social_state.offsite_copy(folder, dest, drive_of=drives) is None
        assert not (dest / folder.name).exists() and list(dest.iterdir()) == []
        failed = [r for r in logged() if r["event"] == "backup_offsite_failed"]
        assert len(failed) == 1 and "ledger.jsonl read back from" in failed[0]["error"]

    def test_an_undamaged_backup_copies_through_the_same_call(self, tmp_path):
        seed_files()
        folder = social_state.make_backup(now=T0)
        assert social_state.offsite_copy(folder, tmp_path / "other", drive_of=drives) == tmp_path / "other" / folder.name


class TestRestoreCheck:
    def healthy(self):
        live = seed_files()
        write_lines(live["ledger.jsonl"], [rec("drafted", "d1", 0, obs_id=1, month="2026-10")])
        return live

    def check(self, **kw):
        kw.setdefault("now", T0 + dt.timedelta(hours=1))
        return social_state.restore_check(**kw)

    def test_a_healthy_backup_passes_and_names_every_restored_file(self, tmp_path):
        live = self.healthy()
        social_state.make_backup(now=T0)
        res = self.check(scratch_root=tmp_path / "scratch")
        assert res["ok"] is True and res["problems"] == []
        assert sorted(res["files"]) == sorted(live)
        assert res["files"]["ledger.jsonl"] == sha(live["ledger.jsonl"].read_bytes())
        assert [r["event"] for r in logged()][-1] == "restore_check"

    def test_the_scratch_folder_is_removed_and_no_live_file_is_touched(self, tmp_path):
        live = self.healthy()
        social_state.make_backup(now=T0)
        before = {n: p.read_bytes() for n, p in live.items() if n != "audit.jsonl"}
        self.check(scratch_root=tmp_path / "scratch")
        assert list((tmp_path / "scratch").iterdir()) == []
        assert {n: live[n].read_bytes() for n in before} == before
        assert len(social_state.backups()) == 1                                # and no pre-restore backup was made

    def test_it_picks_the_newest_backup_unless_one_is_named(self, tmp_path):
        self.healthy()
        old = social_state.make_backup(now=T0, keep=None)
        new = social_state.make_backup(now=T0 + dt.timedelta(hours=2), keep=None)
        assert self.check(scratch_root=tmp_path / "s", now=T0 + dt.timedelta(hours=3))["name"] == new.name
        assert self.check(scratch_root=tmp_path / "s", now=T0 + dt.timedelta(hours=3), name=old.name)["name"] == old.name

    def test_a_backup_of_a_corrupt_ledger_fails(self, tmp_path):
        live = self.healthy()
        live["ledger.jsonl"].write_bytes(b"garbage\n")
        social_state.make_backup(now=T0)
        res = self.check(scratch_root=tmp_path / "s")
        assert res["ok"] is False
        assert len(res["problems"]) == 1 and res["problems"][0].startswith("restored ledger: bad_json at line 1")
        assert [r["event"] for r in logged()][-1] == "restore_check_failed"

    def test_a_file_changed_after_the_backup_fails_on_its_hash(self, tmp_path):
        self.healthy()
        folder = social_state.make_backup(now=T0)
        (folder / "rules.toml").write_bytes(b"changed")
        res = self.check(scratch_root=tmp_path / "s")
        assert res["ok"] is False
        assert len(res["problems"]) == 1 and res["problems"][0].startswith("rules.toml: sha256 ") and "differs from the manifest" in res["problems"][0]
        assert "rules.toml" not in res["files"]

    def test_an_audit_log_with_a_garbled_line_fails_and_a_clean_one_passes(self, tmp_path):
        live = self.healthy()
        live["audit.jsonl"].write_bytes(b'{"event": "a"}\n{"event": "b"\n[1]\n')
        social_state.make_backup(now=T0)
        res = self.check(scratch_root=tmp_path / "s")
        assert res["problems"] == ["restored audit log: 2 line(s) are not JSON objects"]
        live["audit.jsonl"].write_bytes(b'{"event": "a"}\n')
        social_state.make_backup(now=T0 + dt.timedelta(minutes=5))
        assert self.check(scratch_root=tmp_path / "s")["problems"] == []

    def test_a_backup_older_than_the_limit_fails_a_fresh_one_passes(self, tmp_path):
        self.healthy()
        social_state.make_backup(now=T0)
        fresh = self.check(scratch_root=tmp_path / "s", now=T0 + dt.timedelta(days=1, hours=23))
        stale = self.check(scratch_root=tmp_path / "s", now=T0 + dt.timedelta(days=3))
        assert fresh["problems"] == []
        assert stale["problems"] == ["backup 20261021-080005 is 3.0 days old, more than 2"]

    def test_an_approved_draft_edited_after_its_approval_fails_unedited_passes(self, tmp_path):
        live = seed_files()
        approved_draft()
        social_state.make_backup(now=T0)
        assert self.check(scratch_root=tmp_path / "s")["problems"] == []
        doc = json.loads((draft_dir("2026-10-obs1") / "draft.json").read_text(encoding="utf-8"))
        doc["images"][0]["alt"] = "edited after approval"
        (draft_dir("2026-10-obs1") / "draft.json").write_text(json.dumps(doc), encoding="utf-8")
        social_state.make_backup(now=T0 + dt.timedelta(minutes=5))
        problems = self.check(scratch_root=tmp_path / "s")["problems"]
        assert len(problems) == 1 and problems[0].startswith("draft 2026-10-obs1 is approved: its folder hashes to ")
        assert live["ledger.jsonl"].is_file()

    def test_a_posted_draft_with_changed_files_is_not_a_problem(self, tmp_path):
        seed_files()
        approved_draft()
        ledger = Ledger()
        ledger.append("posting", "2026-10-obs1", obs_id=1)
        ledger.append("posted", "2026-10-obs1", obs_id=1, uri="at://x/y/z")
        (draft_dir("2026-10-obs1") / "draft.json").write_text(json.dumps({"images": []}), encoding="utf-8")
        social_state.make_backup(now=T0)
        assert self.check(scratch_root=tmp_path / "s")["problems"] == []

    def test_an_approved_draft_missing_from_the_hash_list_fails(self, tmp_path):
        seed_files()
        approved_draft()
        write_draft(Path(settings.SOCIAL_DIR) / "drafts", "other", 9, "2026-10")
        (draft_dir("2026-10-obs1") / "1.jpg").unlink()
        social_state.make_backup(now=T0)
        problems = self.check(scratch_root=tmp_path / "s")["problems"]
        assert len(problems) == 1 and problems[0].startswith("draft 2026-10-obs1 is approved but the backup has no usable hash of its folder (FileNotFoundError")

    def test_without_backups_it_fails_and_says_where_it_looked(self, tmp_path):
        res = self.check(scratch_root=tmp_path / "s")
        assert res["ok"] is False and res["name"] is None
        assert res["problems"] == [f"no backup in {settings.SOCIAL_BACKUP_DIR}"]

    def test_an_unknown_name_is_a_clear_problem(self, tmp_path):
        self.healthy()
        social_state.make_backup(now=T0)
        res = self.check(scratch_root=tmp_path / "s", name="20200101-000000")
        assert res["ok"] is False and len(res["problems"]) == 1 and res["problems"][0].startswith("no backup '20200101-000000' in ")

    def test_a_path_in_the_name_is_refused(self, tmp_path):
        self.healthy()
        social_state.make_backup(now=T0)
        res = self.check(scratch_root=tmp_path / "s", name="..\\..\\x")
        assert res["problems"] == ["'..\\\\..\\\\x' is not a backup name (see `bio social-restore --list`)"] or "is not a backup name" in res["problems"][0]


class TestCommand:
    def run(self, monkeypatch, capsys, *argv):
        monkeypatch.setattr(sys, "argv", ["bio", *argv])
        return cli.main(), capsys.readouterr().out

    def test_exit_zero_and_a_summary_line_when_the_backup_restores_clean(self, monkeypatch, capsys, tmp_path):
        live = seed_files()
        write_lines(live["ledger.jsonl"], [rec("drafted", "d1", 0, obs_id=1, month="2026-10")])
        social_state.make_backup()
        code, out = self.run(monkeypatch, capsys, "social-restore-check", "--scratch", str(tmp_path / "s"))
        name = social_state.backups()[-1].name
        assert code == 0
        assert out.splitlines()[0] == f"restore check of backup {name}: ok (7 files restored to a scratch folder and removed)"

    def test_exit_one_and_the_problem_when_it_does_not(self, monkeypatch, capsys, tmp_path):
        live = seed_files()
        live["ledger.jsonl"].write_bytes(b"garbage\n")
        social_state.make_backup()
        code, out = self.run(monkeypatch, capsys, "social-restore-check", "--scratch", str(tmp_path / "s"))
        assert code == 1
        assert "FAILED" in out.splitlines()[0] and "  problem: restored ledger: bad_json at line 1" in out
