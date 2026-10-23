"""secrets_guard: redaction (Authorization, DPoP, JWK private members), the scan, the selfcheck item and the backup refusal.
Token-shaped strings are built at run time so this file itself holds none."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_social_selfcheck import NAMES, world  # noqa: F401  (the world fixture)
from test_social_state import T0, seed_files

from bio_automation import secrets_guard, settings, social_state
from bio_automation.social import SocialError
from bio_automation.social_rules import Audit
from bio_automation.social_selfcheck import FAIL, OK

R = secrets_guard.REDACTED
B = "A1b2C3d4E5f6G7h8I9j0"                       # 20 base64url chars
JWT = "eyJ" + "hbGciOiJFUzI1NiJ9" + "." + "eyJzdWIiOiJ4In0" + "." + "c2lnbmF0dXJlMTIzNDU2"
PEM = "-----BEGIN " + "PRIVATE KEY-----\n" + B * 3 + "\n-----END " + "PRIVATE KEY-----"


class TestRedact:
    def test_secret_header_and_field_names_are_replaced_whatever_the_case(self):
        out = secrets_guard.redact({"Authorization": "DPoP " + JWT, "DPoP": JWT, "Cookie": "s=1", "access_token": B, "ok": "visible"})
        assert out == {"Authorization": R, "DPoP": R, "Cookie": R, "access_token": R, "ok": "visible"}

    def test_jwk_private_members_d_p_q_are_replaced_and_public_ones_kept(self):
        jwk = {"kty": "EC", "crv": "P-256", "x": "PUBX", "y": "PUBY", "d": B, "kid": "k1"}
        rsa = {"kty": "RSA", "n": "PUBN", "e": "AQAB", "d": B, "p": B, "q": B, "dp": B, "dq": B, "qi": B}
        assert secrets_guard.redact({"key": jwk, "rsa": rsa}) == {
            "key": {"kty": "EC", "crv": "P-256", "x": "PUBX", "y": "PUBY", "d": R, "kid": "k1"},
            "rsa": {"kty": "RSA", "n": "PUBN", "e": "AQAB", "d": R, "p": R, "q": R, "dp": R, "dq": R, "qi": R}}

    def test_a_dict_without_kty_keeps_its_d_and_p_keys(self):                      # positive twin of the JWK rule
        assert secrets_guard.redact({"d": 3, "p": "x"}) == {"d": 3, "p": "x"}

    def test_strings_lose_jwts_bearer_values_authorization_lines_and_pem_blocks(self):
        assert secrets_guard.redact_text(f"HTTP 401 for Authorization: DPoP {JWT} retry") == f"HTTP 401 for Authorization: {R} retry"
        assert secrets_guard.redact_text(f"sent Bearer {B}{B} ok") == f"sent Bearer {R} ok"
        assert secrets_guard.redact_text(f"token {JWT}!") == f"token {R}!"
        assert secrets_guard.redact_text("x " + PEM + " y") == f"x {R} y"
        assert secrets_guard.redact_text('{"kty":"EC","d":"' + B + '"}') == '{"kty":"EC","d":"' + R + '"}'

    def test_plain_text_is_unchanged_and_the_input_is_not_modified(self):
        data = {"msg": "posted 3 photos of Amanita muscaria", "n": [1, {"a": "b"}]}
        snapshot = json.dumps(data)
        assert secrets_guard.redact(data) == data
        secrets_guard.redact({"Authorization": B})
        assert json.dumps(data) == snapshot

    def test_the_audit_log_writes_redacted_fields_to_disk(self, tmp_path):
        audit = Audit(tmp_path / "a.jsonl", run="r1")
        audit.log("send_failed", error=f"401 Authorization: DPoP {JWT}", headers={"DPoP": JWT, "Accept": "json"})
        text = (tmp_path / "a.jsonl").read_text(encoding="utf-8")
        rec = json.loads(text)
        assert (rec["error"], rec["headers"]) == (f"401 Authorization: {R}", {"DPoP": R, "Accept": "json"})
        assert JWT not in text


class TestScan:
    def test_each_token_shape_is_found_by_kind_and_the_match_is_not_returned(self, tmp_path):
        files = {"jwt.log": f"x {JWT}", "bearer.log": f"Bearer {B}{B}", "pem.txt": PEM, "jwk.json": '{"d":"' + B + '"}', "clean.log": "all fine"}
        for n, t in files.items():
            (tmp_path / n).write_text(t, encoding="utf-8")
        found = {f.path.name: f.kinds for f in secrets_guard.scan_paths([tmp_path])}
        assert found == {"jwt.log": ("jwt",), "bearer.log": ("bearer",), "pem.txt": ("pem",), "jwk.json": ("jwk-private",)}
        assert B not in repr(found)

    def test_a_clean_folder_has_no_findings_and_a_missing_one_is_skipped(self, tmp_path):
        (tmp_path / "a.log").write_text("hello", encoding="utf-8")
        assert secrets_guard.scan_paths([tmp_path, tmp_path / "nope"]) == []


class TestSelfcheckItem:
    def test_clean_logs_audit_drafts_and_backups_are_ok(self, world):
        c = world.run()["secret-scan"]
        assert (c.level, c.message) == (OK, "no credential-like text in the logs, audit, alerts, drafts, digests or backups")

    @pytest.mark.parametrize("where", ["audit", "log", "draft", "backup"])
    def test_a_token_in_any_scanned_place_fails_naming_only_the_file(self, world, where):
        target = {"audit": world.audit_path, "log": world.tmp / "cache" / "logs" / "social.log",
                  "draft": world.drafts / "zz" / "draft.json", "backup": world.tmp / "backups" / "20261021-080005" / "ledger.jsonl"}[where]
        world.mp.setattr(settings, "SOCIAL_BACKUP_DIR", world.tmp / "backups")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"leaked {JWT}\n", encoding="utf-8")
        c = world.run()["secret-scan"]
        assert c.level == FAIL
        assert str(target) in c.message and "(jwt)" in c.message
        assert JWT not in c.message


class TestBackupRefusesKeyMaterial:
    def test_a_source_inside_the_oauth_dir_or_named_like_a_key_is_refused_and_audited(self, tmp_path, monkeypatch):
        live = seed_files()
        oauth = tmp_path / "oauth"
        oauth.mkdir()
        key = oauth / "client_key.jwk.json"
        key.write_text('{"kty":"EC"}', encoding="utf-8")
        monkeypatch.setattr(settings, "BSKY_OAUTH_DIR", oauth)
        with pytest.raises(SocialError, match="never copied into a backup"):
            social_state.make_backup(sources={**live, "client_key.jwk.json": key}, now=T0)
        assert [r["error"] for r in Audit().records() if r["event"] == "backup_failed"] == ["refused: client_key.jwk.json is OAuth key material"]
        assert not settings.SOCIAL_BACKUP_DIR.exists() or social_state.backups() == []

    def test_the_normal_sources_still_back_up(self, tmp_path, monkeypatch):    # positive twin
        seed_files()
        monkeypatch.setattr(settings, "BSKY_OAUTH_DIR", tmp_path / "oauth")
        folder = social_state.make_backup(now=T0)
        assert folder.name == "20261021-080005"
        assert not any(secrets_guard.is_key_material(Path(p), [tmp_path / "oauth"]) for p in folder.iterdir())
