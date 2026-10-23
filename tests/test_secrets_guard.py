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
from bio_automation.social_selfcheck import FAIL, OK, WARN

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


class TestRedactMoreForms:
    def test_cookie_header_values_query_tokens_and_password_pairs_are_replaced_keeping_the_key(self):
        assert secrets_guard.redact_text("Set-Cookie: session=abcdefghij; Path=/") == f"Set-Cookie: {R}"
        assert secrets_guard.redact_text("GET /x?access_token=abcdefghij1234567890&x=1 failed") == f"GET /x?access_token={R}&x=1 failed"
        assert secrets_guard.redact_text("login password=hunter2hunter2 refused") == f"login password={R} refused"
        assert secrets_guard.redact_text('{"refresh_token": "abcdefghij"}') == '{"refresh_token": "' + R + '"}'
        assert secrets_guard.redact_text("GET /x?a=1&token=abcdefgh") == f"GET /x?a=1&token={R}"

    def test_prose_and_short_values_are_left_alone(self):                         # positive twin
        for t in ("the password was changed", "token: ok", "cookie jar is empty", "access_token=abc", "posted 3 photos?x=1&y=2"):
            assert secrets_guard.redact_text(t) == t

    def test_an_already_redacted_value_is_not_redacted_again_so_a_clean_log_stays_clean(self):
        text = '{"access_token": "[REDACTED]", "Cookie": "[REDACTED]"} password=[REDACTED]'
        assert secrets_guard.redact_text(text) == text
        assert secrets_guard.scan_text(text) == ()

    def test_bytes_are_decoded_and_redacted_and_exceptions_become_redacted_text(self):
        assert secrets_guard.redact(b"sent Bearer " + (B * 2).encode()) == f"sent Bearer {R}"
        assert secrets_guard.redact(ValueError(f"401 Authorization: DPoP {JWT}")) == f"401 Authorization: {R}"
        assert secrets_guard.redact({"error": RuntimeError("password=hunter2hunter2")}) == {"error": f"password={R}"}

    def test_numbers_none_and_bools_pass_through_unchanged(self):                 # positive twin
        assert secrets_guard.redact({"a": 1, "b": 2.5, "c": None, "d": True}) == {"a": 1, "b": 2.5, "c": None, "d": True}

    def test_the_audit_log_writes_an_exception_field_redacted_and_as_valid_json(self, tmp_path):
        Audit(tmp_path / "a.jsonl", run="r1").log("post_failed", error=RuntimeError("HTTP 400 Bearer " + B * 2))
        assert json.loads((tmp_path / "a.jsonl").read_text(encoding="utf-8"))["error"] == f"HTTP 400 Bearer {R}"


class TestScanBoundsAndSeverity:
    def test_a_huge_file_is_read_by_its_tail_only_and_the_file_cap_is_reported(self, tmp_path):
        (tmp_path / "big.log").write_text(f"Bearer {B}{B}\n" + "x" * 5000 + "\nclean tail\n", encoding="utf-8")
        assert secrets_guard.scan([tmp_path], max_file_bytes=1000) == secrets_guard.ScanResult([], 1, False)      # the token is in the unread head
        assert [f.path.name for f in secrets_guard.scan([tmp_path], max_file_bytes=10_000).findings] == ["big.log"]
        for i in range(3):
            (tmp_path / f"f{i}.log").write_text("fine", encoding="utf-8")
        res = secrets_guard.scan([tmp_path], max_files=2)
        assert (res.files, res.capped) == (2, True)

    def test_a_weak_form_only_warns_and_a_real_shape_fails(self, world):
        world.mp.setattr(settings, "SOCIAL_BACKUP_DIR", world.tmp / "backups")
        log = world.tmp / "cache" / "logs" / "social.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("login password=hunter2hunter2 refused\n", encoding="utf-8")
        c = world.run()["secret-scan"]
        assert (c.level, str(log) in c.message, "hunter2" in c.message) == (WARN, True, False)
        log.write_text(f"leaked {JWT}\n", encoding="utf-8")
        assert world.run()["secret-scan"].level == FAIL

    def test_in_deploy_mode_a_secret_finding_is_only_a_warning(self):
        from bio_automation import social_selfcheck as sc
        out = sc.for_deploy([sc.Check("secret-scan", FAIL, "1 file(s) hold credential-like text")])
        assert [(c.level, c.message.endswith("[environmental: a warning in --deploy mode]")) for c in out] == [(WARN, True)]


class TestKeyMaterialInDirectories:
    def test_a_directory_source_holding_a_key_file_is_key_material_and_a_clean_one_is_not(self, tmp_path):
        d = tmp_path / "src"
        (d / "sub").mkdir(parents=True)
        (d / "sub" / "notes.txt").write_text("x", encoding="utf-8")
        assert secrets_guard.is_key_material(d) is False
        (d / "sub" / "client_key.jwk.json").write_text("{}", encoding="utf-8")
        assert secrets_guard.is_key_material(d) is True
