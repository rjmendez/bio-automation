"""CLI wiring for `bio export-zip` and `bio serve-downloads`: defaults, printed output, failure exit codes."""
from __future__ import annotations

import getpass
import socket
from pathlib import Path

import pytest
from PIL import Image, ImageCms

from bio_automation import cli
from bio_automation.bundle import BundleResult
from bio_automation.cli import build_parser, cmd_export_zip, cmd_serve_downloads

SRGB = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def jpeg(path: Path, quality=95):
    Image.new("RGB", (64, 48), (9, 99, 199)).save(path, "JPEG", quality=quality, icc_profile=SRGB)


class TestParsing:
    def test_serve_downloads_defaults(self):
        a = build_parser().parse_args(["serve-downloads", "--tailscale-login", "me@example.com"])
        assert (a.port, a.dir, a.tailscale_login) == (8780, None, ["me@example.com"])

    def test_tailscale_login_is_required_and_repeatable(self, capsys):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["serve-downloads"])
        assert "the following arguments are required: --tailscale-login" in capsys.readouterr().err
        a = build_parser().parse_args(["serve-downloads", "--tailscale-login", "a@x.y", "--tailscale-login", "b@x.y"])
        assert a.tailscale_login == ["a@x.y", "b@x.y"]

    @pytest.mark.parametrize("flag", ["--bind", "--allow", "--password-file", "--user"])
    def test_password_and_exposure_options_no_longer_exist(self, capsys, flag):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["serve-downloads", "--tailscale-login", "a@x.y", flag, "x"])
        assert f"unrecognized arguments: {flag} x" in capsys.readouterr().err

    def test_export_zip_needs_a_valid_kind(self, capsys):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["export-zip", "--kind", "adobe", "x"])
        assert "invalid choice: 'adobe'" in capsys.readouterr().err


class TestExportZip:
    def test_default_folder_is_d_bio_cache_downloads(self, monkeypatch, tmp_path):
        calls = []
        monkeypatch.delenv("BIO_DOWNLOAD_DIR", raising=False)
        monkeypatch.setattr(cli, "_tailnet_dns_name", lambda: None)
        monkeypatch.setattr(cli, "build_bundle", lambda kind, sources, out_dir, name=None: calls.append(out_dir) or
                            BundleResult(Path("D:/x.zip"), 1, (), 10, "0" * 64))
        assert cmd_export_zip(build_parser().parse_args(["export-zip", "--kind", "inat", str(tmp_path)])) == 0
        assert calls == [Path("D:/bio-cache/downloads")]

    def test_env_override_wins_and_explicit_flag_wins_over_env(self, monkeypatch, tmp_path):
        calls = []
        monkeypatch.setattr(cli, "_tailnet_dns_name", lambda: None)
        monkeypatch.setattr(cli, "build_bundle", lambda kind, sources, out_dir, name=None: calls.append(out_dir) or
                            BundleResult(Path("D:/x.zip"), 1, (), 10, "0" * 64))
        monkeypatch.setenv("BIO_DOWNLOAD_DIR", str(tmp_path / "from_env"))
        cmd_export_zip(build_parser().parse_args(["export-zip", "--kind", "inat", str(tmp_path)]))
        cmd_export_zip(build_parser().parse_args(["export-zip", "--kind", "inat", str(tmp_path), "--out-dir", str(tmp_path / "flag")]))
        assert calls == [tmp_path / "from_env", tmp_path / "flag"]

    def _build(self, tmp_path):
        src = tmp_path / "set"
        src.mkdir()
        jpeg(src / "a.jpg"); jpeg(src / "b.jpg")
        (src / "observations.csv").write_bytes(b"x\n")
        return build_parser().parse_args(["export-zip", "--kind", "inat", str(src), "--out-dir", str(tmp_path / "out"), "--name", "t"])

    def test_success_prints_path_summary_and_the_tailnet_and_ssh_fetch_lines(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(cli, "_tailnet_dns_name", lambda: "box.tail1234.ts.net")
        assert cmd_export_zip(self._build(tmp_path)) == 0
        out = capsys.readouterr().out.splitlines()
        zip_path = tmp_path / "out" / "t.zip"
        assert out[0] == str(zip_path)
        assert out[1].startswith("  2 photos + 1 extra file(s) ['observations.csv'], ")
        assert out[2] == "  fetch from any of your tailnet devices (no password; the tailnet identifies you):"
        assert out[3] == "    curl -C - -O https://box.tail1234.ts.net:8780/t.zip      (or open that URL in a browser)"
        assert out[4] == f"  or over ssh: scp {getpass.getuser()}@{socket.gethostname()}:{zip_path.as_posix()} ."

    def test_without_tailscale_there_is_no_curl_line_but_still_the_ssh_line(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr(cli, "_tailnet_dns_name", lambda: None)
        assert cmd_export_zip(self._build(tmp_path)) == 0
        out = capsys.readouterr().out.splitlines()
        assert not any("curl" in line for line in out)
        assert out[-1] == f"  or over ssh: scp {getpass.getuser()}@{socket.gethostname()}:{(tmp_path / 'out' / 't.zip').as_posix()} ."

    def test_violation_exits_1_with_the_reason_on_stderr_and_no_zip(self, tmp_path, capsys):
        src = tmp_path / "set"
        src.mkdir()
        jpeg(src / "bad.jpg", quality=70)
        args = build_parser().parse_args(["export-zip", "--kind", "inat", str(src), "--out-dir", str(tmp_path / "out")])
        assert cmd_export_zip(args) == 1
        cap = capsys.readouterr()
        assert cap.out == ""
        assert cap.err.startswith("export-zip failed: 1 of 1 files violate the inat standard, nothing written. First: bad.jpg: JPEG quality below 95")
        assert not (tmp_path / "out").exists()


class TestServeDownloads:
    def test_missing_folder_exits_1(self, tmp_path, capsys):
        args = build_parser().parse_args(["serve-downloads", "--dir", str(tmp_path / "nope"), "--tailscale-login", "a@x.y"])
        assert cmd_serve_downloads(args) == 1
        assert "serve-downloads failed: download folder does not exist" in capsys.readouterr().err

    def test_blank_login_exits_1_and_starts_nothing(self, tmp_path, capsys):
        (tmp_path / "dl").mkdir()
        args = build_parser().parse_args(["serve-downloads", "--dir", str(tmp_path / "dl"), "--tailscale-login", "  "])
        assert cmd_serve_downloads(args) == 1
        assert "refusing to start without at least one allowed tailnet login" in capsys.readouterr().err

    def test_it_serves_on_loopback_and_prints_the_publish_command(self, tmp_path, capsys, monkeypatch):
        (tmp_path / "dl").mkdir()
        started = []

        class Fake:                      # stands in for the blocking serve loop only; make_server itself is the real one
            def __init__(self, real):
                self.real = real
                self.logins = real.logins
            def serve_forever(self):
                started.append(self.real.server_address)
                raise KeyboardInterrupt
            def server_close(self):
                self.real.server_close()
        real_make = cli.make_server
        monkeypatch.setattr(cli, "make_server", lambda *a, **k: Fake(real_make(*a, **k)))
        args = build_parser().parse_args(["serve-downloads", "--dir", str(tmp_path / "dl"), "--port", "0", "--tailscale-login", "Me@Example.com"])
        assert cmd_serve_downloads(args) == 0
        assert started[0][0] == "127.0.0.1"
        out = capsys.readouterr().out.splitlines()
        assert out[0].endswith("for tailnet login(s) ['me@example.com']")
        assert out[1] == "  publish it with: tailscale serve --bg --https=0 http://127.0.0.1:0"
