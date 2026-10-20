"""CLI wiring for `bio export-zip` and `bio serve-downloads`: defaults, printed output, failure exit codes."""
from __future__ import annotations

import getpass
import socket
from pathlib import Path

from PIL import Image, ImageCms

from bio_automation import cli
from bio_automation.bundle import BundleResult
from bio_automation.cli import build_parser, cmd_export_zip, cmd_serve_downloads

SRGB = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def jpeg(path: Path, quality=95):
    Image.new("RGB", (64, 48), (9, 99, 199)).save(path, "JPEG", quality=quality, icc_profile=SRGB)


class TestParsing:
    def test_serve_downloads_defaults_are_loopback_and_a_fixed_port(self):
        a = build_parser().parse_args(["serve-downloads"])
        assert (a.bind, a.port, a.allow, a.user, a.password_file, a.dir) == \
               ("127.0.0.1", 8780, None, "owner", "F:/services/bio-downloads/password.txt", None)

    def test_allow_is_repeatable(self):
        a = build_parser().parse_args(["serve-downloads", "--allow", "10.0.0.0/8", "--allow", "192.168.1.0/24"])
        assert a.allow == ["10.0.0.0/8", "192.168.1.0/24"]

    def test_export_zip_needs_a_valid_kind(self, capsys):
        import pytest
        with pytest.raises(SystemExit):
            build_parser().parse_args(["export-zip", "--kind", "adobe", "x"])
        assert "invalid choice: 'adobe'" in capsys.readouterr().err


class TestExportZip:
    def test_default_folder_is_d_bio_cache_downloads(self, monkeypatch, tmp_path):
        calls = []
        monkeypatch.delenv("BIO_DOWNLOAD_DIR", raising=False)
        monkeypatch.setattr(cli, "build_bundle", lambda kind, sources, out_dir, name=None: calls.append(out_dir) or
                            BundleResult(Path("D:/x.zip"), 1, (), 10, "0" * 64))
        assert cmd_export_zip(build_parser().parse_args(["export-zip", "--kind", "inat", str(tmp_path)])) == 0
        assert calls == [Path("D:/bio-cache/downloads")]

    def test_env_override_wins_and_explicit_flag_wins_over_env(self, monkeypatch, tmp_path):
        calls = []
        monkeypatch.setattr(cli, "build_bundle", lambda kind, sources, out_dir, name=None: calls.append(out_dir) or
                            BundleResult(Path("D:/x.zip"), 1, (), 10, "0" * 64))
        monkeypatch.setenv("BIO_DOWNLOAD_DIR", str(tmp_path / "from_env"))
        cmd_export_zip(build_parser().parse_args(["export-zip", "--kind", "inat", str(tmp_path)]))
        cmd_export_zip(build_parser().parse_args(["export-zip", "--kind", "inat", str(tmp_path), "--out-dir", str(tmp_path / "flag")]))
        assert calls == [tmp_path / "from_env", tmp_path / "flag"]

    def test_success_prints_path_summary_and_the_scp_line(self, tmp_path, capsys):
        src = tmp_path / "set"
        src.mkdir()
        jpeg(src / "a.jpg"); jpeg(src / "b.jpg")
        (src / "observations.csv").write_bytes(b"x\n")
        args = build_parser().parse_args(["export-zip", "--kind", "inat", str(src), "--out-dir", str(tmp_path / "out"), "--name", "t"])
        assert cmd_export_zip(args) == 0
        out = capsys.readouterr().out.splitlines()
        zip_path = (tmp_path / "out" / "t.zip")
        assert out[0] == str(zip_path)
        assert out[1].startswith("  2 photos + 1 extra file(s) ['observations.csv'], ")
        assert out[-1] == f"    scp {getpass.getuser()}@{socket.gethostname()}:{zip_path.as_posix()} ."

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
    def test_empty_password_file_exits_1(self, tmp_path, capsys):
        pw = tmp_path / "pw.txt"
        pw.write_text("\n", encoding="utf-8")
        (tmp_path / "dl").mkdir()
        args = build_parser().parse_args(["serve-downloads", "--dir", str(tmp_path / "dl"), "--password-file", str(pw)])
        assert cmd_serve_downloads(args) == 1
        assert "is empty: refusing to serve with an empty password" in capsys.readouterr().err

    def test_missing_folder_exits_1(self, tmp_path, capsys):
        args = build_parser().parse_args(["serve-downloads", "--dir", str(tmp_path / "nope"), "--password-file", str(tmp_path / "pw.txt")])
        assert cmd_serve_downloads(args) == 1
        assert "download folder does not exist" in capsys.readouterr().err

    def test_bad_cidr_exits_1(self, tmp_path, capsys):
        (tmp_path / "dl").mkdir()
        args = build_parser().parse_args(["serve-downloads", "--dir", str(tmp_path / "dl"), "--password-file", str(tmp_path / "pw.txt"),
                                          "--allow", "300.1.1.1/8"])
        assert cmd_serve_downloads(args) == 1
        assert "does not appear to be an IPv4 or IPv6 network" in capsys.readouterr().err
