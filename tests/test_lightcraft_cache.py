"""Unit tests for bio_automation.lightcraft_cache (hermetic -- no real
lightcraft-cli launched; subprocess.Popen faked, same pattern as
tests/test_focus_stack.py)."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from bio_automation import lightcraft_cache
from bio_automation.lightcraft_cache import (
    LIGHTCRAFT_CLI_ENV,
    LightCraftCliError,
    build_folder_cache,
    cache_path_for,
    render_preview_via_lightcraft,
    review_cache_dir,
)


class FakePopen:
    """Same recording-stub shape as tests/test_focus_stack.py's FakePopen:
    records construction and communicate/kill calls, never raises into
    caller code on its own."""

    instances: list["FakePopen"] = []

    def __init__(self, cmd, returncode=0, communicate_result=("", None),
                 communicate_side_effect=None, **kwargs):
        self.cmd = cmd
        self.kwargs = kwargs
        self.returncode = returncode
        self._communicate_result = communicate_result
        self._communicate_side_effect = communicate_side_effect
        self.communicate_calls: list[dict] = []
        self.kill_calls = 0
        FakePopen.instances.append(self)

    def communicate(self, timeout=None):
        self.communicate_calls.append({"timeout": timeout})
        if self._communicate_side_effect is not None and len(self.communicate_calls) == 1:
            raise self._communicate_side_effect
        return self._communicate_result

    def kill(self):
        self.kill_calls += 1


def _ok_lines(raw_path, dest_path):
    return "\n".join([
        json.dumps({"command": "develop.auto", "ok": True, "ms": 383.3,
                     "result": {"exposure": 0.5}}),
        json.dumps({"command": "app.export", "ok": True, "ms": 3096.9,
                     "result": {"bytes": 123, "path": str(dest_path)}}),
    ])


def _make_binary(tmp_path: Path) -> Path:
    binary = tmp_path / "bin" / "lightcraft-cli.exe"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_bytes(b"fake-exe")
    return binary


class TestExiftoolBinaryStyleResolution:
    """_lightcraft_cli_binary() mirrors exif._exiftool_binary()'s own
    resolution order -- same pattern, same test shape."""

    def test_env_override_wins(self, monkeypatch):
        monkeypatch.setenv(LIGHTCRAFT_CLI_ENV, r"D:\custom\lightcraft-cli.exe")
        monkeypatch.setattr(lightcraft_cache.shutil, "which", lambda name: r"C:\on\path\lightcraft-cli.exe")

        assert lightcraft_cache._lightcraft_cli_binary() == r"D:\custom\lightcraft-cli.exe"

    def test_falls_back_to_path(self, monkeypatch):
        monkeypatch.delenv(LIGHTCRAFT_CLI_ENV, raising=False)
        monkeypatch.setattr(lightcraft_cache.shutil, "which", lambda name: r"C:\on\path\lightcraft-cli.exe")

        assert lightcraft_cache._lightcraft_cli_binary() == r"C:\on\path\lightcraft-cli.exe"

    def test_raises_loudly_when_nothing_resolves(self, monkeypatch):
        monkeypatch.delenv(LIGHTCRAFT_CLI_ENV, raising=False)
        monkeypatch.setattr(lightcraft_cache.shutil, "which", lambda name: None)
        monkeypatch.setattr(lightcraft_cache.Path, "exists", lambda self: False)

        with pytest.raises(LightCraftCliError, match="lightcraft-cli not found"):
            lightcraft_cache._lightcraft_cli_binary()


class TestRenderPreviewViaLightcraft:
    def setup_method(self):
        FakePopen.instances = []

    def test_builds_exact_command_and_returns_dest_path(self, tmp_path):
        binary = _make_binary(tmp_path)
        raw_path = tmp_path / "DSC_0001.NEF"
        raw_path.write_bytes(b"raw")
        dest_path = tmp_path / "out" / "DSC_0001.jpg"

        def factory(cmd, **kwargs):
            fake = FakePopen(cmd, returncode=0, communicate_result=(_ok_lines(raw_path, dest_path), None))
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_path.write_bytes(b"jpeg")
            return fake

        import os
        os.environ[LIGHTCRAFT_CLI_ENV] = str(binary)
        try:
            result = render_preview_via_lightcraft(raw_path, dest_path, long_edge=800, quality=65, popen=factory)
        finally:
            del os.environ[LIGHTCRAFT_CLI_ENV]

        assert result == dest_path
        assert len(FakePopen.instances) == 1
        assert FakePopen.instances[0].cmd == [
            str(binary), "run", "--import", str(raw_path),
            "develop.auto",
            "app.export", f"path={dest_path}", "longEdge=800", "quality=65",
        ]

    def test_raises_on_a_failed_step_even_when_process_exits_zero(self, tmp_path, monkeypatch):
        """lightcraft-cli can exit 0 while one of its own JSON-lines steps
        reports ok: false (per its own --keep-going semantics) -- this must
        surface as a real error, not be read as success just because the
        process itself didn't crash."""
        binary = _make_binary(tmp_path)
        monkeypatch.setenv(LIGHTCRAFT_CLI_ENV, str(binary))
        raw_path = tmp_path / "DSC_0002.NEF"
        raw_path.write_bytes(b"raw")
        dest_path = tmp_path / "out.jpg"
        failing_line = json.dumps({"command": "app.export", "ok": False, "error": "disk full"})

        def factory(cmd, **kwargs):
            return FakePopen(cmd, returncode=0, communicate_result=(failing_line, None))

        with pytest.raises(LightCraftCliError, match=r"app\.export.*disk full"):
            render_preview_via_lightcraft(raw_path, dest_path, popen=factory)

    def test_raises_on_nonzero_exit(self, tmp_path, monkeypatch):
        binary = _make_binary(tmp_path)
        monkeypatch.setenv(LIGHTCRAFT_CLI_ENV, str(binary))
        raw_path = tmp_path / "DSC_0003.NEF"
        dest_path = tmp_path / "out.jpg"

        def factory(cmd, **kwargs):
            return FakePopen(cmd, returncode=1, communicate_result=("fatal error", None))

        with pytest.raises(LightCraftCliError, match=r"lightcraft-cli exited 1"):
            render_preview_via_lightcraft(raw_path, dest_path, popen=factory)

    def test_raises_when_success_reported_but_file_not_written(self, tmp_path, monkeypatch):
        """Positive twin to the normal-success test: ok:true steps with no
        actual output file must not be reported as success."""
        binary = _make_binary(tmp_path)
        monkeypatch.setenv(LIGHTCRAFT_CLI_ENV, str(binary))
        raw_path = tmp_path / "DSC_0004.NEF"
        dest_path = tmp_path / "never_written.jpg"

        def factory(cmd, **kwargs):
            return FakePopen(cmd, returncode=0, communicate_result=(_ok_lines(raw_path, dest_path), None))

        with pytest.raises(LightCraftCliError, match="did not write"):
            render_preview_via_lightcraft(raw_path, dest_path, popen=factory)

    def test_timeout_kills_the_process_and_raises(self, tmp_path, monkeypatch):
        binary = _make_binary(tmp_path)
        monkeypatch.setenv(LIGHTCRAFT_CLI_ENV, str(binary))
        raw_path = tmp_path / "DSC_0005.NEF"
        dest_path = tmp_path / "out.jpg"

        def factory(cmd, **kwargs):
            return FakePopen(cmd, communicate_side_effect=subprocess.TimeoutExpired(cmd, 1.0))

        with pytest.raises(LightCraftCliError, match="timed out"):
            render_preview_via_lightcraft(raw_path, dest_path, timeout=1.0, popen=factory)

        assert FakePopen.instances[0].kill_calls == 1


class TestCachePaths:
    def test_cache_path_for_uses_the_stem_under_work_review_cache(self, tmp_path):
        assert cache_path_for(tmp_path, "DSC_0001.NEF") == tmp_path / "work" / "review_cache" / "DSC_0001.jpg"

    def test_review_cache_dir_is_under_work(self, tmp_path):
        assert review_cache_dir(tmp_path) == tmp_path / "work" / "review_cache"


class TestBuildFolderCache:
    def test_renders_only_missing_files_in_parallel_and_reports_each_bucket(self, tmp_path):
        folder = tmp_path / "obs"
        (folder / "source" / "raw").mkdir(parents=True)
        for name in ("DSC_0001.NEF", "DSC_0002.NEF", "DSC_0003.NEF"):
            (folder / "source" / "raw" / name).write_bytes(b"raw")
        cache_dir = review_cache_dir(folder)
        cache_dir.mkdir(parents=True)
        (cache_dir / "DSC_0001.jpg").write_bytes(b"already-cached")  # pre-existing -> skipped

        calls = []

        def fake_render(raw_path, dest_path, *, long_edge, quality):
            calls.append(raw_path.name)
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_path.write_bytes(b"rendered")
            return dest_path

        result = build_folder_cache(
            folder, ["DSC_0001.NEF", "DSC_0002.NEF", "DSC_0003.NEF"],
            max_workers=2, render_fn=fake_render,
        )

        assert sorted(calls) == ["DSC_0002.NEF", "DSC_0003.NEF"]
        assert result["skipped"] == ["DSC_0001.NEF"]
        assert sorted(result["rendered"]) == ["DSC_0002.NEF", "DSC_0003.NEF"]
        assert result["failed"] == []
        assert (cache_dir / "DSC_0002.jpg").read_bytes() == b"rendered"

    def test_one_failure_does_not_abort_the_rest(self, tmp_path):
        """Positive twin to the all-succeed case: a bad raw's error is
        collected, every other raw in the batch still renders."""
        folder = tmp_path / "obs"
        (folder / "source" / "raw").mkdir(parents=True)
        for name in ("DSC_0001.NEF", "DSC_0002.NEF"):
            (folder / "source" / "raw" / name).write_bytes(b"raw")

        def fake_render(raw_path, dest_path, *, long_edge, quality):
            if raw_path.name == "DSC_0001.NEF":
                raise LightCraftCliError("DSC_0001.NEF: corrupt raw")
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_path.write_bytes(b"rendered")
            return dest_path

        result = build_folder_cache(
            folder, ["DSC_0001.NEF", "DSC_0002.NEF"], render_fn=fake_render,
        )

        assert result["failed"] == [{"file": "DSC_0001.NEF", "error": "DSC_0001.NEF: corrupt raw"}]
        assert result["rendered"] == ["DSC_0002.NEF"]

    def test_no_missing_files_renders_nothing(self, tmp_path):
        folder = tmp_path / "obs"
        cache_dir = review_cache_dir(folder)
        cache_dir.mkdir(parents=True)
        (cache_dir / "DSC_0001.jpg").write_bytes(b"cached")

        def fail_render(*a, **k):
            pytest.fail("must not be called when every file is already cached")

        result = build_folder_cache(folder, ["DSC_0001.NEF"], render_fn=fail_render)

        assert result == {"rendered": [], "skipped": ["DSC_0001.NEF"], "failed": []}
