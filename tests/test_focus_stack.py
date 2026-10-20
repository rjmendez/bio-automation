"""Unit tests for bio_automation.focus_stack (hermetic -- no real focus-stack
binary launched; see tests/test_focus_stack_live.py for the real, unmocked run)."""
from __future__ import annotations

import importlib
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from bio_automation import focus_stack
from bio_automation.focus_stack import (
    FOCUS_STACK_BINARY_ENV,
    FocusStackError,
    OUTPUT_NAME,
    run_focus_stack,
)


class FakePopen:
    """Records its construction and communicate/kill calls; never raises into
    caller code (rule 4: a stub that must not be called records its calls)."""

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


def _make_popen_factory(**popen_kwargs):
    def factory(cmd, **kwargs):
        return FakePopen(cmd, **popen_kwargs)
    return factory


def _make_binary(tmp_path: Path) -> Path:
    binary = tmp_path / "bin" / "focus-stack.exe"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_bytes(b"fake-exe")
    return binary


def _make_frames(folder: Path, *names: str) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in names:
        p = folder / name
        p.write_bytes(name.encode("ascii"))
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# run_focus_stack -- success paths
# ---------------------------------------------------------------------------

class TestRunFocusStackSuccess:
    def setup_method(self):
        FakePopen.instances = []

    def test_builds_exact_command_with_sorted_images_and_output_flag(self, tmp_path):
        binary = _make_binary(tmp_path)
        frames_dir = tmp_path / "frames"
        _make_frames(frames_dir, "b.tif", "a.tif")  # written out of order
        output_path = frames_dir / OUTPUT_NAME

        def factory(cmd, **kwargs):
            fake = FakePopen(cmd, returncode=0, **kwargs)
            output_path.write_bytes(b"stacked")
            return fake

        result = run_focus_stack(frames_dir, binary=binary, popen=factory)

        assert len(FakePopen.instances) == 1
        proc = FakePopen.instances[0]
        assert proc.cmd == [
            str(binary),
            str(frames_dir / "a.tif"),
            str(frames_dir / "b.tif"),
            f"--output={output_path}",
        ]
        assert proc.kwargs["stdout"] == subprocess.PIPE
        assert proc.kwargs["stderr"] == subprocess.STDOUT
        assert proc.kwargs["text"] is True
        assert result == output_path

    def test_excludes_leftover_output_file_from_a_prior_run_from_input_images(self, tmp_path):
        """A leftover OUTPUT_NAME file from a crashed/old run in the same
        folder must not be fed back in as an input frame on the next call --
        positive twin to the exact-command test above, same fixture shape,
        with a 3rd file that must be excluded rather than included."""
        binary = _make_binary(tmp_path)
        frames_dir = tmp_path / "frames"
        _make_frames(frames_dir, "a.tif", "b.tif")
        leftover = frames_dir / OUTPUT_NAME
        leftover.write_bytes(b"stale-output-from-a-prior-run")

        def factory(cmd, **kwargs):
            fake = FakePopen(cmd, returncode=0, **kwargs)
            leftover.write_bytes(b"freshly-stacked")
            return fake

        run_focus_stack(frames_dir, binary=binary, popen=factory)

        proc = FakePopen.instances[0]
        assert str(leftover) not in proc.cmd[:-1]  # never passed as an input frame
        assert proc.cmd == [
            str(binary),
            str(frames_dir / "a.tif"),
            str(frames_dir / "b.tif"),
            f"--output={leftover}",
        ]

    def test_extra_args_are_appended_after_the_output_flag(self, tmp_path):
        binary = _make_binary(tmp_path)
        frames_dir = tmp_path / "frames"
        _make_frames(frames_dir, "a.tif")
        output_path = frames_dir / OUTPUT_NAME

        def factory(cmd, **kwargs):
            fake = FakePopen(cmd, returncode=0, **kwargs)
            output_path.write_bytes(b"stacked")
            return fake

        run_focus_stack(frames_dir, binary=binary, popen=factory,
                         extra_args=["--no-opencl", "--consistency=0"])

        proc = FakePopen.instances[0]
        assert proc.cmd == [
            str(binary),
            str(frames_dir / "a.tif"),
            f"--output={output_path}",
            "--no-opencl",
            "--consistency=0",
        ]


# ---------------------------------------------------------------------------
# run_focus_stack -- error paths, each with a positive twin above/below
# ---------------------------------------------------------------------------

class TestRunFocusStackErrors:
    def setup_method(self):
        FakePopen.instances = []

    def test_missing_binary_raises_without_calling_popen(self, tmp_path):
        """Negative twin to test_builds_exact_command...: a binary path that
        does not exist must fail loudly before any subprocess is launched."""
        frames_dir = tmp_path / "frames"
        _make_frames(frames_dir, "a.tif", "b.tif")
        missing_binary = tmp_path / "does-not-exist.exe"
        recording_popen = _make_popen_factory()

        with pytest.raises(FocusStackError, match=r"binary not found at .*does-not-exist\.exe"):
            run_focus_stack(frames_dir, binary=missing_binary, popen=recording_popen)

        assert FakePopen.instances == []

    def test_no_input_images_raises_without_calling_popen(self, tmp_path):
        """Negative twin: an empty source folder must fail loudly instead of
        launching focus-stack with zero file arguments."""
        binary = _make_binary(tmp_path)
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()

        with pytest.raises(FocusStackError, match=r"No input images found to stack in"):
            run_focus_stack(frames_dir, binary=binary, popen=_make_popen_factory())

        assert FakePopen.instances == []

    def test_insufficient_disk_space_raises_before_popen(self, tmp_path, monkeypatch):
        binary = _make_binary(tmp_path)
        frames_dir = tmp_path / "frames"
        _make_frames(frames_dir, "a.tif", "b.tif")
        monkeypatch.setattr(focus_stack.shutil, "disk_usage",
                             lambda _: MagicMock(free=int(0.5 * 1e9)))  # 0.5 GB free

        with pytest.raises(FocusStackError, match=r"(?i)insufficient disk space"):
            run_focus_stack(frames_dir, binary=binary, popen=_make_popen_factory(),
                             min_free_gb=5.0)

        assert FakePopen.instances == []

    def test_sufficient_disk_space_proceeds_and_launches_subprocess(self, tmp_path, monkeypatch):
        """Positive twin to the disk-guard test above, same fixture."""
        binary = _make_binary(tmp_path)
        frames_dir = tmp_path / "frames"
        _make_frames(frames_dir, "a.tif", "b.tif")
        output_path = frames_dir / OUTPUT_NAME
        monkeypatch.setattr(focus_stack.shutil, "disk_usage",
                             lambda _: MagicMock(free=int(10 * 1e9)))  # 10 GB free

        def factory(cmd, **kwargs):
            fake = FakePopen(cmd, returncode=0, **kwargs)
            output_path.write_bytes(b"stacked")
            return fake

        result = run_focus_stack(frames_dir, binary=binary, popen=factory, min_free_gb=5.0)

        assert len(FakePopen.instances) == 1
        assert result == output_path

    def test_nonzero_exit_raises_and_does_not_return(self, tmp_path):
        binary = _make_binary(tmp_path)
        frames_dir = tmp_path / "frames"
        _make_frames(frames_dir, "a.tif", "b.tif")
        popen = _make_popen_factory(returncode=1, communicate_result=("boom", None))

        with pytest.raises(FocusStackError, match=r"focus-stack exited 1 on"):
            run_focus_stack(frames_dir, binary=binary, popen=popen)

    def test_timeout_kills_process_and_raises(self, tmp_path):
        binary = _make_binary(tmp_path)
        frames_dir = tmp_path / "frames"
        _make_frames(frames_dir, "a.tif", "b.tif")
        popen = _make_popen_factory(
            communicate_side_effect=subprocess.TimeoutExpired(cmd="focus-stack", timeout=5)
        )

        with pytest.raises(FocusStackError, match=r"timed out after 5s"):
            run_focus_stack(frames_dir, binary=binary, popen=popen, timeout=5)

        assert len(FakePopen.instances) == 1
        assert FakePopen.instances[0].kill_calls == 1
        assert len(FakePopen.instances[0].communicate_calls) == 2  # raised, then drained after kill

    def test_clean_exit_with_no_output_file_raises(self, tmp_path):
        binary = _make_binary(tmp_path)
        frames_dir = tmp_path / "frames"
        _make_frames(frames_dir, "a.tif", "b.tif")
        popen = _make_popen_factory(returncode=0, communicate_result=("done", None))

        with pytest.raises(FocusStackError, match=rf"did not write .*{OUTPUT_NAME}"):
            run_focus_stack(frames_dir, binary=binary, popen=popen)


# ---------------------------------------------------------------------------
# DEFAULT_BINARY -- env var override vs. the module-constant fallback
# ---------------------------------------------------------------------------

class TestDefaultBinaryEnvVar:
    def test_env_var_overrides_the_module_constant_default(self, monkeypatch):
        monkeypatch.setenv(FOCUS_STACK_BINARY_ENV, r"C:\custom\focus-stack.exe")
        try:
            importlib.reload(focus_stack)
            assert focus_stack.DEFAULT_BINARY == Path(r"C:\custom\focus-stack.exe")
        finally:
            monkeypatch.delenv(FOCUS_STACK_BINARY_ENV, raising=False)
            importlib.reload(focus_stack)

    def test_no_env_var_falls_back_to_the_module_constant_default(self, monkeypatch):
        """Positive twin to the override test above, on the same reload
        mechanism: with the env var absent, the documented, real install
        path for this machine is used."""
        monkeypatch.delenv(FOCUS_STACK_BINARY_ENV, raising=False)
        importlib.reload(focus_stack)
        assert focus_stack.DEFAULT_BINARY == Path(
            r"C:\Users\rjmendez-admin\AppData\Local\Programs\focus-stack\focus-stack\focus-stack.exe"
        )
