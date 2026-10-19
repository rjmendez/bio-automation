"""Unit tests for bio_automation.zerene (hermetic -- no real Zerene Stacker launched)."""
from __future__ import annotations

import subprocess
from unittest.mock import MagicMock

import pytest

from bio_automation import zerene
from bio_automation.zerene import (
    ZereneError,
    find_output,
    locate_launch_command,
    run_batch,
    write_batch_script,
)

FAKE_LAUNCH_CMD = (
    '"C:\\Program Files\\ZereneStacker\\jre\\bin\\javaw.exe" '
    '-Dsun.java2d.opengl=false -Xmx32723m '
    '-XX:ErrorFile="C:\\Users\\test\\AppData\\Roaming\\ZereneStacker\\hs_err_pid%p.log" '
    '-DjavaBits=64bitJava '
    '-Dlaunchcmddir="C:\\Users\\test\\AppData\\Roaming\\ZereneStacker" '
    '-classpath "C:\\Program Files\\ZereneStacker\\ZereneStacker.jar;'
    'C:\\Program Files\\ZereneStacker\\JREextensions\\*" '
    "com.zerenesystems.stacker.gui.MainFrame"
)


class FakePopen:
    """Records its construction and communicate/kill calls; never raises into caller code."""

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


# ---------------------------------------------------------------------------
# write_batch_script
# ---------------------------------------------------------------------------

class TestWriteBatchScript:
    def test_pmax_writes_exact_xml(self, tmp_path):
        path = write_batch_script(tmp_path, method="pmax", output_name="ZS-OutputImage")
        assert path == tmp_path / "ZereneBatch.xml"
        assert path.read_text(encoding="utf-8") == (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            "<ZereneStackerBatchScript>\n"
            "  <BatchQueue>\n"
            "    <Batches>\n"
            "      <Batch>\n"
            "        <Sources>\n"
            '          <Source value="%CurrentProject%" />\n'
            "        </Sources>\n"
            '        <ProjectDispositionCode value="101" />\n'
            "        <Tasks>\n"
            "          <Task>\n"
            '            <TaskIndicatorCode value="1" />\n'
            '            <OutputImageDispositionCode value="2" />\n'
            "            <Preferences>\n"
            '              <OutputImageNaming.Template value="ZS-OutputImage" />\n'
            "            </Preferences>\n"
            "          </Task>\n"
            "        </Tasks>\n"
            "      </Batch>\n"
            "    </Batches>\n"
            "  </BatchQueue>\n"
            "</ZereneStackerBatchScript>\n"
        )

    def test_unknown_method_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown method 'dmap'"):
            write_batch_script(tmp_path, method="dmap")
        assert not (tmp_path / "ZereneBatch.xml").exists()

    def test_output_name_is_escaped(self, tmp_path):
        path = write_batch_script(tmp_path, output_name="A & B")
        assert "<OutputImageNaming.Template value=\"A &amp; B\" />" in path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# locate_launch_command
# ---------------------------------------------------------------------------

class TestLocateLaunchCommand:
    def test_missing_file_raises(self, tmp_path):
        missing = tmp_path / "zerenstk.launchcmd"
        with pytest.raises(FileNotFoundError, match="launch Zerene Stacker's GUI once"):
            locate_launch_command(missing)

    def test_reads_and_strips_existing_file(self, tmp_path):
        launchcmd = tmp_path / "zerenstk.launchcmd"
        launchcmd.write_text("  " + FAKE_LAUNCH_CMD + "\n", encoding="utf-8")
        assert locate_launch_command(launchcmd) == FAKE_LAUNCH_CMD


# ---------------------------------------------------------------------------
# find_output
# ---------------------------------------------------------------------------

class TestFindOutput:
    def test_no_match_returns_none(self, tmp_path):
        (tmp_path / "unrelated.jpg").write_bytes(b"x")
        assert find_output(tmp_path, "ZS-OutputImage") is None

    def test_single_match_returned(self, tmp_path):
        target = tmp_path / "ZS-OutputImage.tif"
        target.write_bytes(b"x")
        assert find_output(tmp_path, "ZS-OutputImage") == target

    def test_most_recent_of_multiple_matches_returned(self, tmp_path):
        import time

        older = tmp_path / "ZS-OutputImage.tif"
        older.write_bytes(b"old")
        time.sleep(0.01)
        newer = tmp_path / "ZS-OutputImage.jpg"
        newer.write_bytes(b"new")
        assert find_output(tmp_path, "ZS-OutputImage") == newer


# ---------------------------------------------------------------------------
# run_batch
# ---------------------------------------------------------------------------

class TestRunBatch:
    def setup_method(self):
        FakePopen.instances = []

    def test_success_writes_script_builds_exact_command_and_returns_output(self, tmp_path):
        output = tmp_path / "ZS-OutputImage.tif"

        def factory(cmd, **kwargs):
            fake = FakePopen(cmd, returncode=0, **kwargs)
            output.write_bytes(b"stacked")  # simulate Zerene producing the file
            return fake

        result = run_batch(tmp_path, launch_command=FAKE_LAUNCH_CMD, popen=factory)

        assert (tmp_path / "ZereneBatch.xml").exists()
        assert len(FakePopen.instances) == 1
        proc = FakePopen.instances[0]
        assert proc.cmd == (
            FAKE_LAUNCH_CMD + " -noSplashScreen -exitOnBatchScriptCompletion "
            f'-runMinimized "{tmp_path}"'
        )
        assert proc.kwargs["stdout"] == subprocess.PIPE
        assert proc.kwargs["stderr"] == subprocess.STDOUT
        assert proc.kwargs["text"] is True
        assert result == output

    def test_nonzero_exit_raises_and_does_not_return(self, tmp_path):
        (tmp_path / "ZS-OutputImage.tif").write_bytes(b"stale")
        popen = _make_popen_factory(returncode=1, communicate_result=("boom", None))
        with pytest.raises(ZereneError, match=r"Zerene batch exited 1 on"):
            run_batch(tmp_path, launch_command=FAKE_LAUNCH_CMD, popen=popen)

    def test_timeout_kills_process_and_raises(self, tmp_path):
        popen = _make_popen_factory(
            communicate_side_effect=subprocess.TimeoutExpired(cmd="zerene", timeout=5)
        )
        with pytest.raises(ZereneError, match=r"timed out after 5s"):
            run_batch(tmp_path, launch_command=FAKE_LAUNCH_CMD, popen=popen, timeout=5)
        assert len(FakePopen.instances) == 1
        assert FakePopen.instances[0].kill_calls == 1
        assert len(FakePopen.instances[0].communicate_calls) == 2  # raised, then drained after kill

    def test_clean_exit_with_no_output_file_raises(self, tmp_path):
        popen = _make_popen_factory(returncode=0, communicate_result=("done", None))
        with pytest.raises(ZereneError, match=r"produced no ZS-OutputImage\.\* in"):
            run_batch(tmp_path, launch_command=FAKE_LAUNCH_CMD, popen=popen)


# ---------------------------------------------------------------------------
# run_batch -- disk-space guard. Zerene's OutputImageDispositionCode=2 (see
# write_batch_script) writes the merged output back into the source folder,
# so insufficient space there must abort before the batch script is written
# or the subprocess is launched.
# ---------------------------------------------------------------------------

class TestRunBatchDiskGuard:
    def setup_method(self):
        FakePopen.instances = []

    def test_insufficient_space_raises_before_script_or_subprocess(self, tmp_path, monkeypatch):
        monkeypatch.setattr(zerene.shutil, "disk_usage",
                             lambda _: MagicMock(free=int(0.5 * 1e9)))  # 0.5 GB free

        with pytest.raises(ZereneError, match=r"(?i)insufficient disk space"):
            run_batch(tmp_path, launch_command=FAKE_LAUNCH_CMD,
                      popen=_make_popen_factory(), min_free_gb=5.0)

        assert not (tmp_path / "ZereneBatch.xml").exists()
        assert FakePopen.instances == []

    def test_sufficient_space_proceeds_and_launches_subprocess(self, tmp_path, monkeypatch):
        monkeypatch.setattr(zerene.shutil, "disk_usage",
                             lambda _: MagicMock(free=int(10 * 1e9)))  # 10 GB free
        output = tmp_path / "ZS-OutputImage.tif"

        def factory(cmd, **kwargs):
            fake = FakePopen(cmd, returncode=0, **kwargs)
            output.write_bytes(b"stacked")
            return fake

        result = run_batch(tmp_path, launch_command=FAKE_LAUNCH_CMD, popen=factory,
                            min_free_gb=5.0)

        assert (tmp_path / "ZereneBatch.xml").exists()
        assert len(FakePopen.instances) == 1
        assert result == output
