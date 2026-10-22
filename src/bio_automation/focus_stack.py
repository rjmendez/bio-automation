"""focus-stack CLI batch automation (focus stacking).

https://github.com/PetteriAimonen/focus-stack (MIT license). Unlike Zerene
Stacker (see zerene.py's module docstring for that engine's whole story),
focus-stack is a single self-contained command-line binary with no GUI, no
batch-script file, and no documented bug blocking it from actually merging
more than one frame -- it was adopted as the DEFAULT stacking engine
(2026-10-19) specifically because Zerene's own multi-file loader never gets
past the first frame on this machine (decompiled and confirmed in
zerene.py's docstring), so PMax never runs there. Zerene is kept reachable
as an explicit alternate (see pipeline.py's stack_focus() dispatch), not
deleted.

Real `--help` output captured 2026-10-19 from the installed build (no
`--help` flag exists; passing an unrecognized flag, or none, falls through
to printing this same usage block and exiting 1 -- `--version` and
`--opencv-version` are the only two flags that exit 0 on their own):

    Usage: focus-stack.exe [options] file1.jpg file2.jpg ...
      --output=output.jpg   Set output filename
      ...(see the real binary's own usage block for the rest)

Version measured on this machine: focus-stack 1.3-42-g0cd289e (OpenCV
4.12.0), release tag 1.5, MIT license.

Design decisions, reusing zerene.py's own hardening where it still applies
and deliberately dropping the parts that do not:

- Disk-space guard before launching (same min_free_gb pattern as
  zerene.run_batch -- the merged output lands in *folder* itself, so
  insufficient space there dooms the run before it starts) and
  subprocess timeout via communicate(timeout=...) (same kill-then-drain
  pattern on TimeoutExpired) are both reused as-is: they are about the
  subprocess/filesystem reality of running ANY external image-stacking
  binary, not anything specific to Zerene.
- The lock file (_acquire_lock/_release_lock in zerene.py) is NOT reused
  here. It exists in zerene.py because Zerene Stacker is a single
  GUI-app process per machine -- there is exactly one real Zerene to
  fight over. focus-stack is an ordinary, stateless CLI binary: each
  invocation is its own independent process with no shared GUI/singleton
  to serialize against, so two folders can legitimately be stacked by two
  concurrent focus-stack processes at once. Adding a lock here would
  serialize work that does not need to be serialized, for a hazard
  (overlapping launches of one shared Zerene GUI) that does not exist for
  this binary.
- Zerene's output is *discovered* after the fact by globbing for a
  template name, because OutputImageDispositionCode only lets Zerene's
  own batch script choose a template, not an exact path. focus-stack's
  `--output=<path>` flag lets this module *name* the exact output path up
  front instead, so there is no glob/race to get wrong -- the function
  just checks that the path it told the binary to write now exists.
- Zerene's launch command is built as a single pre-formed string because
  it must reproduce Windows' own CreateProcess parsing of a launch-command
  string Zerene itself wrote (zerenstk.launchcmd), which mixes quoted and
  unquoted "-Dkey=value" segments. There is no such opaque pre-existing
  string here -- this module builds its own argv from scratch -- so it is
  passed to subprocess.Popen as a plain list (shell=False, the default),
  which needs no hand-rolled quoting at all.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from . import settings

FOCUS_STACK_BINARY_ENV = settings.FOCUS_STACK_ENV

# Sensible default for this machine, overridable via the FOCUS_STACK_BINARY
# env var (checked at import time, same pattern as zerene.DEFAULT_LAUNCHCMD
# reading APPDATA) or by passing binary= explicitly to run_focus_stack().
DEFAULT_BINARY = Path(
    os.environ.get(
        FOCUS_STACK_BINARY_ENV,
        str(settings.FOCUS_STACK_DEFAULT),
    )
)

OUTPUT_NAME = "FS-OutputImage.tiff"


class FocusStackError(RuntimeError):
    """Raised when a focus-stack batch run fails, times out, or produces no output."""


def _discover_input_images(folder: Path, output_name: str) -> list[Path]:
    """Every regular file in *folder* except *output_name* itself.

    Excluding *output_name* mirrors zerene.py's own reason for excluding a
    leftover ZereneBatch.xml from the DA-CLIP gate in pipeline.py: this
    module writes its output into the same folder it reads frames from (see
    the module docstring), so a leftover output file from a prior run in
    the same folder must never be re-ingested as an input frame on the next
    call.
    """
    folder = Path(folder)
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.name != output_name
    )


def run_focus_stack(folder: Path, output_name: str = OUTPUT_NAME,
                     binary: Path | str | None = None,
                     timeout: float = 1800.0,
                     popen=subprocess.Popen,
                     min_free_gb: float = 5.0,
                     extra_args: list[str] | None = None) -> Path:
    """Run the real focus-stack binary against every frame in *folder* and
    return the merged output image path.

    *folder* is expected to already hold only the frames for one focus
    stack (already-denoised/aligned-ready, per this module's docstring) --
    same assumption zerene.run_batch() makes about its own *folder*
    argument. The output is written to ``folder / output_name`` (so the
    caller -- pipeline.stack_focus() -- can move it into work/ exactly the
    way it already does for Zerene's output).

    Raises FocusStackError if: *binary* does not exist; *folder* holds no
    input frames to stack; disk space at *folder* is below *min_free_gb*;
    the process exits non-zero; the process times out (killed first, same
    as zerene.run_batch); or the process exits 0 but *output_name* was not
    actually written.
    """
    folder = Path(folder)
    binary_path = Path(binary) if binary is not None else DEFAULT_BINARY

    if not binary_path.exists():
        raise FocusStackError(
            f"focus-stack binary not found at {binary_path} -- install it or set "
            f"{FOCUS_STACK_BINARY_ENV}, or pass binary= explicitly"
        )

    free_gb = shutil.disk_usage(folder).free / 1e9
    if free_gb < min_free_gb:
        raise FocusStackError(
            f"Insufficient disk space: {free_gb:.1f} GB free < {min_free_gb:.1f} GB required"
        )

    images = _discover_input_images(folder, output_name)
    if not images:
        raise FocusStackError(f"No input images found to stack in {folder}")

    output_path = folder / output_name
    cmd = [str(binary_path), *[str(p) for p in images], f"--output={output_path}"]
    if extra_args:
        cmd.extend(extra_args)

    proc = popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        output_text, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        raise FocusStackError(f"focus-stack timed out after {timeout}s on {folder}")

    if proc.returncode != 0:
        raise FocusStackError(f"focus-stack exited {proc.returncode} on {folder}:\n{output_text}")

    if not output_path.exists():
        raise FocusStackError(f"focus-stack finished but did not write {output_path}")

    return output_path
