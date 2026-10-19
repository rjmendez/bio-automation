"""Zerene Stacker headless batch automation (focus stacking).

Zerene Stacker has no public API or MCP server, but it does have a
documented Batch API (https://www.zerenesystems.com/cms/stacker/docs/batchapi):
launch the bundled JRE directly -- not zerenstk.exe, which returns before
stacking finishes and leaves the GUI running -- with
-noSplashScreen -exitOnBatchScriptCompletion -runMinimized and the source
image folder as the final argument. The batch script is implicitly
ZereneBatch.xml, read from that same folder.

The exact launch command (javaw.exe path, -Xmx sizing, classpath,
-Dlaunchcmddir) is machine- and install-specific, so it is not
reconstructed here: Zerene writes it to its own preferences dir on every
normal GUI launch, as zerenstk.launchcmd. Confirmed on this machine at
%APPDATA%\\ZereneStacker\\zerenstk.launchcmd (captured 2026-10-18 via a
one-time GUI launch, per the docs' own instructions for Windows/Linux/macOS).

Only TaskIndicatorCode 1 (Align and Stack All / PMax) is used: it is the
only stacking-method code the docs publish. DMap's code is undocumented.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from xml.sax.saxutils import escape

DEFAULT_LAUNCHCMD = Path(os.environ.get("APPDATA", "")) / "ZereneStacker" / "zerenstk.launchcmd"
OUTPUT_NAME_TEMPLATE = "ZS-OutputImage"

# Lock file guarding against two overlapping run_batch() calls both launching
# Zerene Stacker at once (it is a single GUI-app process on this machine, not
# something that can run two headless batches in parallel). Co-located with
# zerenstk.launchcmd -- Zerene's own preferences dir -- rather than inside the
# per-call source *folder* (there can be many of those, created/removed per
# batch) or the repo (irrelevant to which machine's Zerene install is actually
# running). One lock path per machine matches the one-Zerene-GUI-per-machine
# reality this module already assumes for DEFAULT_LAUNCHCMD.
DEFAULT_LOCK_PATH = DEFAULT_LAUNCHCMD.parent / "bio_automation_zerene_batch.lock"

METHOD_TASK_CODES = {
    "pmax": 1,  # Align and Stack All (PMax) -- the only documented code
}


class ZereneError(RuntimeError):
    """Raised when a Zerene batch run fails, times out, or produces no output."""


def locate_launch_command(launchcmd_path: Path | None = None) -> str:
    """Read Zerene's self-written launch command line.

    Raises FileNotFoundError if Zerene Stacker has never been launched
    normally (via its GUI) on this machine -- that one-time launch is how
    the docs say to learn the install's classpath/launchcmddir/heap size;
    there is no other documented way to obtain it.
    """
    path = launchcmd_path if launchcmd_path is not None else DEFAULT_LAUNCHCMD
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- launch Zerene Stacker's GUI once first so it "
            "writes its own launch command (see Zerene's Batch API docs)."
        )
    return path.read_text(encoding="utf-8").strip()


def write_batch_script(folder: Path, method: str = "pmax",
                        output_name: str = OUTPUT_NAME_TEMPLATE) -> Path:
    """Write ZereneBatch.xml into *folder* and return its path.

    ProjectDispositionCode 101 = new project is temporary (we only want the
    output image, not a saved .zpr). OutputImageDispositionCode 2 = save in
    the source folder.
    """
    if method not in METHOD_TASK_CODES:
        raise ValueError(f"Unknown method {method!r}; choose from {sorted(METHOD_TASK_CODES)}")
    task_code = METHOD_TASK_CODES[method]
    xml = (
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
        f'            <TaskIndicatorCode value="{task_code}" />\n'
        '            <OutputImageDispositionCode value="2" />\n'
        "            <Preferences>\n"
        f'              <OutputImageNaming.Template value="{escape(output_name)}" />\n'
        "            </Preferences>\n"
        "          </Task>\n"
        "        </Tasks>\n"
        "      </Batch>\n"
        "    </Batches>\n"
        "  </BatchQueue>\n"
        "</ZereneStackerBatchScript>\n"
    )
    script_path = Path(folder) / "ZereneBatch.xml"
    script_path.write_text(xml, encoding="utf-8")
    return script_path


def find_output(folder: Path, output_name: str = OUTPUT_NAME_TEMPLATE) -> Path | None:
    """Return the most recently modified file matching ``output_name.*`` in *folder*.

    None if no match. "Most recent" because a folder can carry leftover
    output from a prior run; the one just produced is always the newest.
    """
    matches = sorted(Path(folder).glob(f"{output_name}.*"), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def _acquire_lock(lock_path: Path) -> None:
    """Atomically create *lock_path*, writing this process's PID into it.

    Uses os.open with O_CREAT|O_EXCL: the filesystem performs the
    check-and-create as one atomic operation, so two callers racing to
    acquire the lock cannot both win -- unlike a separate ``exists()`` check
    followed by a ``write()``, where both callers could pass the check
    before either creates the file.

    Raises ZereneError (not FileExistsError) if the lock is already held,
    naming the PID found in the existing lock file where available, so a
    human or a future caller can tell what is holding it.

    Stale-lock recovery is deliberately NOT implemented in this pass: there
    is no check here for "the PID in an existing lock file no longer
    refers to a running process". A Zerene batch run is a long-lived
    desktop-GUI launch (see the module docstring -- zerenstk.exe itself
    returns early, which is exactly the kind of premature-exit confusion
    this module already works around); if a run_batch() call dies without
    reaching its `finally` (e.g. the whole Python process is killed), the
    safest default is to make the next caller fail loudly and require a
    human to confirm via the PID in the lock file / Task Manager that
    nothing is actually running before deleting it by hand. An automatic
    liveness check (e.g. psutil.pid_exists) risks a reclaimed-but-wrong
    lock if that PID has since been reused by an unrelated process, which
    would let two Zerene instances fight over the same output folder --
    worse than the manual-cleanup inconvenience being traded away here.
    """
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        held_by = "?"
        try:
            held_by = Path(lock_path).read_text(encoding="utf-8").strip() or "?"
        except OSError:
            pass
        raise ZereneError(
            f"Zerene stacking is already in progress (lock {lock_path} held by pid {held_by})"
        )
    try:
        os.write(fd, str(os.getpid()).encode("ascii"))
    finally:
        os.close(fd)


def _release_lock(lock_path: Path) -> None:
    """Remove *lock_path*. Swallows a missing file -- release must never be
    the thing that masks the real success/failure/timeout outcome of the
    run it is cleaning up after."""
    try:
        os.remove(lock_path)
    except OSError:
        pass


def run_batch(folder: Path, method: str = "pmax",
              output_name: str = OUTPUT_NAME_TEMPLATE,
              launch_command: str | None = None,
              timeout: float = 1800.0,
              popen=subprocess.Popen,
              min_free_gb: float = 5.0,
              lock_path: Path | None = None) -> Path:
    """Run Zerene Stacker headless against *folder* and return the output image path.

    Writes ZereneBatch.xml, launches Zerene via the captured launch command
    plus -noSplashScreen -exitOnBatchScriptCompletion -runMinimized "<folder>",
    passed as a single command-line string (not a pre-split arg list) so
    Windows' own CreateProcess parses it exactly as it parsed the string
    Zerene wrote for itself -- avoiding a hand-rolled Windows quoting parser
    for a line that mixes quoted and unquoted "-Dkey=value" segments.

    Disk space at *folder* is checked before the subprocess is launched --
    OutputImageDispositionCode=2 (see write_batch_script) has Zerene write
    its merged output back into that same folder, so insufficient space
    there dooms the run before it starts.

    The docs warn the caller must drain stdout/stderr while waiting or both
    processes can hang; ``communicate()`` does this. Raises ZereneError on a
    non-zero exit, a timeout (the process is killed first), insufficient
    disk space, or a missing output file after a clean exit.

    Before any of that, a lock file at *lock_path* (default
    ``DEFAULT_LOCK_PATH``) is created atomically to guard against a second,
    overlapping run_batch() call also launching Zerene -- it is a single
    GUI-app process, so two headless batches cannot run at once. If the
    lock is already held, raises ZereneError immediately without writing
    the batch script or calling *popen* at all (no queueing, no blocking).
    The lock is always released in a ``finally``, whether this call
    succeeds, Zerene fails, or the run times out.
    """
    folder = Path(folder)
    lock_path = lock_path if lock_path is not None else DEFAULT_LOCK_PATH

    _acquire_lock(lock_path)
    try:
        free_gb = shutil.disk_usage(folder).free / 1e9
        if free_gb < min_free_gb:
            raise ZereneError(
                f"Insufficient disk space: {free_gb:.1f} GB free < {min_free_gb:.1f} GB required"
            )

        write_batch_script(folder, method=method, output_name=output_name)

        base = launch_command if launch_command is not None else locate_launch_command()
        full_cmd = f'{base} -noSplashScreen -exitOnBatchScriptCompletion -runMinimized "{folder}"'

        proc = popen(full_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            output_text, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            raise ZereneError(f"Zerene batch timed out after {timeout}s on {folder}")

        if proc.returncode != 0:
            raise ZereneError(f"Zerene batch exited {proc.returncode} on {folder}:\n{output_text}")

        output_path = find_output(folder, output_name)
        if output_path is None:
            raise ZereneError(f"Zerene batch finished but produced no {output_name}.* in {folder}")
        return output_path
    finally:
        _release_lock(lock_path)
