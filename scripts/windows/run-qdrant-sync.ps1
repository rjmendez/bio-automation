# The Qdrant freshness task (Bio-Qdrant-Sync): `bio qdrant-sync`, then `bio qdrant-inat-sync`, daily 05:00.
# Both syncs embed gallery images with DA-CLIP on the GPU (only files not yet in the vector cache), so this must never overlap GPU-heavy work:
#   * the named mutex Global\bio-social-run (held by Bio-Social-Prep, 03:00, up to 2 h, and by every post run) is taken for the whole run; a run
#     that cannot get it within 90 minutes gives up with exit 1 instead of competing for the GPU;
#   * the named mutex Global\bio-qdrant-sync keeps two copies of this script (a manual run plus the task) from running together.
#   * the NAS pipeline's GPU lane is not a mutex holder; 05:00 is after its nightly window and after prep (03:00 + 2 h limit). Check
#     nvidia-smi if you move the trigger.
# Exit code: 0 both syncs finished; 1 when a sync failed or the script could not run (no package, mutex wait expired). A failed sync does not
# stop the other one. On failure an alert is sent through BIO_SOCIAL_ALERT_COMMAND (else notify-to-inbox.ps1 next to this script), and the python
# side has audited qdrant_sync_failed, which social-alerts turns into a warning. A success writes the sync marker that social-alerts and
# social-selfcheck read to warn when an index is older than BIO_QDRANT_STALE_DAYS (default 3).
# Runs the pinned production package (D:\apps\bio-automation), never a development checkout.
$ErrorActionPreference = 'Stop'
$log = 'D:\bio-cache\logs\social.log'
$utf8 = New-Object System.Text.UTF8Encoding($false)
function Log([string]$msg) {
    $line = "$(Get-Date -Format s) [qdrant] $msg`n"
    try { [IO.File]::AppendAllText($log, $line, $utf8) }
    catch { try { [IO.File]::AppendAllText((Join-Path $env:TEMP 'bio-social-fallback.log'), $line, $utf8) } catch { } }
}
$mutexes = @()
$code = 1
$failed = @()
try {
    try { New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null } catch { }
    Log "run start (pid $PID, user $env:USERNAME)"
    $py = 'C:\Python314\python.exe'
    $deploySrc = if ($env:BIO_PACKAGE_DEPLOY) { $env:BIO_PACKAGE_DEPLOY } else { 'D:\apps\bio-automation\src' }
    if (-not (Test-Path (Join-Path $deploySrc 'bio_automation\__init__.py'))) { Log "no bio_automation package under $deploySrc; nothing synced"; throw 'no package' }
    $env:PYTHONPATH = $deploySrc
    [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
    $env:PYTHONIOENCODING = 'utf-8'

    foreach ($spec in @(@('Global\bio-qdrant-sync', 1), @('Global\bio-social-run', 90))) {
        $m = New-Object System.Threading.Mutex($false, $spec[0])
        $got = $false
        try { $got = $m.WaitOne([TimeSpan]::FromMinutes($spec[1])) }
        catch [System.Threading.AbandonedMutexException] { $got = $true; Log "the previous holder of $($spec[0]) died; taking the mutex" }
        if (-not $got) { $m.Dispose(); Log "gave up: $($spec[0]) was still held after $($spec[1]) minute(s) (GPU work or another sync in progress); nothing synced"; throw 'mutex timeout' }
        $mutexes += $m
    }

    $ErrorActionPreference = 'Continue'          # python writing to stderr is not a PowerShell error; the exit code decides
    foreach ($cmd in 'qdrant-sync', 'qdrant-inat-sync') {
        Log "bio $cmd"
        & $py -m bio_automation.cli $cmd 2>&1 | ForEach-Object { [IO.File]::AppendAllText($log, "$_`n", $utf8) }
        $c = $LASTEXITCODE
        if ($c -ne 0) { Log "bio $cmd exited $c"; $failed += "$cmd (exit $c)" } else { Log "bio $cmd finished" }
    }
    $ErrorActionPreference = 'Stop'
    $code = if ($failed.Count -eq 0) { 0 } else { 1 }
    if ($code -eq 0) { Log 'qdrant sync finished' }
}
catch {
    Log "FAILED before or during the run: $($_.Exception.Message) (line $($_.InvocationInfo.ScriptLineNumber))"
    $failed += "script: $($_.Exception.Message)"
    $code = 1
}
finally {
    foreach ($m in $mutexes) { try { $m.ReleaseMutex() } catch { }; $m.Dispose() }
}
if ($code -ne 0) {
    # alert on failure: the message goes to the alert command on stdin as UTF-8, like social-alerts does
    try {
        $cmdline = $env:BIO_SOCIAL_ALERT_COMMAND
        if (-not $cmdline) {
            $notifier = Join-Path $PSScriptRoot 'notify-to-inbox.ps1'
            $cmdline = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe -NoProfile -ExecutionPolicy Bypass -File $notifier"
        }
        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName = "$env:SystemRoot\System32\cmd.exe"          # the command line is run as written, like social-alerts does
        $psi.Arguments = "/d /s /c ""$cmdline"""
        $psi.RedirectStandardInput = $true
        $psi.UseShellExecute = $false
        $psi.CreateNoWindow = $true
        $p = [System.Diagnostics.Process]::Start($psi)
        $bytes = $utf8.GetBytes("qdrant_sync_failed: $($failed -join '; ') ($(Get-Date -Format s)); see $log")
        $p.StandardInput.BaseStream.Write($bytes, 0, $bytes.Length)
        $p.StandardInput.Close()
        if (-not $p.WaitForExit(30000)) { try { $p.Kill() } catch { }; Log 'alert command timed out' } elseif ($p.ExitCode -ne 0) { Log "alert command exited $($p.ExitCode)" }
    }
    catch { Log "could not send the alert: $($_.Exception.Message)" }
}
exit $code
