# Daily preparation for the Bluesky posts (bio social-prep): disk preflight, social-index (incremental), darkroom-sync, social-draft.
# Scheduled task Bio-Social-Prep, daily 03:00. Posting runs separately (run-social-post.ps1) and never waits for this beyond 10 minutes; a
# failed prep only means fewer new drafts, and this script exits with the social-prep exit code (1 when it could not run) so the task shows it.
# Mutual exclusion: the named Windows mutex Global\bio-social-run is held for the whole prep. A prep started late (StartWhenAvailable after a
# missed 03:00) therefore waits up to 30 minutes for a post run that is in progress, and a post run waits up to 10 minutes for the prep.
# Runs the pinned production package (D:\apps\bio-automation, the tree the NAS pipeline uses), never a development checkout.
$ErrorActionPreference = 'Stop'
$log = 'D:\bio-cache\logs\social.log'
# Windows PowerShell 5.1 writes `>>` as UTF-16 and decodes native output with the OEM code page: capture python's output as UTF-8
# and append it to the log with a BOM-less UTF-8 encoding.
$utf8 = New-Object System.Text.UTF8Encoding($false)
# Log never throws: when social.log cannot be written the line goes to a fallback file in %TEMP%, so a failure is never completely silent.
function Log([string]$msg) {
    $line = "$(Get-Date -Format s) [prep] $msg`n"
    try { [IO.File]::AppendAllText($log, $line, $utf8) }
    catch { try { [IO.File]::AppendAllText((Join-Path $env:TEMP 'bio-social-fallback.log'), $line, $utf8) } catch { } }
}
$mutex = $null
$held = $false
$code = 1
try {
    try { New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null } catch { }
    Log "run start (pid $PID, user $env:USERNAME)"
    $py = 'C:\Python314\python.exe'
    $deploySrc = if ($env:BIO_PACKAGE_DEPLOY) { $env:BIO_PACKAGE_DEPLOY } else { 'D:\apps\bio-automation\src' }
    if (-not (Test-Path (Join-Path $deploySrc 'bio_automation\__init__.py'))) { Log "no bio_automation package under $deploySrc; nothing prepared"; throw 'no package' }
    $env:PYTHONPATH = $deploySrc
    [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
    $env:PYTHONIOENCODING = 'utf-8'

    $mutex = New-Object System.Threading.Mutex($false, 'Global\bio-social-run')
    try { $held = $mutex.WaitOne([TimeSpan]::FromMinutes(30)) }
    catch [System.Threading.AbandonedMutexException] { $held = $true; Log 'the previous holder of Global\bio-social-run died; taking the mutex' }
    if (-not $held) { Log 'gave up: Global\bio-social-run was still held after 30 minutes (a post run is stuck?); prep skipped'; throw 'mutex timeout' }

    Log 'bio social-prep'
    $ErrorActionPreference = 'Continue'          # python writing to stderr is not a PowerShell error; the exit code decides
    & $py -m bio_automation.cli social-prep 2>&1 | ForEach-Object { [IO.File]::AppendAllText($log, "$_`n", $utf8) }
    $code = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($code -ne 0) { Log "bio social-prep exited $code" } else { Log 'prep finished' }
}
catch {
    Log "FAILED before or during the run: $($_.Exception.Message) (line $($_.InvocationInfo.ScriptLineNumber))"
    $code = 1
}
finally {
    if ($held -and $mutex) { try { $mutex.ReleaseMutex() } catch { } }
    if ($mutex) { $mutex.Dispose() }
}
exit $code
