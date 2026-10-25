# One Bluesky post run (bio social-post --live), scheduled task Bio-Social-Post at 08:00, 12:00 and 20:00. It does no prep: drafts are
# made by run-social-prep.ps1 at 03:00, so a slow index can no longer delay a post. When the social-selfcheck command exists it runs first;
# its result is logged (exit 1 as a warning, only exit 2 as an error) and never blocks the post (the post's own gates decide what goes out).
# Exit code: the exit code of `bio social-post --live` (0 = the run finished; a blocked or withdrawn status is for social-alerts to report),
# 1 when the script could not start the run at all (no package, no session, an unexpected error). Every failure lands in social.log.
# Mutual exclusion with prep: the named Windows mutex Global\bio-social-run. Prep holds it while it drafts; this script waits up to 10 minutes
# for it and then RUNS ANYWAY (a missed post is worse than an overlap), logging the overlap.
# The Qdrant sync (run-qdrant-sync.ps1) takes the same mutex and is capped at 150 minutes (05:00 -> 07:30), so this wait is normally zero; an overlap
# with it or with prep can still happen after the 10 minutes, by design.
# Sign-in: Bluesky OAuth via the bsky-oauth library, "poster" profile; `bio social-login` once (session in F:\services\bio-social\oauth).
$ErrorActionPreference = 'Stop'
$log = 'D:\bio-cache\logs\social.log'
$utf8 = New-Object System.Text.UTF8Encoding($false)
# Log never throws: when social.log cannot be written the line goes to a fallback file in %TEMP%, so a failure is never completely silent.
function Log([string]$msg) {
    $line = "$(Get-Date -Format s) [post] $msg`n"
    try { [IO.File]::AppendAllText($log, $line, $utf8) }
    catch { try { [IO.File]::AppendAllText((Join-Path $env:TEMP 'bio-social-fallback.log'), $line, $utf8) } catch { } }
}
$mutex = $null
$held = $false
$code = 1
try {
    try { New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null } catch { }
    Log "run start (pid $PID, user $env:USERNAME)"
    $session = 'F:\services\bio-social\oauth\account.json'
    $py = 'C:\Python314\python.exe'
    $deploySrc = if ($env:BIO_PACKAGE_DEPLOY) { $env:BIO_PACKAGE_DEPLOY } else { 'D:\apps\bio-automation\src' }
    if (-not (Test-Path (Join-Path $deploySrc 'bio_automation\__init__.py'))) { Log "no bio_automation package under $deploySrc; nothing posted"; throw 'no package' }
    $env:PYTHONPATH = $deploySrc
    [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
    $env:PYTHONIOENCODING = 'utf-8'

    # Wait for prep (it holds the mutex while it drafts), at most 10 minutes; then post anyway.
    $mutex = New-Object System.Threading.Mutex($false, 'Global\bio-social-run')
    try { $held = $mutex.WaitOne([TimeSpan]::FromMinutes(10)) }
    catch [System.Threading.AbandonedMutexException] { $held = $true; Log 'the previous holder of Global\bio-social-run died; taking the mutex' }
    if (-not $held) { Log 'OVERLAP: prep still holds Global\bio-social-run after 10 minutes; posting anyway' }

    # The child's output is appended line by line as it arrives, so a hang or a kill at the time limit leaves a trace.
    function Bio([string[]]$cmd) {
        Log "bio $($cmd -join ' ')"
        $ErrorActionPreference = 'Continue'         # python writing to stderr is not a PowerShell error; the exit code decides
        & $py -m bio_automation.cli @cmd 2>&1 | ForEach-Object { [IO.File]::AppendAllText($log, "$_`n", $utf8) }
        $c = $LASTEXITCODE
        if ($c -ne 0 -and $cmd[0] -ne 'social-selfcheck') { Log "bio $($cmd[0]) exited $c" }      # selfcheck reports its own level below
        return $c
    }

    # Is there a social-selfcheck command in this package? (the module may not be deployed yet)
    $ErrorActionPreference = 'Continue'
    $help = & $py -m bio_automation.cli --help 2>&1 | Out-String
    $ErrorActionPreference = 'Stop'
    if ($help -match 'social-selfcheck') {
        # social-selfcheck: 0 ok, 1 warnings (logged as a warning, not an error), 2 a failure. Neither blocks the post: the post's own gates decide.
        $sc = Bio @('social-selfcheck')
        if ($sc -eq 1) { Log 'selfcheck: warn (exit 1): warnings only, see the lines above; the post goes on' }
        elseif ($sc -ne 0) { Log "ERROR: social-selfcheck failed (exit $sc): see the lines above; the post goes on and its gates decide" }
    }

    if (-not (Test-Path $session)) {
        Log "no Bluesky session (${session}): run ``bio social-login`` once; nothing posted"
        $code = 1
    }
    else {
        $code = Bio @('social-post', '--live')
        if ($code -ne 0) { Log "post run failed (exit $code)" } else { Log 'post run finished' }
    }
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
