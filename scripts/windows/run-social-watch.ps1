# The watch task (Bio-Social-Watch): tells the owner when something is wrong, and writes the daily digest.
#   09:00, 13:00, 21:00   bio social-alerts --notify     (60 minutes after the 08:00, 12:00 and 20:00 post slots, which is when a missing post
#                                                         or a silent run can first be told from a slow one)
#   21:30                 bio social-engagement --refresh, then bio social-digest, then bio social-audit-live --weekly   (the evening run: -Step evening, chosen by the clock
#                                                         when run without -Step: from 21:20 on)
# Exit code: 1 when social-alerts reports an alert of level error (its exit code 2), when a step cannot run, or when the alert command failed;
# 0 otherwise (warnings do not fail the task; they are in the digest and the alerts log). So Task Scheduler's "last run result" shows red
# exactly while an error alert is inside the 24 h window.
# The alert command: BIO_SOCIAL_ALERT_COMMAND if set (machine/user environment), else the file inbox notify-to-inbox.ps1 next to this script,
# which appends each alert to D:\bio-results\social\ALERTS.txt. No desktop exists for an S4U task, so a toast or message box cannot be used.
param([ValidateSet('auto', 'alerts', 'evening')][string]$Step = 'auto')
$ErrorActionPreference = 'Stop'
$log = 'D:\bio-cache\logs\social.log'
$utf8 = New-Object System.Text.UTF8Encoding($false)
function Log([string]$msg) {
    $line = "$(Get-Date -Format s) [watch] $msg`n"
    try { [IO.File]::AppendAllText($log, $line, $utf8) }
    catch { try { [IO.File]::AppendAllText((Join-Path $env:TEMP 'bio-social-fallback.log'), $line, $utf8) } catch { } }
}
$code = 0
try {
    try { New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null } catch { }
    $now = Get-Date
    if ($Step -eq 'auto') { $Step = if ($now.Hour -gt 21 -or ($now.Hour -eq 21 -and $now.Minute -ge 20)) { 'evening' } else { 'alerts' } }
    Log "run start, step $Step (pid $PID, user $env:USERNAME)"
    $py = 'C:\Python314\python.exe'
    $deploySrc = if ($env:BIO_PACKAGE_DEPLOY) { $env:BIO_PACKAGE_DEPLOY } else { 'D:\apps\bio-automation\src' }
    if (-not (Test-Path (Join-Path $deploySrc 'bio_automation\__init__.py'))) { Log "no bio_automation package under $deploySrc"; throw 'no package' }
    $env:PYTHONPATH = $deploySrc
    [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
    $env:PYTHONIOENCODING = 'utf-8'
    if (-not $env:BIO_SOCIAL_ALERT_COMMAND) {
        $notifier = Join-Path $PSScriptRoot 'notify-to-inbox.ps1'
        $env:BIO_SOCIAL_ALERT_COMMAND = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe -NoProfile -ExecutionPolicy Bypass -File $notifier"
        Log "BIO_SOCIAL_ALERT_COMMAND not set: using the file inbox $notifier"
    }

    function Bio([string[]]$cmd) {
        Log "bio $($cmd -join ' ')"
        $ErrorActionPreference = 'Continue'
        & $py -m bio_automation.cli @cmd 2>&1 | ForEach-Object { [IO.File]::AppendAllText($log, "$_`n", $utf8) }
        $c = $LASTEXITCODE
        if ($c -ne 0) { Log "bio $($cmd[0]) exited $c" }
        return $c
    }

    if ($Step -eq 'evening') {
        # Engagement and the digest are best effort: a failure of either is logged and fails the task, but never stops the alert scan below.
        if ((Bio @('social-engagement', '--refresh')) -ne 0) { $code = 1 }
        if ((Bio @('social-digest')) -ne 0) { $code = 1 }
        # Weekly (the command skips itself while its last audit is under 7 days old): re-run the privacy gates on what is actually public. Exit 1 =
        # problems found, which the alert scan below turns into an error alert (live_audit_problem); 2 or more = the AppView could not be read.
        if ((Bio @('social-audit-live', '--weekly')) -ge 2) { $code = 1 }
    }
    $alerts = Bio @('social-alerts', '--notify')
    # social-alerts: 0 none, 1 warnings, 2 errors. Warnings pass; errors and a crash (any other code) fail the task.
    if ($alerts -ne 0 -and $alerts -ne 1) { Log "error-level alerts exist or social-alerts failed (exit $alerts): see the lines above and D:\bio-results\social\ALERTS.txt"; $code = 1 }
    if ($code -eq 0) { Log 'watch finished' }
}
catch {
    Log "FAILED: $($_.Exception.Message) (line $($_.InvocationInfo.ScriptLineNumber))"
    $code = 1
}
exit $code
