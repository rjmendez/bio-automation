<#
.SYNOPSIS
  Runs the iNat suggestion + gallery refresh once (pipeline/stages/suggest_refresh.py) and returns its exit code.

.DESCRIPTION
  What it does, in order, stopping at the first failing step (exit 1, the step is named in <data>\suggest_refresh_last.json and suggest_refresh.log):
  live iNat observations -> photos of newly posted observations -> BioCLIP 2 vectors -> neighbour-vote bins -> zero-shot calibration (scores the
  observations posted since the last run on their own and prints DRIFT lines where the earlier numbers were optimistic) -> per queue: suggestions,
  then the inat-pending gallery folders. A lock file stops two runs of the refresh overlapping. The steps and interpreters come from <data>\suggest_refresh.json.

  GPU courtesy: it takes the named mutex Global\bio-social-run, the one Bio-Social-Prep, the Qdrant sync (run-qdrant-sync.ps1) and every post run hold,
  so GPU-heavy work never starts while another of them works. It waits up to -MutexWaitMinutes (default 90, like the sync) and then gives up with exit 1
  and a line in suggest_refresh.log. While it holds the mutex a post run waits for it for at most 10 minutes and then posts anyway (that script's rule:
  a missed post is worse than a shared GPU); at 06:30 the refresh is done in about 15 minutes, long before the 08:00 post.

  This script registers nothing. The scheduled task Bio-Suggest-Refresh (daily 06:30, S4U, limited, 2 h limit, IgnoreNew) was registered on 2026-10-24
  at the owner's request (AGENTS.md: persistent tasks need an explicit grant). It needs the GPU (BioCLIP 2 embeds on the card with the most free VRAM).

.EXAMPLE
  # one run by hand
  powershell -NoProfile -File scripts\run-suggest-refresh.ps1

.EXAMPLE
  # how the task was registered (kept here so it can be recreated)
  $a = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument '-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "C:\path\to\bio-automation\scripts\run-suggest-refresh.ps1"'
  $t = New-ScheduledTaskTrigger -Daily -At '06:30'
  $p = New-ScheduledTaskPrincipal -UserId '<your-windows-username>' -LogonType S4U -RunLevel Limited
  $s = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 2) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
  Register-ScheduledTask -TaskName 'Bio-Suggest-Refresh' -Action $a -Trigger $t -Principal $p -Settings $s
#>
[CmdletBinding()]
param(
    [string]$Python = 'C:\Python314\python.exe',
    [string]$Repo = '',
    [string]$DataDir = 'F:\bio-automation-data',
    [double]$MutexWaitMinutes = 90
)

# $PSScriptRoot is empty inside a param() default under some hosts (it broke scripts/premerge.ps1 the same way), so it is read here.
if (-not $Repo) { $Repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path) }
$env:BIO_DATA = $DataDir
$script = Join-Path $Repo 'pipeline\stages\suggest_refresh.py'
$log = Join-Path $DataDir 'suggest_refresh.log'
function Log([string]$msg) {
    try { Add-Content -LiteralPath $log -Value ("{0} [wrapper] {1}" -f (Get-Date -Format 'HH:mm:ss'), $msg) } catch { }
    Write-Host $msg
}
if (-not (Test-Path -LiteralPath $script)) { Log "run-suggest-refresh: $script does not exist"; exit 2 }

$mutex = New-Object System.Threading.Mutex($false, 'Global\bio-social-run')
$got = $false
try { $got = $mutex.WaitOne([TimeSpan]::FromMinutes($MutexWaitMinutes)) }
catch [System.Threading.AbandonedMutexException] { $got = $true; Log 'the previous holder of Global\bio-social-run died; taking the mutex' }
if (-not $got) {
    $mutex.Dispose()
    Log "gave up: Global\bio-social-run was still held after $MutexWaitMinutes minute(s) (Bio-Social-Prep, the Qdrant sync or a post run); nothing refreshed"
    exit 1
}
$code = 1
try {
    & $Python $script
    if ($null -ne $LASTEXITCODE) { $code = $LASTEXITCODE }
}
finally {
    try { $mutex.ReleaseMutex() } catch { }
    $mutex.Dispose()
}
exit $code
