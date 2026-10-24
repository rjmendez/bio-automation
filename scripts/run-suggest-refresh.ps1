<#
.SYNOPSIS
  Runs the iNat suggestion + gallery refresh once (pipeline/stages/suggest_refresh.py) and returns its exit code.

.DESCRIPTION
  What it does, in order, stopping at the first failing step (exit 1, the step is named in <data>\suggest_refresh_last.json and suggest_refresh.log):
  live iNat observations -> photos of newly posted observations -> BioCLIP 2 vectors -> neighbour-vote bins -> zero-shot calibration (scores the
  observations posted since the last run on their own and prints DRIFT lines where the earlier numbers were optimistic) -> per queue: suggestions,
  then the inat-pending gallery folders. A lock file stops two runs overlapping. The steps and interpreters come from <data>\suggest_refresh.json.

  THIS SCRIPT REGISTERS NOTHING. A scheduled task that calls it is the owner's to create (AGENTS.md: persistent tasks need an explicit grant).
  It needs the GPU (BioCLIP 2 embeds on the card with the most free VRAM) and about 15 minutes with nothing new to fetch, so pick a quiet hour.

.EXAMPLE
  # one run by hand
  powershell -NoProfile -File scripts\run-suggest-refresh.ps1

.EXAMPLE
  # what the owner could register (daily 06:30, runs whether or not anyone is logged on); review before running
  $a = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument '-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File D:\apps\bio-automation\scripts\run-suggest-refresh.ps1'
  $t = New-ScheduledTaskTrigger -Daily -At 6:30am
  Register-ScheduledTask -TaskName 'Bio-Suggest-Refresh' -Action $a -Trigger $t -RunLevel Limited
#>
[CmdletBinding()]
param(
    [string]$Python = 'C:\Python314\python.exe',
    [string]$Repo = '',
    [string]$DataDir = 'F:\bio-automation-data'
)

# $PSScriptRoot is empty inside a param() default under some hosts (it broke scripts/premerge.ps1 the same way), so it is read here.
if (-not $Repo) { $Repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path) }
$env:BIO_DATA = $DataDir
$script = Join-Path $Repo 'pipeline\stages\suggest_refresh.py'
if (-not (Test-Path -LiteralPath $script)) { Write-Host "run-suggest-refresh: $script does not exist"; exit 2 }
& $Python $script
exit $LASTEXITCODE
