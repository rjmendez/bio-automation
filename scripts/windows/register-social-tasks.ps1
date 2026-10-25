<#
.SYNOPSIS
  Registers the scheduled tasks of the split Bluesky workflow (prep, post, watch) next to the old single task Bio-Social-Daily.

.DESCRIPTION
  Tasks (the scripts are the ones in this folder, copied to F:\services\bio-social by scripts\deploy_prod.ps1):
    Bio-Social-Prep   daily 03:00                     run-social-prep.ps1    (at most 2 hours)
    Bio-Social-Post   08:00, 12:00, 20:00             run-social-post.ps1    (at most 30 minutes)
    Bio-Social-Watch  09:00, 13:00, 21:00, 21:30      run-social-watch.ps1   (at most 20 minutes)
  The principal (user, S4U logon, run level) and the settings (instances, battery rules, start-when-available, priority) are COPIED from the
  existing task Bio-Social-Daily, so the new tasks run exactly as the old one does; only the execution time limit differs per task. Without
  Bio-Social-Daily the same values are used (S4U, limited, IgnoreNew, StartWhenAvailable, no start on battery).

  Nothing double-posts by default: without -Cutover the new tasks are registered DISABLED (a task that already exists keeps its enabled state)
  and Bio-Social-Daily is not touched. With -Cutover, Bio-Social-Daily is DISABLED first (never deleted: it is the rollback) and the three new
  tasks are enabled. Rollback:
      Disable-ScheduledTask Bio-Social-Prep, Bio-Social-Post, Bio-Social-Watch;  Enable-ScheduledTask Bio-Social-Daily
  Run it in an elevated Windows PowerShell 5.1 or PowerShell 7. -WhatIf prints what would happen and changes nothing.

.EXAMPLE
  powershell -File register-social-tasks.ps1 -WhatIf
  powershell -File register-social-tasks.ps1              # registers the three tasks, disabled
  powershell -File register-social-tasks.ps1 -Cutover     # disables Bio-Social-Daily and enables the three tasks
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$Dir = 'F:\services\bio-social',
    [string]$DailyName = 'Bio-Social-Daily',
    [switch]$Cutover
)
$ErrorActionPreference = 'Stop'

function Say([string]$msg) { Write-Host ("{0} {1}" -f (Get-Date -Format 'yyyy-MM-ddTHH:mm:ss'), $msg) }

# The task definitions: name, script, trigger times, execution time limit.
$defs = @(
    @{ Name = 'Bio-Social-Prep';  Script = 'run-social-prep.ps1';  Times = @('03:00');                          Limit = (New-TimeSpan -Hours 2) },
    @{ Name = 'Bio-Social-Post';  Script = 'run-social-post.ps1';  Times = @('08:00', '12:00', '20:00');        Limit = (New-TimeSpan -Minutes 30) },
    @{ Name = 'Bio-Social-Watch'; Script = 'run-social-watch.ps1'; Times = @('09:00', '13:00', '21:00', '21:30'); Limit = (New-TimeSpan -Minutes 20) }
)

# Principal and settings of the old task.
$daily = Get-ScheduledTask -TaskName $DailyName -ErrorAction SilentlyContinue
if ($daily) {
    $who = $daily.Principal
    $s = $daily.Settings
    Say "copying the principal ($($who.UserId), $($who.LogonType), $($who.RunLevel)) and settings of $DailyName"
}
else {
    $who = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType S4U -RunLevel Limited
    $s = $null
    Say "no task ${DailyName}: using the default principal ($env:USERDOMAIN\$env:USERNAME, S4U, Limited) and settings"
}
$ps = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'

function New-Settings([TimeSpan]$limit, [bool]$disabled) {
    $p = @{ ExecutionTimeLimit = $limit; MultipleInstances = 'IgnoreNew'; StartWhenAvailable = $true }
    if ($s) {
        $p.MultipleInstances = [string]$s.MultipleInstances
        $p.StartWhenAvailable = [bool]$s.StartWhenAvailable
        if (-not $s.DisallowStartIfOnBatteries) { $p.AllowStartIfOnBatteries = $true }
        if (-not $s.StopIfGoingOnBatteries) { $p.DontStopIfGoingOnBatteries = $true }
        if ($s.WakeToRun) { $p.WakeToRun = $true }
    }
    if ($disabled) { $p.Disable = $true }
    $set = New-ScheduledTaskSettingsSet @p
    if ($s -and $s.Priority) { $set.Priority = $s.Priority }
    return $set
}

foreach ($d in $defs) {
    $script = Join-Path $Dir $d.Script
    if (-not (Test-Path -LiteralPath $script)) { Say "WARNING: $script does not exist yet (deploy_prod.ps1 copies it); the task is registered anyway" }
    $existing = Get-ScheduledTask -TaskName $d.Name -ErrorAction SilentlyContinue
    # always registered disabled (an existing task keeps its state); -Cutover enables the three below, AFTER Bio-Social-Daily is disabled
    $enable = [bool]($existing -and $existing.State -ne 'Disabled')
    $action = New-ScheduledTaskAction -Execute $ps -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$script`""
    $triggers = @($d.Times | ForEach-Object { New-ScheduledTaskTrigger -Daily -At $_ })
    $state = if ($enable) { 'ENABLED' } else { 'DISABLED' }
    if ($PSCmdlet.ShouldProcess($d.Name, "register ($($d.Times -join ', '), $state)")) {
        Register-ScheduledTask -TaskName $d.Name -Action $action -Trigger $triggers -Settings (New-Settings $d.Limit (-not $enable)) -Principal $who -Force | Out-Null
        Say "registered $($d.Name) at $($d.Times -join ', '), $state"
    }
    else { Say "would register $($d.Name) at $($d.Times -join ', '), $state" }
}

if ($Cutover) {
    # the old task goes first: at no moment do both post
    if ($daily) {
        if ($PSCmdlet.ShouldProcess($DailyName, 'disable (not delete)')) { Disable-ScheduledTask -TaskName $DailyName | Out-Null; Say "disabled $DailyName (kept as the rollback)" }
        else { Say "would disable $DailyName (kept as the rollback)" }
    }
    foreach ($d in $defs) {
        if ($PSCmdlet.ShouldProcess($d.Name, 'enable')) { Enable-ScheduledTask -TaskName $d.Name | Out-Null; Say "enabled $($d.Name)" }
    }
    Say 'cutover done. Rollback: Disable-ScheduledTask Bio-Social-Prep, Bio-Social-Post, Bio-Social-Watch; Enable-ScheduledTask Bio-Social-Daily'
}
else {
    Say "the new tasks are DISABLED unless they already were enabled; $DailyName is untouched. Run again with -Cutover to switch over."
}
