<#
.SYNOPSIS
  Fast-forward-only deploy of the prod tree (D:\apps\bio-automation) to origin/main, with a test gate and a rollback.

.DESCRIPTION
  The prod tree live-reloads the NAS pipeline workers and runs the scheduled Bluesky posting task, so a bad merge breaks posting at the next run.
  Steps (each is logged to D:\bio-cache\logs\deploy.log):
    1. fetch origin; print the commits and the diffstat that would be deployed
    2. refuse unless the prod tree is on the target branch, clean (tracked files), and origin/<branch> is a fast-forward of HEAD
    3. export origin/<branch> to a temporary folder under D:\bio-cache\tmp (git archive; the prod tree is not touched), run
       scripts\check_test_honesty.py and the targeted social tests there, basetemp on D:
    4. git merge --ff-only in the prod tree
    5. run `bio social-selfcheck` from the prod tree (PYTHONPATH=<prod>\src, the editable install may point elsewhere);
       exit code 2 (or any failure to run it) -> git reset --hard to the previous head and exit 1; exit 1 (warnings) is accepted and logged
  -WhatIf prints the plan (it still fetches, which only updates origin/* refs) and changes nothing else.

  Windows PowerShell 5.1 and PowerShell 7 both run it.

.EXAMPLE
  powershell -File scripts\deploy_prod.ps1 -WhatIf
  powershell -File scripts\deploy_prod.ps1
  powershell -File scripts\deploy_prod.ps1 -ProdDir D:\bio-cache\tmp\scratch-clone -LogFile D:\bio-cache\tmp\deploy-test.log   # a rehearsal
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$ProdDir = 'D:\apps\bio-automation',
    [string]$Branch = 'main',
    [string]$Remote = 'origin',
    [string]$Python = 'C:\Python314\python.exe',
    [string]$TmpRoot = 'D:\bio-cache\tmp',
    [string]$LogFile = 'D:\bio-cache\logs\deploy.log',
    [double]$MinFreeGbC = 15,
    [switch]$AllowOtherBranch,
    [switch]$SkipTests
)

$ErrorActionPreference = 'Continue'      # git writes progress to stderr; success is judged by $LASTEXITCODE, not by stderr
$plan = [bool]$WhatIfPreference
$WhatIfPreference = $false               # the plan is handled by hand below; leaving it set would make every cmdlet (Add-Content, module imports) only pretend
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$work = Join-Path $TmpRoot "deploy-$stamp"
$basetemp = Join-Path $TmpRoot "pytest-deploy-$stamp"

function Log([string]$msg) {
    $line = '{0} {1}' -f (Get-Date -Format 'yyyy-MM-ddTHH:mm:ss'), $msg
    Write-Host $line
    try {
        $dir = Split-Path -Parent $LogFile
        if ($dir -and -not (Test-Path -LiteralPath $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
        Add-Content -LiteralPath $LogFile -Value $line -Encoding UTF8
    } catch { Write-Host "(deploy log not writable: $($_.Exception.Message))" }
}

function Invoke-Git { & git -C $ProdDir @args 2>&1 | ForEach-Object { "$_" } }

function Cleanup {
    foreach ($p in @($work, $basetemp)) {
        if ($p -and (Test-Path -LiteralPath $p) -and $p.StartsWith($TmpRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
            Remove-Item -LiteralPath $p -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}

function Fail([string]$msg) {
    Log "DEPLOY FAILED: $msg"
    Cleanup
    exit 1
}

Log "deploy start: prod=$ProdDir branch=$Branch remote=$Remote whatif=$plan"

if (-not (Test-Path -LiteralPath (Join-Path $ProdDir '.git'))) { Fail "$ProdDir is not a git checkout" }
$c = Get-PSDrive -Name 'C' -ErrorAction SilentlyContinue
if ($c -and ($c.Free / 1GB) -lt $MinFreeGbC) { Fail ("C: has {0:N1} GB free, below {1:N0} (the NAS pipeline pauses there)" -f ($c.Free / 1GB), $MinFreeGbC) }

# 1. fetch and show what would change
$fetch = Invoke-Git fetch -q $Remote
if ($LASTEXITCODE -ne 0) { Fail "git fetch $Remote failed: $($fetch -join ' ')" }
$prev = (Invoke-Git rev-parse HEAD | Select-Object -First 1).Trim()
$target = (Invoke-Git rev-parse "$Remote/$Branch" | Select-Object -First 1).Trim()
$current = (Invoke-Git rev-parse --abbrev-ref HEAD | Select-Object -First 1).Trim()
Log "prod head $prev on '$current'; $Remote/$Branch is $target"
if ($prev -eq $target) {
    Log "nothing to deploy: prod already is $Remote/$Branch"
}
else {
    Log "commits to deploy ($prev..$target):"
    Invoke-Git log --oneline "$prev..$target" | ForEach-Object { Log "  $_" }
    Log 'diffstat:'
    Invoke-Git diff --stat "$prev..$target" | ForEach-Object { Log "  $_" }
}

# 2. preconditions
$problems = @()
if ($current -ne $Branch -and -not $AllowOtherBranch) { $problems += "prod tree is on '$current', not '$Branch' (-AllowOtherBranch to deploy anyway)" }
$dirty = @(Invoke-Git status --porcelain --untracked-files=no)
if ($dirty.Count -gt 0) { $problems += "prod tree has uncommitted changes to tracked files: $($dirty -join '; ')" }
Invoke-Git merge-base --is-ancestor $prev $target | Out-Null
if ($LASTEXITCODE -ne 0) { $problems += "$Remote/$Branch is not a fast-forward of the prod head (diverged or behind); nothing will be merged" }

if ($plan) {
    Log 'PLAN (-WhatIf): nothing below is executed'
    if ($problems.Count -gt 0) { $problems | ForEach-Object { Log "  would stop: $_" } }
    if ($prev -eq $target) { Log '  1. nothing to merge; would only run social-selfcheck from the prod tree' }
    else {
        if (-not $SkipTests) {
            Log "  1. git archive $Remote/$Branch -> $work; run scripts\check_test_honesty.py and tests\test_social*.py tests\test_themes.py tests\test_darkroom.py there (basetemp $basetemp)"
        }
        Log "  2. git -C $ProdDir merge --ff-only $Remote/$Branch   (prod live-reloads from here)"
        Log "  3. PYTHONPATH=$ProdDir\src $Python -m bio_automation.cli social-selfcheck   (exit 2 or a crash -> git reset --hard $prev, exit 1)"
    }
    Log "log file: $LogFile"
    exit 0
}

if ($problems.Count -gt 0) { Fail ($problems -join ' | ') }

# 3. tests from a temporary export (the prod tree stays as it is)
if ($prev -ne $target -and -not $SkipTests) {
    New-Item -ItemType Directory -Force -Path $work | Out-Null
    $tar = Join-Path $work 'src.tar'
    Invoke-Git archive --format=tar -o $tar $target | Out-Null
    if ($LASTEXITCODE -ne 0) { Fail "git archive $target failed" }
    $tree = Join-Path $work 'tree'
    New-Item -ItemType Directory -Force -Path $tree | Out-Null
    $tarExe = Join-Path $env:SystemRoot 'System32\tar.exe'          # not Git's GNU tar: it reads D:\... as host:path
    if (-not (Test-Path -LiteralPath $tarExe)) { $tarExe = 'tar' }
    Push-Location $tree
    & $tarExe -xf '..\src.tar'
    $rc = $LASTEXITCODE
    Pop-Location
    if ($rc -ne 0) { Fail 'could not unpack the export' }
    Remove-Item -LiteralPath $tar -Force

    $oldPath = $env:PYTHONPATH
    $env:PYTHONPATH = Join-Path $tree 'src'
    try {
        Log 'test honesty check'
        Push-Location $tree
        & $Python 'scripts\check_test_honesty.py' 2>&1 | ForEach-Object { Log "  $_" }
        if ($LASTEXITCODE -ne 0) { Pop-Location; Fail 'scripts\check_test_honesty.py reported violations' }
        $tests = @(Get-ChildItem -Path 'tests' -Filter 'test_social*.py' | ForEach-Object { "tests/$($_.Name)" }) + @('tests/test_themes.py', 'tests/test_darkroom.py')
        Log ("targeted tests: " + ($tests -join ' '))
        & $Python -m pytest -q -p no:cacheprovider "--basetemp=$basetemp" @tests 2>&1 | ForEach-Object { Log "  $_" }
        $rc = $LASTEXITCODE
        Pop-Location
        if ($rc -ne 0) { Fail "targeted tests failed (pytest exit $rc); prod was not touched" }
        Log 'targeted tests passed'
    }
    finally { $env:PYTHONPATH = $oldPath }
}

# 4. fast-forward the prod tree
$rolledBack = $false
if ($prev -ne $target) {
    $merge = Invoke-Git merge --ff-only $target
    if ($LASTEXITCODE -ne 0) { Fail "git merge --ff-only failed: $($merge -join ' ')" }
    Log "prod fast-forwarded to $target"
}

# 5. self-check from the prod tree; roll back on a failure
$oldPath = $env:PYTHONPATH
$env:PYTHONPATH = Join-Path $ProdDir 'src'
try {
    Log 'social-selfcheck from the prod tree'
    $out = & $Python -m bio_automation.cli social-selfcheck 2>&1 | ForEach-Object { "$_" }
    $rc = $LASTEXITCODE
    $out | ForEach-Object { Log "  $_" }
}
finally { $env:PYTHONPATH = $oldPath }

if ($rc -ge 2 -or $null -eq $rc) {
    Log "selfcheck failed (exit $rc)"
    if ($prev -ne $target) {
        $reset = Invoke-Git reset --hard $prev
        if ($LASTEXITCODE -ne 0) { Fail "ROLLBACK FAILED, prod is at $target and needs a hand: $($reset -join ' ')" }
        Log "rolled back: prod reset to $prev"
        $rolledBack = $true
    }
    Fail ('selfcheck failed after the deploy' + $(if ($rolledBack) { '; rolled back' } else { '; nothing was merged' }))
}
if ($rc -eq 1) { Log 'selfcheck passed with warnings (see above)' }
Log "DEPLOY OK: prod is at $(Invoke-Git rev-parse HEAD | Select-Object -First 1)"
Cleanup
exit 0
