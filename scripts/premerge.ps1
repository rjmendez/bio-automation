<#
.SYNOPSIS
  Pre-merge gate to run locally until the GitHub workflow can be relied on: the test-honesty check, then the FULL suite (live tests excluded).

.DESCRIPTION
  Exactly what CI's tests-private job runs, from this checkout, with PYTHONPATH=<checkout>\src (the editable install may point at another tree)
  and the pytest basetemp on D: (never C:). The basetemp is removed afterwards. Nothing is skipped quietly: the script exits 1 unless both the honesty check
  and pytest exit 0, and it says so on the last line (`premerge: PASS` / `premerge: FAIL (...)`).
  The suite needs the bsky-oauth library (private repo) for tests/test_social_cleanup.py and the bsky boundary tests in tests/test_social.py; when it is
  not importable the script says so up front, and those tests fail rather than being left out.
  Windows PowerShell 5.1 and PowerShell 7 both run it.

.EXAMPLE
  powershell -File scripts\premerge.ps1
  powershell -File scripts\premerge.ps1 -Extra '-k','social' -KeepTemp
#>
[CmdletBinding()]
param(
    [string]$Repo = (Split-Path -Parent $PSScriptRoot),
    [string]$Python = 'C:\Python314\python.exe',
    [string]$TmpRoot = 'D:\bio-cache\tmp',
    [double]$MinFreeGbC = 15,
    [string[]]$Extra = @(),          # extra pytest arguments, e.g. -k expressions while iterating
    [switch]$KeepTemp
)

$ErrorActionPreference = 'Continue'
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$basetemp = Join-Path $TmpRoot "premerge-$stamp-$PID"

function Done([bool]$ok, [string]$why) {
    if (-not $KeepTemp -and (Test-Path -LiteralPath $basetemp) -and $basetemp.StartsWith($TmpRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        Remove-Item -LiteralPath $basetemp -Recurse -Force -ErrorAction SilentlyContinue
    }
    $env:PYTHONPATH = $script:oldPath
    if ($ok) { Write-Host 'premerge: PASS'; exit 0 }
    Write-Host "premerge: FAIL ($why)"; exit 1
}

$script:oldPath = $env:PYTHONPATH
if (-not ($TmpRoot -match '^[dD]:')) { Write-Host "premerge: FAIL (TmpRoot $TmpRoot is not on D:)"; exit 1 }
if (-not (Test-Path -LiteralPath (Join-Path $Repo 'scripts\check_test_honesty.py'))) { Write-Host "premerge: FAIL ($Repo is not a bio-automation checkout)"; exit 1 }
$c = Get-PSDrive -Name 'C' -ErrorAction SilentlyContinue
if ($c -and ($c.Free / 1GB) -lt $MinFreeGbC) { Write-Host ("premerge: FAIL (C: has {0:N1} GB free, below {1:N0})" -f ($c.Free / 1GB), $MinFreeGbC); exit 1 }

$env:PYTHONPATH = Join-Path $Repo 'src'
$fail = ''
Push-Location $Repo
try {
    $branch = (& git rev-parse --abbrev-ref HEAD 2>$null); $head = (& git rev-parse --short HEAD 2>$null)
    Write-Host "premerge: $Repo at $branch $head; basetemp $basetemp"
    & $Python -c "import bsky_oauth" 2>$null
    if ($LASTEXITCODE -ne 0) { Write-Host 'premerge: WARNING: bsky_oauth is not importable here; the tests that need it will FAIL (they are not skipped)' }

    Write-Host '== test honesty'
    & $Python 'scripts\check_test_honesty.py' --fail-stale
    if ($LASTEXITCODE -ne 0) { $fail = 'scripts\check_test_honesty.py reported violations' }
    else {
        Write-Host '== full suite (live tests excluded)'
        New-Item -ItemType Directory -Force -Path $TmpRoot | Out-Null
        & $Python -m pytest tests -q -p no:cacheprovider "--basetemp=$basetemp" '--ignore-glob=tests/test_*_live.py' @Extra
        if ($LASTEXITCODE -ne 0) { $fail = "pytest exit $LASTEXITCODE" }
    }
}
finally { Pop-Location }
Done ($fail -eq '') $fail
