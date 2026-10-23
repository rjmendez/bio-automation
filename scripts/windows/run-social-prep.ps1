# Daily preparation for the Bluesky posts (bio social-prep): disk preflight, social-index (incremental), darkroom-sync, social-draft.
# Scheduled task Bio-Social-Prep, daily 03:00. Posting runs separately (run-social-post.ps1) and never waits for this; a failed prep
# only means fewer new drafts, and this script exits 1 so the task shows it.
# Runs the pinned production package (D:\apps\bio-automation, the tree the NAS pipeline uses), never a development checkout.
$ErrorActionPreference = 'Stop'
$log = 'D:\bio-cache\logs\social.log'
$py = 'C:\Python314\python.exe'
$deploySrc = if ($env:BIO_PACKAGE_DEPLOY) { $env:BIO_PACKAGE_DEPLOY } else { 'D:\apps\bio-automation\src' }
if (-not (Test-Path (Join-Path $deploySrc 'bio_automation\__init__.py'))) { Write-Error "no bio_automation package under $deploySrc"; exit 1 }
$env:PYTHONPATH = $deploySrc
New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null

# Windows PowerShell 5.1 writes `>>` as UTF-16 and decodes native output with the OEM code page: capture python's output as UTF-8
# and append it to the log with a BOM-less UTF-8 encoding.
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$env:PYTHONIOENCODING = 'utf-8'
$utf8 = [Text.UTF8Encoding]::new($false)
function Log([string]$msg) { [IO.File]::AppendAllText($log, "$(Get-Date -Format s) $msg`n", $utf8) }

Log 'bio social-prep'
$ErrorActionPreference = 'Continue'          # python writing to stderr is not a PowerShell error; the exit code decides
$out = & $py -m bio_automation.cli social-prep 2>&1 | ForEach-Object { "$_" } | Out-String
$code = $LASTEXITCODE
if ($out) { [IO.File]::AppendAllText($log, $out, $utf8) }
if ($code -ne 0) { Log "bio social-prep exited $code"; exit 1 }
Log 'prep finished'
exit 0
