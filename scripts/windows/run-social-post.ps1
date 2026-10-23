# One Bluesky post run (bio social-post --live), scheduled task Bio-Social-Post at 08:00, 12:00 and 20:00. It does no prep: drafts are
# made by run-social-prep.ps1 at 03:00, so a slow index can no longer delay a post. When the social-selfcheck command exists it runs first;
# its result is logged and never blocks the post (the post's own gates decide what goes out). Exits 1 when the post run fails.
# Sign-in: Bluesky OAuth via the bsky-oauth library, "poster" profile; `bio social-login` once (session in F:\services\bio-social\oauth).
$ErrorActionPreference = 'Stop'
$log = 'D:\bio-cache\logs\social.log'
$session = 'F:\services\bio-social\oauth\account.json'
$py = 'C:\Python314\python.exe'
$deploySrc = if ($env:BIO_PACKAGE_DEPLOY) { $env:BIO_PACKAGE_DEPLOY } else { 'D:\apps\bio-automation\src' }
if (-not (Test-Path (Join-Path $deploySrc 'bio_automation\__init__.py'))) { Write-Error "no bio_automation package under $deploySrc"; exit 1 }
$env:PYTHONPATH = $deploySrc
New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null

[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$env:PYTHONIOENCODING = 'utf-8'
$utf8 = [Text.UTF8Encoding]::new($false)
function Log([string]$msg) { [IO.File]::AppendAllText($log, "$(Get-Date -Format s) $msg`n", $utf8) }

function Bio([string[]]$cmd) {
    Log "bio $($cmd -join ' ')"
    $ErrorActionPreference = 'Continue'
    $out = & $py -m bio_automation.cli @cmd 2>&1 | ForEach-Object { "$_" } | Out-String
    $code = $LASTEXITCODE
    if ($out) { [IO.File]::AppendAllText($log, $out, $utf8) }
    if ($code -ne 0) { Log "bio $($cmd[0]) exited $code" }
    return $code
}

# Is there a social-selfcheck command in this package? (the module may not be deployed yet)
$ErrorActionPreference = 'Continue'
$help = & $py -m bio_automation.cli --help 2>&1 | Out-String
$ErrorActionPreference = 'Stop'
if ($help -match 'social-selfcheck') { Bio @('social-selfcheck') | Out-Null }

if (-not (Test-Path $session)) {
    Log "no Bluesky session (${session}): run ``bio social-login`` once; nothing posted"
    exit 1
}
$code = Bio @('social-post', '--live')
if ($code -ne 0) { Log 'post run failed'; exit 1 }
Log 'post run finished'
exit 0
