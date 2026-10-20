# run-console.ps1: keep one inat_console.py instance running on loopback (127.0.0.1:<Port>) for the lab front door.
#   -Tag  ''  = exported-files queue,  '_lc' = LightCraft queue (BIO_QUEUE_TAG)
# The console only listens on loopback; reachability from the tailnet/LAN is the front door's job (Caddy, TLS + password).
# Task Scheduler ends this script with TerminateProcess (no cleanup), so on start any earlier instance of THIS console on THIS port
# is stopped first (identified by command line), then python runs in the foreground; on exit: log, back off 5 s, restart.
param([string]$Tag = '', [int]$Port = 8777)
$ErrorActionPreference = 'Continue'
$py = 'F:\bio-automation-data\venv\Scripts\python.exe'
$script = 'F:\bio-automation-data\inat_console.py'
$log = "F:\services\inat-console\logs\console$Tag.log"
function Log([string]$m) { try { Add-Content -Path $log -Value ("{0} {1}" -f (Get-Date -Format 's'), $m) } catch { } }
$env:BIO_QUEUE_TAG = $Tag
$env:BIO_CONSOLE_PORT = "$Port"
$env:PYTHONIOENCODING = 'utf-8'
while ($true) {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {
        $_.CommandLine -and $_.CommandLine -match 'inat_console\.py' -and $_.ProcessId -ne $PID
    } | ForEach-Object {
        # only an instance on this port: read its environment is not possible, so match by the listening socket owner
        $own = (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue).OwningProcess
        if ($own -contains $_.ProcessId) { Log "stopping stale console pid $($_.ProcessId) on :$Port"; Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    }
    Log "starting console tag='$Tag' port=$Port"
    $p = Start-Process -FilePath $py -ArgumentList @('-I', $script) -WorkingDirectory 'F:\bio-automation-data' -PassThru -WindowStyle Hidden `
         -RedirectStandardOutput "F:\services\inat-console\logs\console$Tag.out" -RedirectStandardError "F:\services\inat-console\logs\console$Tag.err"
    $p.WaitForExit()
    Log "console exited with code $($p.ExitCode); restarting in 5 s"
    Start-Sleep -Seconds 5
}
