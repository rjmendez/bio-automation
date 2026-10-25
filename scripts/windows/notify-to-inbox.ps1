# Alert command for BIO_SOCIAL_ALERT_COMMAND: appends the alert text it reads from stdin (UTF-8) to a plain-text inbox the owner can open.
#   BIO_SOCIAL_ALERT_COMMAND = powershell.exe -NoProfile -ExecutionPolicy Bypass -File F:\services\bio-social\notify-to-inbox.ps1
# Scheduled tasks run as S4U with no desktop, so a toast, a message box or msg.exe cannot reach anyone; a file can. Exit 0 = written.
# Output file: BIO_SOCIAL_ALERT_INBOX, else D:\bio-results\social\ALERTS.txt. Delete lines or the file whenever you have read them.
$ErrorActionPreference = 'Stop'
$inbox = if ($env:BIO_SOCIAL_ALERT_INBOX) { $env:BIO_SOCIAL_ALERT_INBOX } else { 'D:\bio-results\social\ALERTS.txt' }
$utf8 = New-Object System.Text.UTF8Encoding($false)
try {
    $reader = New-Object System.IO.StreamReader([Console]::OpenStandardInput(), $utf8)
    $text = $reader.ReadToEnd().TrimEnd()
    if (-not $text) { [Console]::Error.WriteLine('notify-to-inbox: nothing on stdin'); exit 2 }
    New-Item -ItemType Directory -Force (Split-Path $inbox) | Out-Null
    [IO.File]::AppendAllText($inbox, "$(Get-Date -Format s)  $text`r`n", $utf8)
    exit 0
}
catch {
    [Console]::Error.WriteLine("notify-to-inbox: $($_.Exception.Message)")
    exit 1
}
