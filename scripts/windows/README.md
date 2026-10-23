# Scheduled-task scripts for the Bluesky workflow

Prep and posting are separate tasks. Posting never waits for prep (it posts what is already drafted), so a slow `social-index` can no longer delay a post.

| File | Task | When | What |
|---|---|---|---|
| `run-social-prep.ps1` | `Bio-Social-Prep` | daily 03:00 | `bio social-prep`: disk preflight (C: >= 15 GB, D: >= 20 GB), incremental `social-index`, `darkroom-sync`, `social-draft`; each step timed and audited as `prep_step` |
| `run-social-post.ps1` | `Bio-Social-Post` | 08:00, 12:00, 20:00 | `social-selfcheck` when that command exists (never blocks), then `social-post --live`; exit 1 on failure |

Both run the package in `$env:BIO_PACKAGE_DEPLOY`, else `D:\apps\bio-automation\src`, and append UTF-8 to `D:\bio-cache\logs\social.log`.
Nothing here is applied automatically. Copy the scripts to `F:\services\bio-social\` (next to `run-social.ps1`), then run the commands below in an elevated Windows PowerShell 5.1.

## Register

```powershell
$dir = 'F:\services\bio-social'
$ps  = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
$who = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType S4U -RunLevel Limited

# prep: daily 03:00, at most 2 hours
$prep = New-ScheduledTaskAction -Execute $ps -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$dir\run-social-prep.ps1`""
$prepSettings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 2)
Register-ScheduledTask -TaskName 'Bio-Social-Prep' -Action $prep -Trigger (New-ScheduledTaskTrigger -Daily -At 03:00) -Settings $prepSettings -Principal $who

# post: 08:00, 12:00, 20:00, at most 30 minutes
$post = New-ScheduledTaskAction -Execute $ps -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$dir\run-social-post.ps1`""
$postSettings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 30)
$times = '08:00', '12:00', '20:00' | ForEach-Object { New-ScheduledTaskTrigger -Daily -At $_ }
Register-ScheduledTask -TaskName 'Bio-Social-Post' -Action $post -Trigger $times -Settings $postSettings -Principal $who
```

## Switch over from the single script

Instead of registering a second post task, point the existing `Bio-Social-Daily` at the new post script and add only the prep task:

```powershell
Set-ScheduledTask -TaskName 'Bio-Social-Daily' -Action (New-ScheduledTaskAction -Execute $ps -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$dir\run-social-post.ps1`"") `
  -Settings (New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 30))
```

Run `Start-ScheduledTask Bio-Social-Prep` once by hand and read `D:\bio-cache\logs\social.log` (and the `prep_step` records in `social_audit.jsonl`) before relying on 03:00.

## Roll back to the old single script

```powershell
Unregister-ScheduledTask -TaskName 'Bio-Social-Prep' -Confirm:$false
# (and, if Bio-Social-Post was registered:)  Unregister-ScheduledTask -TaskName 'Bio-Social-Post' -Confirm:$false
Set-ScheduledTask -TaskName 'Bio-Social-Daily' -Action (New-ScheduledTaskAction -Execute $ps -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$dir\run-social.ps1`"") `
  -Settings (New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 1))
```

`run-social.ps1` is untouched and does prep (once a day, marker file) plus the post, exactly as before.
