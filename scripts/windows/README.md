# Scheduled-task scripts for the Bluesky workflow

Prep, posting and watching are separate tasks. Posting never waits for prep beyond 10 minutes (it posts what is already drafted), so a slow `social-index` can no longer delay a post.

| File | Task | When | What | Exit code |
|---|---|---|---|---|
| `run-social-prep.ps1` | `Bio-Social-Prep` | daily 03:00 | `bio social-prep`: disk preflight (C: >= 15 GB, D: >= 20 GB), incremental `social-index`, `darkroom-sync`, `social-draft`; each step timed and audited as `prep_step` | that of `social-prep`; 1 when it could not run (no package, mutex wait expired) |
| `run-social-post.ps1` | `Bio-Social-Post` | 08:00, 12:00, 20:00 | `social-selfcheck` when that command exists (never blocks), then `social-post --live` | that of `social-post --live`; 1 when it could not start the run |
| `run-social-watch.ps1` | `Bio-Social-Watch` | 09:00, 13:00, 21:00 and 21:30 | `social-alerts --notify` (60 minutes after each slot); from 21:20 on also `social-engagement --refresh`, `social-digest` and (weekly, self-skipping) `social-audit-live` | 1 when an error-level alert exists, a step failed or the alert command failed; 0 otherwise (warnings pass) |
| `run-qdrant-sync.ps1` | `Bio-Qdrant-Sync` | daily 05:00 | `bio qdrant-sync` then `bio qdrant-inat-sync`; holds `Global\bio-social-run` (waits up to 90 minutes, so it never overlaps prep or a post run on the GPU) and `Global\bio-qdrant-sync`; a failed sync does not stop the other; failure sends an alert | 1 when a sync failed or the script could not run; 0 otherwise |
| `notify-to-inbox.ps1` | (not a task) | called by `social-alerts` | the default alert command: appends the alert to `D:\bio-results\social\ALERTS.txt` | 0 written |

All of them run the package in `$env:BIO_PACKAGE_DEPLOY`, else `D:\apps\bio-automation\src`, and append UTF-8 to `D:\bio-cache\logs\social.log` (one tagged line per step, `[prep]`, `[post]`, `[watch]`). A start line is logged before anything can fail, the child's output is appended line by line as it arrives (a hang leaves a trace), and when social.log itself cannot be written the line goes to `%TEMP%\bio-social-fallback.log`.

**Mutual exclusion.** Prep and post hold the named Windows mutex `Global\bio-social-run`. Prep holds it for the whole run and waits up to 30 minutes for it (a prep that missed 03:00 and starts at wake-up waits for a post run that is in progress; if it still cannot get it, it logs that and exits 1). A post run waits up to 10 minutes for a running prep, then posts anyway and logs `OVERLAP` (a missed post is worse than an overlap). A holder that died abandons the mutex; the next script logs that and takes it.

Drift: `bio social-selfcheck` compares these files byte for byte (sha256) with the copies in `F:\services\bio-social` (setting `BIO_SOCIAL_SCRIPTS_DIR`) and warns (`script-drift`) for a missing or different one; `deploy_prod.ps1` refreshes them and verifies the hashes. Edit the repo copy, never the deployed one.

Nothing here is applied automatically. `scripts\deploy_prod.ps1` copies every `scripts\windows\*.ps1` to `F:\services\bio-social\` after a deploy whose selfcheck passed (never `run-social.ps1`, the old single script, which stays as the rollback). `register-social-tasks.ps1` registers `Bio-Social-Prep`, `Bio-Social-Post` and `Bio-Social-Watch` with the principal and settings of `Bio-Social-Daily`, DISABLED by default so nothing double-posts; `-Cutover` disables `Bio-Social-Daily` (it is not deleted) and enables the three (`-WhatIf` shows the plan). The commands below do the same by hand, in an elevated Windows PowerShell 5.1.

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

# watch: 09:00, 13:00, 21:00 (alerts) and 21:30 (alerts + engagement + digest), at most 20 minutes
$watch = New-ScheduledTaskAction -Execute $ps -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$dir\run-social-watch.ps1`""
$watchSettings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 20)
$watchTimes = '09:00', '13:00', '21:00', '21:30' | ForEach-Object { New-ScheduledTaskTrigger -Daily -At $_ }
Register-ScheduledTask -TaskName 'Bio-Social-Watch' -Action $watch -Trigger $watchTimes -Settings $watchSettings -Principal $who
```

The slot times are the ones in `BIO_SOCIAL_SLOTS` (08:00, 12:00, 20:00). If you change them, move the watch triggers to slot + 60 minutes: `expected_post_missing` fires 60 minutes after a slot and `run_silent` after 30, so an earlier scan cannot see a missing post. A scan before a slot's check time is harmless, it just finds nothing yet.

## The alert command (SOCIAL_ALERT_COMMAND)

Without a command, alerts are only appended to `D:\bio-cache\state\social_alerts.jsonl`, which nobody reads. The tasks run as S4U: there is no desktop, so a toast, a message box or `msg.exe` cannot reach you. What does work there is a file, a network call or mail.

**Default (what `run-social-watch.ps1` uses when `BIO_SOCIAL_ALERT_COMMAND` is not set): a file inbox.** `notify-to-inbox.ps1` reads the alert from stdin as UTF-8 and appends one line to `D:\bio-results\social\ALERTS.txt` (override with `BIO_SOCIAL_ALERT_INBOX`). Open that file; delete lines when read. The watch task then exits 1 while an error alert is inside the 24 hour window, so Task Scheduler's "Last Run Result" and `Get-ScheduledTaskInfo Bio-Social-Watch` show it too.

To set the command explicitly (everywhere `bio social-alerts` runs, including the 03:00 prep and by hand), set it for the user once, in Windows PowerShell:

```powershell
[Environment]::SetEnvironmentVariable('BIO_SOCIAL_ALERT_COMMAND',
  "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe -NoProfile -ExecutionPolicy Bypass -File F:\services\bio-social\notify-to-inbox.ps1", 'User')
```

How the command is run: split with Windows quoting rules and no shell, so paths with spaces must be double-quoted; the alert text (`[level] kind: message`) is written to its stdin as UTF-8 bytes and stdin is closed; stdout is discarded and stderr is kept for the log. It has 30 seconds in total: on a timeout the command and every process it started are killed (`taskkill /T /F`) and the failure is recorded. Exit 0 = delivered (`alert_sent` in the audit log and a `notified` record in the alerts log); anything else, or a timeout, is `alert_failed` and the alert is sent again by the next scan. To send something else (mail, a webhook) write a script that reads UTF-8 stdin and exits 0 on success and point the variable at it.

What was and was not tested: the parse of all four scripts under Windows PowerShell 5.1 (5.1.26100), the mutex waits (waiting for a holder then running, post running anyway after its wait with the overlap logged, prep giving up with exit 1; an abandoned mutex was not simulated), the exit codes of post/prep/watch against a fake `python`, the UTF-8 inbox write, and the Python side of the notifier (stdin bytes, exit code, stderr, a hard 30 s bound with a grandchild that holds the handles, process-tree kill). Not tested: registering the tasks under S4U, running at the real times, and whether the `User` environment variable is visible to an S4U task (S4U may not load the user's registry hive, which is why `run-social-watch.ps1` sets a default itself; the 03:00 prep scan then still runs without a command and its alerts are sent by the next watch run, because a logged alert stays pending until it is notified).

## Switch over from the single script

Instead of registering a second post task, point the existing `Bio-Social-Daily` at the new post script and add only the prep and watch tasks:

```powershell
Set-ScheduledTask -TaskName 'Bio-Social-Daily' -Action (New-ScheduledTaskAction -Execute $ps -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$dir\run-social-post.ps1`"") `
  -Settings (New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 30))
```

Run `Start-ScheduledTask Bio-Social-Prep` once by hand and read `D:\bio-cache\logs\social.log` (and the `prep_step` records in `social_audit.jsonl`) before relying on 03:00; do the same with `Start-ScheduledTask Bio-Social-Watch` and look for the line in `ALERTS.txt` after the first real alert (`bio social-alerts --notify --since 7d` replays a recent one that was never notified).

## Roll back to the old single script

```powershell
Unregister-ScheduledTask -TaskName 'Bio-Social-Prep' -Confirm:$false
Unregister-ScheduledTask -TaskName 'Bio-Social-Watch' -Confirm:$false
# (and, if Bio-Social-Post was registered:)  Unregister-ScheduledTask -TaskName 'Bio-Social-Post' -Confirm:$false
Set-ScheduledTask -TaskName 'Bio-Social-Daily' -Action (New-ScheduledTaskAction -Execute $ps -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$dir\run-social.ps1`"") `
  -Settings (New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 1))
```

`run-social.ps1` is untouched and does prep (once a day, marker file) plus the post, exactly as before.
