param(
    [int]$IntervalMinutes = 10,
    [switch]$AutoCommit,
    [switch]$Push,
    [string]$TaskName = "SP Smart Git Checkpoint"
)

$ErrorActionPreference = "Stop"

$Repo = (git rev-parse --show-toplevel).Trim()
$Guard = Join-Path $Repo "scripts\smart_git_guard.ps1"

if (-not (Test-Path $Guard)) {
    throw "Missing guard script: $Guard"
}

$Args = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-WindowStyle", "Hidden",
    "-File", "`"$Guard`"",
    "-Name", "scheduled",
    "-IncludeUntracked"
)

if ($AutoCommit) { $Args += "-AutoCommit" }
if ($Push) { $Args += "-Push" }

$Action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument ($Args -join " ") -WorkingDirectory $Repo
$Trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes $IntervalMinutes) -RepetitionDuration (New-TimeSpan -Days 3650)
$Principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Principal $Principal -Settings $Settings -Force | Out-Null

Write-Host "Installed scheduled Git guard: $TaskName" -ForegroundColor Green
Write-Host "Repo: $Repo"
Write-Host "Interval: every $IntervalMinutes minutes"
Write-Host "AutoCommit: $AutoCommit"
Write-Host "Push: $Push"
Write-Host ""
Write-Host "Remove it with:"
Write-Host "  Unregister-ScheduledTask -TaskName `"$TaskName`" -Confirm:`$false"
