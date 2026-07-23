param(
    [string]$Name = "smart-checkpoint",
    [switch]$AutoCommit,
    [switch]$Push,
    [switch]$Watch,
    [int]$IntervalSeconds = 600,
    [switch]$IncludeUntracked
)

$ErrorActionPreference = "Stop"

function Write-Step($Text) {
    Write-Host ""
    Write-Host "== $Text ==" -ForegroundColor Cyan
}

function Invoke-Git($Arguments) {
    & git @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "git $($Arguments -join ' ') failed"
    }
}

function Test-SourcePath($Path) {
    $Normalized = $Path -replace '\\', '/'
    if ($Normalized -match '(^|/)(__pycache__|\.pytest_cache|\.mypy_cache)(/|$)') { return $false }
    if ($Normalized -match '\.(mp4|mov|avi|mkv|wav|webm|jpg|jpeg|png|gif|pt|pth|onnx|ckpt|safetensors|bin|log|bak)$') { return $false }
    if ($Normalized -match '(^|/)(\.codex|\.agents|\.tmp|_tmp|tmp_|debug_|test_output|04_AI|mulu)') { return $false }
    return (
        $Normalized -eq "AGENTS.md" -or
        $Normalized -eq "README.md" -or
        $Normalized -eq "WORK_RECORD.md" -or
        $Normalized -like "docs/*" -or
        $Normalized -like "scripts/*" -or
        $Normalized -like "spvideo/*" -or
        $Normalized -like "tests/*" -or
        $Normalized -like "web_ui/*" -or
        $Normalized -like "ui/*" -or
        $Normalized -like "*.py" -or
        $Normalized -eq ".gitignore"
    )
}

function Get-TrackedChangedPaths {
    $Paths = @()
    $Paths += git diff --name-only
    $Paths += git diff --name-only --cached
    $Paths += git diff --name-only --diff-filter=D
    return $Paths | Where-Object { $_ } | Sort-Object -Unique
}

function Get-UntrackedSourcePaths {
    if (-not $IncludeUntracked) { return @() }
    return git ls-files --others --exclude-standard |
        Where-Object { $_ -and (Test-SourcePath $_) } |
        Sort-Object -Unique
}

function Get-WorktreeFingerprint {
    $Status = git status --porcelain=v1
    $Diff = git diff --binary
    $Raw = ($Status -join "`n") + "`n" + ($Diff -join "`n")
    $Bytes = [System.Text.Encoding]::UTF8.GetBytes($Raw)
    $Hash = [System.Security.Cryptography.SHA256]::Create().ComputeHash($Bytes)
    return [BitConverter]::ToString($Hash).Replace("-", "").ToLowerInvariant()
}

function Invoke-OneCheckpoint {
    $Repo = (git rev-parse --show-toplevel).Trim()
    Set-Location $Repo

    $Status = git status --porcelain=v1
    if (-not $Status) {
        Write-Step "Clean"
        Write-Host "No changes to checkpoint."
        return
    }

    Write-Step "Safe guard"
    $SafeScript = Join-Path $Repo "scripts/safe_checkpoint.ps1"
    & powershell -NoProfile -ExecutionPolicy Bypass -File $SafeScript -Name $Name
    if ($LASTEXITCODE -ne 0) {
        throw "safe_checkpoint.ps1 failed; no commit or push will be attempted."
    }

    if (-not $AutoCommit) {
        Write-Step "Local patch only"
        Write-Host "AutoCommit is off. A patch backup was created; no Git commit was made."
        return
    }

    $Tracked = Get-TrackedChangedPaths
    $Untracked = Get-UntrackedSourcePaths
    $StagePaths = @($Tracked + $Untracked | Where-Object { Test-SourcePath $_ } | Sort-Object -Unique)

    if (-not $StagePaths) {
        Write-Step "Nothing staged"
        Write-Host "Changes exist, but none match the source-code checkpoint allowlist."
        return
    }

    Write-Step "Stage allowlisted source files"
    foreach ($Path in $StagePaths) {
        Write-Host "  $Path"
        Invoke-Git @("add", "--", $Path)
    }

    $Staged = git diff --cached --name-only
    if (-not $Staged) {
        Write-Step "No staged diff"
        Write-Host "No staged source changes after filtering."
        return
    }

    Write-Step "Commit"
    $Stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Invoke-Git @("commit", "-m", "auto-checkpoint: $Name ($Stamp)")
    Write-Host "Local auto-checkpoint commit created." -ForegroundColor Green

    if ($Push) {
        Write-Step "Push"
        Invoke-Git @("push")
        Write-Host "Pushed checkpoint to GitHub remote." -ForegroundColor Green
    }
    else {
        Write-Step "Push skipped"
        Write-Host "Use -Push only after this repository is allowed to sync stable checkpoints automatically."
    }
}

$RepoRoot = (git rev-parse --show-toplevel).Trim()
Set-Location $RepoRoot

if ($Push -and -not $AutoCommit) {
    throw "-Push requires -AutoCommit."
}

if (-not $Watch) {
    Invoke-OneCheckpoint
    exit 0
}

Write-Step "Watch mode"
Write-Host "Repo: $RepoRoot"
Write-Host "Interval: $IntervalSeconds seconds"
Write-Host "AutoCommit: $AutoCommit"
Write-Host "Push: $Push"

$LastFingerprint = ""
while ($true) {
    try {
        $Current = Get-WorktreeFingerprint
        if ($Current -ne $LastFingerprint) {
            $LastFingerprint = $Current
            Invoke-OneCheckpoint
        }
    }
    catch {
        Write-Host "[guard-error] $($_.Exception.Message)" -ForegroundColor Red
    }
    Start-Sleep -Seconds $IntervalSeconds
}
