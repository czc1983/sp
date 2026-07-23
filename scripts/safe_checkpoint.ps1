param(
    [string]$Name = "checkpoint",
    [switch]$Commit,
    [switch]$Push
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

$Repo = (git rev-parse --show-toplevel).Trim()
Set-Location $Repo

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$SafeName = ($Name -replace '[^A-Za-z0-9_.-]', '-').Trim('-')
if ([string]::IsNullOrWhiteSpace($SafeName)) {
    $SafeName = "checkpoint"
}

Write-Step "Repository"
Write-Host $Repo

Write-Step "Git status"
git status --short

$PatchDir = Join-Path $Repo ".codex\checkpoints"
New-Item -ItemType Directory -Force -Path $PatchDir | Out-Null
$PatchFile = Join-Path $PatchDir "$Stamp-$SafeName.patch"
git diff --binary | Out-File -FilePath $PatchFile -Encoding utf8
Write-Host "Patch backup: $PatchFile" -ForegroundColor Green

$UntrackedList = @(git ls-files --others --exclude-standard)
if ($UntrackedList.Count -gt 0) {
    $UntrackedDir = Join-Path $PatchDir "$Stamp-$SafeName-untracked"
    New-Item -ItemType Directory -Force -Path $UntrackedDir | Out-Null
    $Manifest = Join-Path $UntrackedDir "_manifest.txt"
    $UntrackedList | Out-File -FilePath $Manifest -Encoding utf8
    foreach ($Relative in $UntrackedList) {
        $Source = Join-Path $Repo $Relative
        if (Test-Path -LiteralPath $Source -PathType Leaf) {
            $Target = Join-Path $UntrackedDir $Relative
            $TargetParent = Split-Path -Parent $Target
            New-Item -ItemType Directory -Force -Path $TargetParent | Out-Null
            Copy-Item -LiteralPath $Source -Destination $Target -Force
        }
    }
    Write-Host "Untracked file snapshot: $UntrackedDir" -ForegroundColor Green
}

Write-Step "Encoding guard"
$CriticalFiles = @(
    "web_ui/story_generate_dashboard.html",
    "web_ui/splitter_dashboard.html",
    "web_ui/server.py",
    "spvideo/scail2_client.py",
    "AGENTS.md",
    "README.md",
    "docs/GIT_CHECKPOINT_WORKFLOW.md",
    "docs/MODE2_FRONTEND_ACCIDENT_POSTMORTEM.md"
)

foreach ($File in $CriticalFiles) {
    $Path = Join-Path $Repo $File
    if (Test-Path $Path) {
        $Text = Get-Content -LiteralPath $Path -Raw -Encoding UTF8
        if ($Text.Contains([char]0xFFFD)) {
            throw "Replacement character found in $File. Stop and inspect encoding before continuing."
        }
    }
}
Write-Host "No replacement-character mojibake found in critical files." -ForegroundColor Green

Write-Step "HTML script syntax"
$Dashboard = Join-Path $Repo "web_ui/story_generate_dashboard.html"
if (Test-Path $Dashboard) {
    $Html = Get-Content -LiteralPath $Dashboard -Raw -Encoding UTF8
    $Matches = [regex]::Matches($Html, '(?is)<script\b[^>]*>(.*?)</script>')
    $Combined = New-Object System.Text.StringBuilder
    foreach ($Match in $Matches) {
        [void]$Combined.AppendLine($Match.Groups[1].Value)
    }
    $TempJs = Join-Path $env:TEMP "story_generate_dashboard_$Stamp.js"
    [System.IO.File]::WriteAllText($TempJs, $Combined.ToString(), [System.Text.UTF8Encoding]::new($false))
    node --check $TempJs
    if ($LASTEXITCODE -ne 0) {
        throw "node --check failed for extracted dashboard scripts: $TempJs"
    }
    Remove-Item -LiteralPath $TempJs -Force
    Write-Host "node --check passed for story_generate_dashboard.html scripts." -ForegroundColor Green
}

Write-Step "Whitespace diff check"
git diff --check
if ($LASTEXITCODE -ne 0) {
    throw "git diff --check failed"
}
Write-Host "git diff --check passed." -ForegroundColor Green

if ($Commit) {
    Write-Step "Commit"
    Invoke-Git @("add", "-A")
    Invoke-Git @("commit", "-m", "checkpoint: $Name")
    Write-Host "Local checkpoint commit created." -ForegroundColor Green
}
else {
    Write-Step "Commit suggestion"
    Write-Host "Review changes, then run:"
    Write-Host "  git add -A"
    Write-Host "  git commit -m `"checkpoint: $Name`""
}

if ($Push) {
    if (-not $Commit) {
        throw "-Push requires -Commit so a verified local checkpoint exists first."
    }
    Write-Step "Push"
    Invoke-Git @("push")
    Write-Host "Pushed current branch to remote." -ForegroundColor Green
}
else {
    Write-Step "Push policy"
    Write-Host "Not pushed. Push only after the user confirms this version is good:"
    Write-Host "  git push"
}
