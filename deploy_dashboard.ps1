param(
    [string]$CommitMessage = 'Refresh GitHub Pages dashboard',
    [switch]$SkipPush
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Write-Step {
    param([string]$Message)
    Write-Host "[deploy] $Message" -ForegroundColor Cyan
}

if (-not (Test-Path '.git')) {
    throw 'Run this script from the repository root (folder containing .git).'
}

$statusLines = git status --porcelain
if ($statusLines) {
    throw "Working tree is not clean. Commit/stash your changes first.`n$statusLines"
}

$branch = (git rev-parse --abbrev-ref HEAD).Trim()
if ($branch -ne 'main') {
    throw "Current branch is '$branch'. Switch to 'main' before deploy."
}

$pythonExe = Join-Path '.venv\Scripts' 'python.exe'
if (-not (Test-Path $pythonExe)) {
    $pythonExe = 'python'
}

Write-Step 'Regenerating static dashboard from Data workbook'
& $pythonExe generate_static_dashboard.py

if (-not (Test-Path 'static_dashboard.html')) {
    throw 'Expected static_dashboard.html was not generated.'
}

Write-Step 'Syncing static_dashboard.html into publish entrypoint index.html'
Copy-Item -Path 'static_dashboard.html' -Destination 'index.html' -Force

$hashStatic = (Get-FileHash 'static_dashboard.html').Hash
$hashIndex = (Get-FileHash 'index.html').Hash
if ($hashStatic -ne $hashIndex) {
    throw 'Safety check failed: index.html does not match static_dashboard.html after sync.'
}

$hasIndexChanges = git status --porcelain -- index.html
if (-not $hasIndexChanges) {
    Write-Step 'No dashboard changes detected in index.html. Nothing to commit.'
    exit 0
}

Write-Step 'Committing publish artifact (index.html only)'
git add index.html
git commit -m $CommitMessage

if ($SkipPush) {
    Write-Step 'SkipPush set. Commit created locally; push was skipped.'
    exit 0
}

Write-Step 'Pushing main to origin'
git push origin main

Write-Step 'Deploy complete. GitHub Pages will update shortly.'