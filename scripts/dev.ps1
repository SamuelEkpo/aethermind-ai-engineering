# Start AetherMind API (no reload — stable on Windows)
# Usage:  .\scripts\dev.ps1
#         .\scripts\dev.ps1 -Reload

param(
    [switch]$Reload
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$python = Join-Path $PWD ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Error "No .venv found. Create one and: pip install -e `".[dev,ml]`""
}

if ($Reload) {
    Write-Host "Starting with reload (apps + packages only)..."
    & $python -m uvicorn apps.api.main:app --reload --reload-dir apps --reload-dir packages --host 0.0.0.0 --port 8000
} else {
    Write-Host "Starting stable server (no reload) on http://0.0.0.0:8000 ..."
    & $python -m uvicorn apps.api.main:app --host 0.0.0.0 --port 8000
}
