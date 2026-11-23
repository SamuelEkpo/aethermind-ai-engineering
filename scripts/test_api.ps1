# Hit health / model / generate endpoints.
# Run this in a SECOND terminal while the API is up.
# Usage:  .\scripts\test_api.ps1

$ErrorActionPreference = "Stop"
$base = "http://127.0.0.1:8000"

Write-Host "GET $base/health"
try {
    $h = Invoke-RestMethod -Uri "$base/health" -Method Get
    $h | ConvertTo-Json -Compress
} catch {
    Write-Host "FAILED: $_"
    Write-Host "Is the server running? Start with: .\scripts\dev.ps1"
    exit 1
}

Write-Host "`nGET $base/v1/model"
try {
    $m = Invoke-RestMethod -Uri "$base/v1/model" -Method Get
    $m | ConvertTo-Json
} catch {
    $resp = $_.Exception.Response
    if ($resp -and [int]$resp.StatusCode -eq 503) {
        Write-Host "503 — model not loaded. Train first:"
        Write-Host "  .\.venv\Scripts\python.exe -m training.scripts.train_lm --config tiny --steps 200 --batch-size 4 --seq-len 48"
    } else {
        Write-Host "FAILED: $_"
    }
}

Write-Host "`nPOST $base/v1/generate"
$body = @{ prompt = "def "; max_new_tokens = 30; temperature = 0.8 } | ConvertTo-Json
try {
    $g = Invoke-RestMethod -Uri "$base/v1/generate" -Method Post -Body $body -ContentType "application/json"
    $g | ConvertTo-Json
} catch {
    $resp = $_.Exception.Response
    if ($resp -and [int]$resp.StatusCode -eq 503) {
        Write-Host "503 — model not loaded (train a checkpoint first)."
    } else {
        Write-Host "FAILED: $_"
    }
}
