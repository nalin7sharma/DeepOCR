# DeepOCR one-click launcher (Windows PowerShell).
#   .\start.ps1            build the UI if needed, start the server, open the browser
#   .\start.ps1 -Rebuild   force a fresh frontend build
#   .\start.ps1 -Port 8080
param([int]$Port = 8000, [switch]$Rebuild, [switch]$NoBrowser)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

Write-Host "DeepOCR launcher" -ForegroundColor Cyan

$python = (Get-Command python -ErrorAction SilentlyContinue)
if (-not $python) { Write-Host "Python 3.10+ is required (python.org)." -ForegroundColor Red; exit 1 }

python -c "import fastapi, uvicorn, fitz, cv2, torch" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing Python dependencies (web-requirements.txt)..." -ForegroundColor Yellow
    python -m pip install -r web-requirements.txt
    if ($LASTEXITCODE -ne 0) { Write-Host "Dependency installation failed." -ForegroundColor Red; exit 1 }
}

if ($Rebuild -or -not (Test-Path "web\dist\index.html")) {
    $npm = (Get-Command npm -ErrorAction SilentlyContinue)
    if (-not $npm) { Write-Host "Node.js 20+ is needed once to build the UI (nodejs.org)." -ForegroundColor Red; exit 1 }
    Write-Host "Building the web interface..." -ForegroundColor Yellow
    Push-Location web
    if (-not (Test-Path "node_modules")) { npm ci --no-audit --no-fund }
    npm run build
    $built = $LASTEXITCODE
    Pop-Location
    if ($built -ne 0) { Write-Host "Frontend build failed." -ForegroundColor Red; exit 1 }
}

foreach ($model in "cnn_bilstm", "crnn") {
    if (-not (Test-Path "checkpoints\$model\best_model.pth")) {
        Write-Host "Note: no trained $model checkpoint (python scripts\train.py --model $model). PaddleOCR/TrOCR still work." -ForegroundColor DarkYellow
    }
}

$url = "http://127.0.0.1:$Port"
Write-Host "Starting DeepOCR at $url  (Ctrl+C to stop)" -ForegroundColor Green
if (-not $NoBrowser) {
    Start-Job -ScriptBlock {
        param($u)
        for ($i = 0; $i -lt 60; $i++) {
            try { Invoke-WebRequest "$u/api/v1/health/live" -UseBasicParsing -TimeoutSec 2 | Out-Null; Start-Process $u; break } catch { Start-Sleep -Seconds 1 }
        }
    } -ArgumentList $url | Out-Null
}
python -m uvicorn api.main:app --host 127.0.0.1 --port $Port
