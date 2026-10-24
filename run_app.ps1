<#
.SYNOPSIS
    IMU Gesture Controller PowerShell Launcher
    Supervisor: Dr. Khalid Ali Abbas
    University of Basrah · College of Engineering · Computer Engineering Department
#>

$ProjectDir = $PSScriptRoot
Set-Location $ProjectDir

$VenvPython = Join-Path $ProjectDir ".venv\Scripts\python.exe"

Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host " IMU Gesture Controller - University of Basrah" -ForegroundColor White
Write-Host " Supervisor: Dr. Khalid Ali Abbas" -ForegroundColor White
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host " Project Root : $ProjectDir"
Write-Host " Interpreter  : $VenvPython"
Write-Host "========================================================================" -ForegroundColor Cyan

if (-not (Test-Path $VenvPython)) {
    Write-Host "[ERROR] Virtual environment not found at:" -ForegroundColor Red
    Write-Host "  $VenvPython" -ForegroundColor Yellow
    Write-Host "`nPlease initialize the virtual environment:" -ForegroundColor Yellow
    Write-Host "  py -3.12 -m venv .venv"
    Write-Host "  .\.venv\Scripts\python.exe -m pip install -r requirements.txt`n"
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "Starting IMU Gesture Controller in Replay & Dry Run mode...`n" -ForegroundColor Green
& $VenvPython (Join-Path $ProjectDir "main.py")

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n[ERROR] Application exited with code $LASTEXITCODE" -ForegroundColor Red
    Read-Host "Press Enter to exit"
}
