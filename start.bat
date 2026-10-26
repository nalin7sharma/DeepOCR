@echo off
rem Double-click to launch DeepOCR (wraps start.ps1).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" %*
pause
