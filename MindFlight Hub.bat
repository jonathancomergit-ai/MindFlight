@echo off
rem Double-click me: starts the MindFlight Hub and opens it in your browser. Close this window to stop.
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python -m mindflight.hub %*
pause
