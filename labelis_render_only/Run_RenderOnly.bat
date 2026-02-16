@echo off
setlocal

REM Optional helper for Windows users. Edit to activate your environment if needed.
REM Example:
REM   call conda activate labelis-render-only

python "%~dp0render_only_app.py"
pause
