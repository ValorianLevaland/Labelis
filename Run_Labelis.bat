@echo off
REM Optional convenience launcher for Windows + Conda.
REM Edit CONDA_BAT path if needed, or just run: python -m labelis

set ENV_NAME=labelis

REM Try to use conda from PATH
where conda >nul 2>nul
if %ERRORLEVEL%==0 (
    call conda activate %ENV_NAME%
    python -m labelis
    pause
    exit /b
)

echo Conda not found in PATH. Please open an Anaconda Prompt and run:
echo   conda activate %ENV_NAME%
echo   python -m labelis
pause
