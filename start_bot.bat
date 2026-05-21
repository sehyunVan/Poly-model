@echo off
:: ============================================================
:: start_bot.bat — Polymarket automated trading bot launcher
::
:: Usage:
::   Double-click, or run from cmd:  start_bot.bat
::   Or schedule via Windows Task Scheduler (see docs/deployment.md)
::
:: Modes (set in .env):
::   VIRTUAL_MODE=true   → paper trading  (default)
::   VIRTUAL_MODE=false  → live trading   (requires KEY + FUNDER in .env)
:: ============================================================

setlocal

:: ── Project root (same directory as this file) ──────────────────────────────
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

:: ── Python executable (adjust if needed) ────────────────────────────────────
set "PYTHON=python"

:: ── Log directory ───────────────────────────────────────────────────────────
set "LOG_DIR=%ROOT%\logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

:: ── Activate virtual environment if present ─────────────────────────────────
if exist "%ROOT%\.venv\Scripts\activate.bat" (
    call "%ROOT%\.venv\Scripts\activate.bat"
) else if exist "%ROOT%\venv\Scripts\activate.bat" (
    call "%ROOT%\venv\Scripts\activate.bat"
)

:: ── Change to project root ───────────────────────────────────────────────────
cd /d "%ROOT%"

echo [%date% %time%] Starting Polymarket bot... >> "%LOG_DIR%\launcher.log"

:: ── Run the bot (stdout+stderr go to main.log via RotatingFileHandler) ──────
:: The bot itself handles log rotation — we just need to launch it.
"%PYTHON%" src\main.py %*

set "EXIT_CODE=%ERRORLEVEL%"
echo [%date% %time%] Bot exited with code %EXIT_CODE% >> "%LOG_DIR%\launcher.log"

endlocal
exit /b %EXIT_CODE%
