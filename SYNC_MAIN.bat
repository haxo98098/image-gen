@echo off
setlocal
title Image Gen - Sync Main
cd /d "%~dp0"

where git >nul 2>nul
if errorlevel 1 (
    echo ERROR: Git is not installed or is not in PATH.
    pause
    exit /b 1
)

git rev-parse --is-inside-work-tree >nul 2>nul
if errorlevel 1 (
    echo ERROR: Run this file inside your cloned image-gen repository.
    echo Clone once with: git clone https://github.com/haxo98098/image-gen.git
    pause
    exit /b 1
)

echo.
echo Updating from origin/main...
echo WARNING: Tracked local edits and untracked non-ignored files will be discarded.
echo Ignored files such as local-config.json, outputs, virtual environments and models stay untouched.
echo.

git fetch origin main
if errorlevel 1 goto failed

git checkout -f -B main origin/main
if errorlevel 1 goto failed

git reset --hard origin/main
if errorlevel 1 goto failed

git clean -fd
if errorlevel 1 goto failed

echo.
echo SUCCESS: Your local main now matches origin/main.
git log -1 --format="Current commit: %%h - %%s"
echo.
pause
exit /b 0

:failed
echo.
echo ERROR: Sync failed. See Git errors above.
pause
exit /b 1
