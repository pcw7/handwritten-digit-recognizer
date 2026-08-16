@echo off
REM Double-click this file in Windows Explorer to launch the digit recognizer GUI.
setlocal
cd /d "%~dp0"

if not exist "digit_model.joblib" (
    echo Model file not found. Please run train_model.py first.
    pause
    exit /b 1
)

REM Try pythonw first (no console window). Fall back to python if pythonw is missing.
where pythonw >nul 2>nul
if %errorlevel%==0 (
    start "" pythonw draw_and_recognize.py
) else (
    python draw_and_recognize.py
    if errorlevel 1 pause
)
