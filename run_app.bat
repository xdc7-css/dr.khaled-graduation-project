@echo off
setlocal

:: =========================================================================
:: IMU Gesture Controller Launcher
:: Academic Project: Machine Learning Hand Gesture Recognition for PowerPoint Control
:: Supervisor: Dr. Khalid Ali Abbas
:: University of Basrah - College of Engineering - Computer Engineering Department
:: =========================================================================

set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

set "VENV_PYTHON=%PROJECT_DIR%.venv\Scripts\python.exe"

echo ========================================================================
echo  IMU Gesture Controller - University of Basrah
echo  Supervisor: Dr. Khalid Ali Abbas
echo ========================================================================
echo  Project Root : %PROJECT_DIR%
echo  Interpreter  : %VENV_PYTHON%
echo ========================================================================

if not exist "%VENV_PYTHON%" (
    echo [ERROR] Virtual environment interpreter not found at:
    echo   "%VENV_PYTHON%"
    echo.
    echo Please create the virtual environment and install dependencies:
    echo   py -3.12 -m venv .venv
    echo   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

if not exist "%PROJECT_DIR%main.py" (
    echo [ERROR] main.py entry point not found in "%PROJECT_DIR%".
    echo.
    pause
    exit /b 1
)

echo Starting IMU Gesture Controller in Replay and Dry Run mode...
echo.

"%VENV_PYTHON%" "%PROJECT_DIR%main.py"
set "EXIT_CODE=%ERRORLEVEL%"

if %EXIT_CODE% neq 0 (
    echo.
    echo ========================================================================
    echo [APPLICATION EXITED WITH ERROR CODE %EXIT_CODE%]
    echo Review the error message above before closing.
    echo ========================================================================
    pause
)

exit /b %EXIT_CODE%
