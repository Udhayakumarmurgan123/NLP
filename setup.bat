@echo off
echo ==========================================
echo Energy Consumption Forecasting - Setup
echo ==========================================
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo [X] Python is not installed. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

echo [OK] Python found
python --version
echo.

REM Create virtual environment
echo [*] Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo [*] Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo [*] Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo [*] Installing dependencies...
pip install -r requirements.txt

echo.
echo ==========================================
echo [OK] Setup Complete!
echo ==========================================
echo.
echo To run the application:
echo.
echo 1. Activate virtual environment: venv\Scripts\activate
echo 2. Run Streamlit: streamlit run app.py
echo.
echo ==========================================
pause
