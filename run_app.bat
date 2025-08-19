@echo off
setlocal

REM Change to script directory
cd /d "%~dp0"

REM Use Python 3 if available
where python >nul 2>nul
if errorlevel 1 (
  echo Python not found. Please install Python 3.10+ from https://www.python.org/ and re-run.
  pause
  exit /b 1
)

REM Create venv if missing
if not exist ".venv" (
  echo Creating virtual environment...
  python -m venv .venv
)

REM Activate venv
call ".venv\Scripts\activate.bat"

REM Upgrade pip & install deps
python -m pip install --upgrade pip
pip install -r requirements.txt

REM Load env (optional): copy sample if .env missing
if not exist ".env" (
  copy ".env.example" ".env" >nul 2>nul
)

REM ---- Choose your app command ----
REM Streamlit:
streamlit run app.py --server.address=0.0.0.0 --server.port=8501
REM FastAPI (if you use it): 
REM uvicorn app:app --host 0.0.0.0 --port 8000 --reload

pause
