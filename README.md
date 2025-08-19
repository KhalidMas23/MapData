# Your App

## Quick start
1. Install Python 3.10+ (https://www.python.org/downloads/).
2. Unzip the folder.
3. Windows: double‑click `run_app.bat`
   macOS/Linux: double‑click `run_app.sh` (or run `chmod +x run_app.sh` once, then double‑click)
4. Your browser opens to http://localhost:8501 (Streamlit) or http://localhost:8000 (FastAPI).

## Contents
- app.py – the app entrypoint
- data/ – bundled datasets (e.g., GeoPackage `mydata.gpkg`)
- requirements.txt – Python dependencies
- .env.example – sample config; copy to `.env` to override defaults

## Notes
- No admin rights required; everything installs into a local `.venv` folder next to the app.
- If you already have a Python virtualenv active, deactivate it before running.
