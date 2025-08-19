#!/usr/bin/env bash
set -euo pipefail

# cd to script directory
cd "$(dirname "$0")"

# Check python
if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3.10+ not found. Install from https://www.python.org/ or via Homebrew/apt."
  exit 1
fi

# Create venv if missing
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi

# Activate venv
# shellcheck disable=SC1091
source .venv/bin/activate

# Upgrade pip & install deps
python -m pip install --upgrade pip
pip install -r requirements.txt

# Copy sample env if missing
[ -f ".env" ] || cp ".env.example" ".env"

# ---- Choose your app command ----
# Streamlit:
streamlit run app.py --server.address=0.0.0.0 --server.port=8501
# FastAPI (if you use it):
# uvicorn app:app --host 0.0.0.0 --port 8000 --reload
