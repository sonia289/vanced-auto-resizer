#!/bin/bash
cd "$(dirname "$0")"
if [ ! -d .venv ]; then python3 -m venv .venv; fi
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
