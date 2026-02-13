#!/bin/sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT_DIR"

# Compute metrics once on server start (uses cache if available).
python scripts/compute_metrics.py

# Start Streamlit app.
exec streamlit run main.py --server.address=0.0.0.0 --server.port=8501
