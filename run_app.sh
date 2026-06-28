#!/bin/bash
# Launch the Bankruptcy Prediction Dashboard

set -e

echo "=== Bankruptcy Prediction Dashboard ==="
echo ""

# Install streamlit if needed
pip3 install -q streamlit pillow 2>/dev/null || true

# Run from project root
cd "$(dirname "$0")"

echo "Starting app at http://localhost:8501"
echo "Press Ctrl+C to stop."
echo ""

streamlit run app.py
