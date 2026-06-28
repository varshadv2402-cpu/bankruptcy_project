#!/bin/bash
# Run the full bankruptcy prediction pipeline

set -e

echo "=== Bankruptcy Prediction Pipeline ==="
echo ""

# Check Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found. Please install Python 3.8+."
    exit 1
fi

# Install dependencies if needed
echo "Checking dependencies..."
pip3 install -q pandas numpy scikit-learn imbalanced-learn matplotlib seaborn joblib 2>/dev/null || true

# Run from the project root so relative paths work
cd "$(dirname "$0")"

echo "Running pipeline..."
echo ""
python3 src/main.py

echo ""
echo "Done! Outputs saved to outputs/"
echo "  - outputs/results.json         (metrics)"
echo "  - outputs/bankruptcy_model.pkl (trained model)"
echo "  - outputs/scaler.pkl           (feature scaler)"
echo "  - outputs/figures/             (all charts)"
