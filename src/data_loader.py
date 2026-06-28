"""
data_loader.py
Loads and cleans the raw bankruptcy dataset.

Dataset: Taiwanese Bankruptcy Prediction Dataset
Source: Taiwan Economic Journal (1999-2009), via UCI Machine Learning Repository
https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction
"""

import pandas as pd


def load_data(path="data/bankruptcy_data.csv"):
    """Load the raw CSV and clean up column names."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def basic_info(df):
    """Return a quick summary dict used for the EDA report."""
    target_col = "Bankrupt?"
    info = {
        "n_rows": df.shape[0],
        "n_features": df.shape[1] - 1,
        "missing_values": int(df.isnull().sum().sum()),
        "class_counts": df[target_col].value_counts().to_dict(),
        "bankruptcy_rate_pct": round(df[target_col].mean() * 100, 2),
    }
    return info


if __name__ == "__main__":
    df = load_data()
    info = basic_info(df)
    print("Dataset shape:", df.shape)
    for k, v in info.items():
        print(f"  {k}: {v}")
