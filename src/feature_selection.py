"""
feature_selection.py
Identifies the most predictive financial-ratio features for bankruptcy risk,
combining two methods for robustness: correlation with the target, and
feature importance from a tree-based model.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

TARGET = "Bankrupt?"


def correlation_ranking(df, top_n=20):
    """Rank features by absolute correlation with the target."""
    corr = df.corr(numeric_only=True)[TARGET].drop(TARGET)
    return corr.abs().sort_values(ascending=False).head(top_n).index.tolist()


def importance_ranking(df, top_n=20, random_state=42):
    """Rank features by Random Forest feature importance (captures non-linear signal
    that a simple correlation ranking can miss)."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    rf = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    return importances.sort_values(ascending=False).head(top_n).index.tolist()


def select_features(df, top_n=20):
    """
    Combine correlation ranking and importance ranking: a feature that shows up
    in BOTH lists is a more trustworthy signal than one that only looks good
    under a single method. Falls back to the union if the intersection is small.
    """
    corr_feats = set(correlation_ranking(df, top_n))
    importance_feats = set(importance_ranking(df, top_n))
    combined = corr_feats & importance_feats

    if len(combined) < 8:
        combined = corr_feats | importance_feats

    return sorted(combined)


if __name__ == "__main__":
    from data_loader import load_data
    df = load_data()
    selected = select_features(df, top_n=20)
    print(f"Selected {len(selected)} features:")
    for f in selected:
        print(" -", f)
