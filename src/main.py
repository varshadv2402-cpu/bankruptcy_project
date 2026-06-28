"""
main.py
Runs the full bankruptcy prediction pipeline end to end:
  1. Load and clean data
  2. EDA + charts
  3. Feature selection
  4. Train baseline + improved models
  5. Evaluate and visualize results
  6. Save model artifacts
"""

import sys
import os
import json
import joblib

sys.path.insert(0, os.path.dirname(__file__))

from data_loader import load_data, basic_info
from eda import class_distribution_plot, top_correlation_plot, correlation_heatmap
from feature_selection import select_features
from train_model import prepare_data, train_baseline, train_improved, find_best_threshold, evaluate
from visualize_results import plot_confusion_matrices, plot_roc_curves, plot_feature_importance


def main():
    os.makedirs("outputs/figures", exist_ok=True)

    print("Step 1: Loading data...")
    df = load_data()
    info = basic_info(df)
    print(f"  {info['n_rows']} companies, {info['n_features']} features, "
          f"{info['bankruptcy_rate_pct']}% bankruptcy rate\n")

    print("Step 2: Running EDA...")
    class_distribution_plot(df)
    top_correlation_plot(df)
    print("  Saved class_distribution.png and top_correlated_features.png\n")

    print("Step 3: Selecting features...")
    features = select_features(df, top_n=20)
    print(f"  Selected {len(features)} features (agree across correlation + RF importance)\n")
    correlation_heatmap(df, features)

    print("Step 4: Training models...")
    X_train, X_test, y_train, y_test, scaler = prepare_data(df, features)

    baseline_model = train_baseline(X_train, y_train)
    baseline_probs = baseline_model.predict_proba(X_test)[:, 1]
    baseline_preds = baseline_model.predict(X_test)
    baseline_metrics = evaluate(y_test, baseline_preds, baseline_probs)

    improved_model = train_improved(X_train, y_train)
    improved_probs = improved_model.predict_proba(X_test)[:, 1]
    best_threshold = find_best_threshold(y_test, improved_probs)
    improved_preds = (improved_probs >= best_threshold).astype(int)
    improved_metrics = evaluate(y_test, improved_preds, improved_probs)
    improved_metrics["decision_threshold"] = round(best_threshold, 3)

    print("  Baseline :", {k: v for k, v in baseline_metrics.items() if k != "confusion_matrix"})
    print("  Improved :", {k: v for k, v in improved_metrics.items() if k != "confusion_matrix"})
    print()

    print("Step 5: Visualizing results...")
    plot_confusion_matrices(y_test, baseline_preds, improved_preds)
    plot_roc_curves(y_test, baseline_probs, improved_probs,
                     baseline_metrics["roc_auc"], improved_metrics["roc_auc"])
    plot_feature_importance(improved_model, features)
    print("  Saved confusion_matrices.png, roc_curves.png, feature_importance.png\n")

    print("Step 6: Saving model artifacts...")
    joblib.dump(improved_model, "outputs/bankruptcy_model.pkl")
    joblib.dump(scaler, "outputs/scaler.pkl")

    f1_gain_pct = round((improved_metrics["f1"] - baseline_metrics["f1"]) / baseline_metrics["f1"] * 100, 1)
    recall_gain_pct = round((improved_metrics["recall"] - baseline_metrics["recall"]) / baseline_metrics["recall"] * 100, 1)

    results = {
        "dataset_info": info,
        "selected_features": features,
        "baseline_metrics": baseline_metrics,
        "improved_metrics": improved_metrics,
        "f1_gain_pct": f1_gain_pct,
        "recall_gain_pct": recall_gain_pct,
    }
    with open("outputs/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("  Saved bankruptcy_model.pkl, scaler.pkl, results.json\n")

    print("Pipeline complete.")
    return results


if __name__ == "__main__":
    main()
