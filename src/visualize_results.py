"""
visualize_results.py
Generates the model-evaluation charts: confusion matrices (baseline vs.
improved), ROC curves, and feature importance from the final model.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, ConfusionMatrixDisplay

FIG_DIR = "outputs/figures"


def plot_confusion_matrices(y_test, baseline_preds, improved_preds):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, preds, title in zip(
        axes, [baseline_preds, improved_preds],
        ["Baseline (Logistic Regression)", "Improved (SMOTE + Tuned RF)"]
    ):
        ConfusionMatrixDisplay.from_predictions(
            y_test, preds, display_labels=["Not Bankrupt", "Bankrupt"],
            cmap="Blues", colorbar=False, ax=ax
        )
        ax.set_title(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/confusion_matrices.png", dpi=150)
    plt.close()


def plot_roc_curves(y_test, baseline_probs, improved_probs, baseline_auc, improved_auc):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for probs, auc, label, color in [
        (baseline_probs, baseline_auc, "Baseline (LogReg)", "#888888"),
        (improved_probs, improved_auc, "Improved (SMOTE + RF)", "#C0392B"),
    ]:
        fpr, tpr, _ = roc_curve(y_test, probs)
        ax.plot(fpr, tpr, label=f"{label} (AUC = {auc:.3f})", color=color, linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="lightgray", label="Random guess")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve: Baseline vs. Improved Model", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/roc_curves.png", dpi=150)
    plt.close()


def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(
        [feature_names[i] for i in order][::-1],
        [importances[i] for i in order][::-1],
        color="#1F4E79",
    )
    ax.set_title("Feature Importance — Final Model", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Importance")
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.4, right=0.95, top=0.93, bottom=0.08)
    plt.savefig(f"{FIG_DIR}/feature_importance.png", dpi=150)
    plt.close()
