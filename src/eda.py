"""
eda.py
Exploratory data analysis and visualizations for the bankruptcy dataset.
Saves all charts to outputs/figures/.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

FIG_DIR = "outputs/figures"
TARGET = "Bankrupt?"


def class_distribution_plot(df):
    """Bar chart showing severe class imbalance — motivates the modeling choices later."""
    os.makedirs(FIG_DIR, exist_ok=True)
    counts = df[TARGET].value_counts().sort_index()
    labels = ["Not Bankrupt", "Bankrupt"]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(labels, counts.values, color=["#1F4E79", "#C0392B"])
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 60, f"{val:,}",
                 ha="center", fontsize=11, fontweight="bold")
    ax.set_title("Class Distribution: Bankrupt vs. Not Bankrupt", fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of Companies")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/class_distribution.png", dpi=150)
    plt.close()


def top_correlation_plot(df, top_n=15):
    """Bar chart of the features most correlated (absolute value) with bankruptcy."""
    corr = df.corr(numeric_only=True)[TARGET].drop(TARGET)
    top_corr = corr.abs().sort_values(ascending=False).head(top_n)
    signed = corr.loc[top_corr.index]

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ["#C0392B" if v > 0 else "#1F4E79" for v in signed.values]
    ax.barh(signed.index[::-1], signed.values[::-1], color=colors[::-1])
    ax.set_title(f"Top {top_n} Features Correlated with Bankruptcy", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Correlation with Bankruptcy Label")
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.5, right=0.95, top=0.93, bottom=0.08)
    plt.savefig(f"{FIG_DIR}/top_correlated_features.png", dpi=150)
    plt.close()
    return top_corr


def correlation_heatmap(df, columns):
    """Heatmap of pairwise correlation among the top selected features."""
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(df[columns].corr(), cmap="coolwarm", center=0, annot=False, ax=ax)
    ax.set_title("Correlation Heatmap — Top Selected Features", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/correlation_heatmap.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    from data_loader import load_data
    df = load_data("../data/bankruptcy_data.csv") if not os.path.exists("data/bankruptcy_data.csv") else load_data()
    class_distribution_plot(df)
    top_corr = top_correlation_plot(df)
    print("Top correlated features:\n", top_corr)
