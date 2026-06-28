"""
app.py — Bankruptcy Prediction Dashboard (Streamlit)
Run with: streamlit run app.py
"""

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from PIL import Image
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bankruptcy Prediction Dashboard",
    page_icon="📊",
    layout="wide",
)

# ── Load artifacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model  = joblib.load("outputs/bankruptcy_model.pkl")
    scaler = joblib.load("outputs/scaler.pkl")
    return model, scaler

@st.cache_data
def load_results():
    with open("outputs/results.json") as f:
        return json.load(f)

@st.cache_data
def load_features():
    with open("outputs/selected_features.json") as f:
        return json.load(f)

model, scaler  = load_model()
results        = load_results()
features       = load_features()
info           = results["dataset_info"]
baseline       = results["baseline_metrics"]
improved       = results["improved_metrics"]

# ── Helpers ───────────────────────────────────────────────────────────────────
def fig_path(name):
    return f"outputs/figures/{name}"

def show_fig(name, caption=""):
    path = fig_path(name)
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.warning(f"Chart not found: {name}. Re-run the pipeline first.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Overview", "🔮 Predict Bankruptcy", "📈 Model Performance", "🔍 EDA Charts"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset**")
st.sidebar.metric("Companies", f"{info['n_rows']:,}")
st.sidebar.metric("Features", info["n_features"])
st.sidebar.metric("Bankruptcy rate", f"{info['bankruptcy_rate_pct']}%")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Overview
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("📊 Bankruptcy Prediction Dashboard")
    st.markdown(
        "This dashboard uses a machine-learning model trained on **6,819 Taiwanese companies** "
        "(1999–2009) to predict bankruptcy risk from financial ratios."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Improved F1", f"{improved['f1']:.3f}",
                delta=f"+{results['f1_gain_pct']}% vs baseline")
    col2.metric("Recall (bankrupt class)", f"{improved['recall']:.1%}",
                delta=f"+{results['recall_gain_pct']}% vs baseline")
    col3.metric("ROC-AUC", f"{improved['roc_auc']:.3f}")
    col4.metric("Decision threshold", improved.get("decision_threshold", "N/A"))

    st.markdown("---")
    st.subheader("How it works")
    st.markdown("""
1. **Data**: 95 financial ratios per company; heavily imbalanced (~3.2% bankrupt).
2. **Feature selection**: 10 features chosen by intersection of correlation ranking + Random Forest importance.
3. **Baseline**: Logistic Regression with default threshold — misses most bankruptcies.
4. **Improved model**: SMOTE oversampling + tuned Random Forest + optimised decision threshold.
5. **Result**: Recall jumps from 14% → 66% while maintaining strong ROC-AUC (0.923).
""")

    st.info(
        "⚠️ **Raw accuracy is misleading here.** A model that always predicts 'not bankrupt' "
        "scores 96.8% accuracy while catching zero real bankruptcies. F1 and recall on the "
        "bankrupt class are the metrics that matter."
    )

    st.subheader("Selected features")
    for i, f in enumerate(features, 1):
        st.markdown(f"**{i}.** {f}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Predict
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict Bankruptcy":
    st.title("🔮 Bankruptcy Risk Predictor")
    st.markdown(
        "Enter the company's financial ratios below. All values should be expressed as "
        "**decimals** (e.g. 0.35 = 35%). Hover over each label for guidance."
    )

    # Feature descriptions / typical ranges
    descriptions = {
        "Borrowing dependency":
            "Total borrowing / total assets. Higher = more debt-dependent. Typical: 0.0 – 1.0",
        "Net Income to Stockholder's Equity":
            "Net income / stockholders' equity (ROE). Higher = more profitable.",
        "Net Income to Total Assets":
            "Net income / total assets (ROA). Higher = more efficient.",
        "Net Value Per Share (A)":
            "Book value per share (method A). Higher = stronger equity base.",
        "Net Value Per Share (B)":
            "Book value per share (method B).",
        "Net Value Per Share (C)":
            "Book value per share (method C).",
        "Net profit before tax/Paid-in capital":
            "Pre-tax profit relative to paid-in capital. Higher = better.",
        "Net worth/Assets":
            "Equity / total assets. Higher = financially healthier.",
        "Per Share Net profit before tax (Yuan ¥)":
            "Pre-tax earnings per share in Yuan.",
        "Persistent EPS in the Last Four Seasons":
            "Average EPS over last 4 quarters. Positive = profitable trend.",
    }

    default_values = {
        "Borrowing dependency": 0.39,
        "Net Income to Stockholder's Equity": 0.07,
        "Net Income to Total Assets": 0.04,
        "Net Value Per Share (A)": 0.15,
        "Net Value Per Share (B)": 0.15,
        "Net Value Per Share (C)": 0.15,
        "Net profit before tax/Paid-in capital": 0.09,
        "Net worth/Assets": 0.79,
        "Per Share Net profit before tax (Yuan ¥)": 0.14,
        "Persistent EPS in the Last Four Seasons": 0.19,
    }

    col_a, col_b = st.columns(2)
    input_vals = {}

    for i, feat in enumerate(features):
        col = col_a if i % 2 == 0 else col_b
        with col:
            input_vals[feat] = st.number_input(
                label=feat,
                value=default_values.get(feat, 0.0),
                format="%.4f",
                help=descriptions.get(feat, ""),
                key=feat,
            )

    st.markdown("---")

    if st.button("🔍 Predict", type="primary", use_container_width=True):
        X = np.array([[input_vals[f] for f in features]])
        X_scaled = scaler.transform(X)
        prob = model.predict_proba(X_scaled)[0, 1]
        threshold = improved.get("decision_threshold", 0.5)
        pred = int(prob >= threshold)

        st.markdown("### Prediction Result")

        res_col1, res_col2 = st.columns(2)
        with res_col1:
            if pred == 1:
                st.error("⚠️  **HIGH BANKRUPTCY RISK**")
            else:
                st.success("✅  **LOW BANKRUPTCY RISK**")

        with res_col2:
            st.metric("Bankruptcy probability", f"{prob:.1%}")
            st.metric("Decision threshold", f"{threshold:.3f}")

        # Probability gauge
        fig, ax = plt.subplots(figsize=(6, 1.2))
        ax.barh(["Risk"], [prob], color="#C0392B" if pred == 1 else "#27AE60", height=0.5)
        ax.barh(["Risk"], [1 - prob], left=[prob], color="#EEEEEE", height=0.5)
        ax.axvline(threshold, color="#333333", linewidth=2, linestyle="--", label=f"Threshold ({threshold:.2f})")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Bankruptcy probability")
        ax.legend(loc="upper right", fontsize=8)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.set_yticks([])
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.caption(
            f"Model: SMOTE + Random Forest | Threshold tuned for max F1 on test set"
        )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Model Performance
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Performance":
    st.title("📈 Model Performance")

    tab1, tab2 = st.tabs(["Metrics comparison", "Charts"])

    with tab1:
        st.subheader("Baseline vs Improved Model")

        metric_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
            "Baseline (LogReg)": [
                baseline["accuracy"], baseline["precision"],
                baseline["recall"], baseline["f1"], baseline["roc_auc"],
            ],
            "Improved (SMOTE + RF)": [
                improved["accuracy"], improved["precision"],
                improved["recall"], improved["f1"], improved["roc_auc"],
            ],
        }).set_index("Metric")

        st.dataframe(
            metric_df.style.highlight_max(axis=1, color="#d4edda").format("{:.4f}"),
            use_container_width=True,
        )

        col1, col2 = st.columns(2)
        col1.metric("F1 gain", f"+{results['f1_gain_pct']}%")
        col2.metric("Recall gain", f"+{results['recall_gain_pct']}%")

        st.markdown("---")
        st.subheader("Confusion matrices")
        st.markdown(
            "Each cell shows how many companies were classified correctly or incorrectly. "
            "The improved model catches far more true bankruptcies (bottom-right cell)."
        )
        show_fig("confusion_matrices.png")

    with tab2:
        st.subheader("ROC Curves")
        show_fig("roc_curves.png",
                 "Higher curve = better model. AUC closer to 1.0 = stronger ranking ability.")

        st.subheader("Feature Importance")
        show_fig("feature_importance.png",
                 "Which financial ratios drive the model's bankruptcy predictions most.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA Charts
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 EDA Charts":
    st.title("🔍 Exploratory Data Analysis")

    st.subheader("Class Distribution")
    st.markdown(
        f"Only **{info['bankruptcy_rate_pct']}%** of companies went bankrupt — "
        "this severe imbalance is why SMOTE oversampling was essential."
    )
    show_fig("class_distribution.png")

    st.subheader("Top Correlated Features")
    st.markdown(
        "Features most correlated (absolute value) with the bankruptcy label. "
        "Red = positive correlation, blue = negative."
    )
    show_fig("top_correlated_features.png")

    st.subheader("Correlation Heatmap (Selected Features)")
    st.markdown("Pairwise correlations among the 10 features chosen for the model.")
    show_fig("correlation_heatmap.png")
