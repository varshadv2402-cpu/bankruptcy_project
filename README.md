# Bankruptcy Prediction System

A machine learning system that classifies companies by bankruptcy risk using financial ratio data, built around a realistic challenge: severe class imbalance (only 3.2% of companies in the dataset went bankrupt).

## Dataset
**Taiwanese Bankruptcy Prediction Dataset** — financial ratios for 6,819 companies collected from the Taiwan Economic Journal (1999–2009), via the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction). 95 financial-ratio features, no missing values, 220 bankrupt companies (3.23%).

## The Core Challenge: Class Imbalance
A model that predicts "not bankrupt" for every single company already scores **96.8% accuracy** on this dataset — while catching almost none of the actual bankruptcies. That makes raw accuracy the wrong metric to optimize here. This project instead treats **recall on the bankrupt class** (catching real risk cases) and **F1 score** as the metrics that matter, with **ROC-AUC** as the threshold-independent check on overall model quality.

## Approach
1. **EDA** — examined class balance and ranked financial ratios by correlation with bankruptcy (profitability ratios like ROA and net income, and leverage ratios like debt ratio, surfaced as the strongest signals — consistent with financial theory).
2. **Feature Selection** — combined correlation ranking with Random Forest feature importance, keeping only the 10 features that both methods agreed were predictive.
3. **Baseline Model** — plain Logistic Regression, default 0.5 decision threshold, no imbalance handling.
4. **Iterative Optimization**:
   - Applied **SMOTE** to oversample the minority (bankrupt) class during training
   - Switched to a tuned **Random Forest** to capture non-linear interactions between financial ratios
   - Tuned the **decision threshold** on the precision-recall curve (instead of the default 0.5 cutoff) to maximize F1

## Results

| Metric | Baseline (Logistic Regression) | Improved (SMOTE + Tuned Random Forest) |
|---|---|---|
| Accuracy | 96.8% | 94.4% |
| Precision | 0.50 | 0.32 |
| **Recall (bankrupt class)** | **0.14** | **0.66** |
| **F1 Score** | **0.214** | **0.433** |
| ROC-AUC | 0.917 | 0.923 |

- **Recall improved ~4.8x** (catching 29 of 44 actual bankruptcies in the test set, vs. 6 of 44 for the baseline)
- **F1 score improved ~102%** (from 0.214 to 0.433)
- Accuracy *dropped* from the baseline — expected and correct, since the baseline's high accuracy came from simply predicting the majority class most of the time
- ROC-AUC, the threshold-independent measure of model quality, held steady and slightly improved (0.917 → 0.923)

See `outputs/figures/` for the confusion matrices, ROC curves, and feature importance chart that visualize these results.

## Tech Stack
- **Language:** Python
- **ML:** Scikit-learn, imbalanced-learn (SMOTE)
- **Data:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn

## Project Structure
```
bankruptcy-prediction-system/
├── data/
│   └── bankruptcy_data.csv
├── src/
│   ├── data_loader.py        # Load and clean the dataset
│   ├── eda.py                 # Exploratory analysis and charts
│   ├── feature_selection.py  # Correlation + Random Forest importance
│   ├── train_model.py        # Baseline + improved model training
│   ├── visualize_results.py  # Confusion matrices, ROC curves, importance chart
│   └── main.py                # Runs the full pipeline end to end
├── outputs/
│   ├── figures/                # All generated charts
│   ├── bankruptcy_model.pkl   # Trained final model
│   ├── scaler.pkl              # Fitted StandardScaler
│   └── results.json            # Full metrics output
├── requirements.txt
└── README.md
```

## Getting Started
```bash
pip install -r requirements.txt
python src/main.py
```
This runs the full pipeline: loads data, generates EDA charts, selects features, trains both models, evaluates them, and saves all charts and model artifacts to `outputs/`.

## Future Improvements
- Try cost-sensitive learning with business-specific misclassification costs (a missed bankruptcy is far costlier than a false alarm)
- Add time-aware validation (the dataset spans 1999–2009; a walk-forward split would be more realistic than random splitting)
- Experiment with XGBoost/LightGBM and SHAP values for more interpretable feature attribution
