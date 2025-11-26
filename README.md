# Credit Approval Risk Model – Fairness & Governance Dashboard

This project uses the **UCI Credit Approval** dataset (via Kaggle) to build a
credit approval risk model with a full **fairness audit** and **governance dashboard**.

- Dataset: [UCI Credit Approval Data (Kaggle)](https://www.kaggle.com/datasets/anthonypm/uci-credit-approval-data)
- Target: `Approved` (1 = approved, 0 = not approved)
- Sensitive attribute: `Gender`

## Features

- Binary classification model (RandomForest + scikit-learn Pipeline)
- Fairness metrics:
  - Demographic Parity (DP)
  - Equalized Odds (EO)
- Explainability with SHAP (global + local)
- Streamlit dashboard:
  - Model Performance
  - Fairness Metrics
  - SHAP Explainability
  - What-if Analysis (sliders and dropdowns)

## Run

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# place the Kaggle CSV as:
# data/uci_credit_approval.csv

streamlit run streamlit_app.py
