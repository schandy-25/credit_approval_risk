# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from src.modeling import train_model
from src.fairness import demographic_parity, equalized_odds
from src.data_utils import clean_data
from src.config import NUMERIC_COLS, CATEGORICAL_COLS, SENSITIVE_COL
from src.explainability import compute_shap_values

shap.initjs()

@st.cache_resource
def get_trained_model():
    clf, eval_data = train_model()
    return clf, eval_data

def main():
    st.set_page_config(page_title="Credit Risk Fairness Dashboard", layout="wide")
    st.title("Credit Approval Risk Model – Fairness & Governance Dashboard")

    clf, (X_test, y_test, s_test, y_pred, y_proba, metrics) = get_trained_model()

    tab_perf, tab_fair, tab_shap, tab_whatif = st.tabs([
        "📈 Model Performance",
        "⚖️ Fairness Metrics",
        "🔍 SHAP Explainability",
        "🧪 What-if Analysis"
    ])

    # 1) Model Performance
    with tab_perf:
        st.subheader("Model Performance Overview")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        with col2:
            st.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

        st.write("### Confusion Matrix")
        cm = metrics["confusion_matrix"]
        cm_df = pd.DataFrame(cm,
                             index=["Actual 0", "Actual 1"],
                             columns=["Pred 0", "Pred 1"])
        st.dataframe(cm_df)

        st.write("### Score Distribution (P(Approved=1))")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(y_proba, bins=20)
        ax.set_xlabel("Predicted Probability of Approval")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    # 2) Fairness tab
    with tab_fair:
        st.subheader("Fairness Metrics (Demographic Parity & Equalized Odds)")

        df_eval = X_test.copy()
        df_eval["y_true"] = y_test.values
        df_eval["y_pred"] = y_pred
        df_eval[SENSITIVE_COL] = s_test.values

        st.write(f"Sensitive attribute: **{SENSITIVE_COL}**")

        st.write("### Approval Rates by Group (Demographic Parity)")
        rates, dp_diff = demographic_parity(df_eval["y_pred"], df_eval[SENSITIVE_COL])
        rates_df = pd.DataFrame(
            {"group": list(rates.keys()), "approval_rate": list(rates.values())}
        )
        st.dataframe(rates_df)
        st.write(f"**DP difference (max approval rate gap):** {dp_diff:.3f}")

        st.write("### Equalized Odds (TPR / FPR by Group)")
        tpr, fpr, tpr_gap, fpr_gap = equalized_odds(
            df_eval["y_true"], df_eval["y_pred"], df_eval[SENSITIVE_COL]
        )
        eo_df = pd.DataFrame({
            "group": list(tpr.keys()),
            "TPR": list(tpr.values()),
            "FPR": list(fpr.values())
        })
        st.dataframe(eo_df)
        st.write(f"**TPR gap:** {tpr_gap:.3f}")
        st.write(f"**FPR gap:** {fpr_gap:.3f}")

        st.info(
            "Smaller gaps indicate better fairness under Demographic Parity and Equalized Odds.\n"
            "This tab acts as a 'monitoring report' for fairness compliance."
        )

    # 3) SHAP Explainability
    with tab_shap:
        st.subheader("Global & Local Explainability (SHAP)")

        df_full = clean_data()
        st.write("We sample a subset for SHAP to keep it responsive.")
        sample = df_full.sample(n=min(200, len(df_full)), random_state=42)
        feature_cols = NUMERIC_COLS + CATEGORICAL_COLS
        X_sample = sample[feature_cols]

        
        shap_vals, expected_value, X_trans = compute_shap_values(clf, X_sample)

        st.write("### Global Feature Importance (SHAP Summary Plot)")
        # Transform features for SHAP summary plot
        preprocess = clf.named_steps["preprocess"]
        X_trans = preprocess.transform(X_sample)

        fig, ax = plt.subplots(figsize=(8, 5))
        shap.summary_plot(shap_vals, X_trans, show=False)
        st.pyplot(fig)

        st.write("### Single Example Explanation (Force Plot)")
        idx = st.slider("Pick an index from the sample", 0, len(X_sample) - 1, 0)
        x_row = X_trans[idx:idx+1]
        shap_vals_row = shap_vals[idx:idx+1]

        fig2, ax2 = plt.subplots(figsize=(8, 3))
        shap.force_plot(expected_value, shap_vals_row, matplotlib=True, show=False)
        st.pyplot(fig2)

        st.caption("This tab acts as an explainability surface for governance and model review.")

    # 4) What-if Analysis
    with tab_whatif:
        st.subheader("What-if Analysis – Individual-level Governance")

        df = clean_data()

        st.write(
            "Adjust applicant features and see how the approval probability changes. "
            "This can be used as a compliance slice to understand decision boundaries."
        )

        base = df[NUMERIC_COLS].median().to_dict()
        user_input = {}

        st.write("### Numeric Features")
        for col in NUMERIC_COLS:
            minv = float(df[col].quantile(0.05))
            maxv = float(df[col].quantile(0.95))
            user_input[col] = st.slider(
                col,
                min_value=round(minv, 2),
                max_value=round(maxv, 2),
                value=float(round(base[col], 2))
            )

        st.write("### Categorical Features")
        for col in CATEGORICAL_COLS:
            options = df[col].value_counts().index.tolist()
            user_input[col] = st.selectbox(col, options, index=0)

        input_df = pd.DataFrame([user_input])

        prob = clf.predict_proba(input_df)[0, 1]
        pred = int(prob >= 0.5)

        st.write("### Model Output")
        st.write(f"**Approval probability:** {prob:.3f}")
        st.write(f"**Decision:** {'✅ Approved (1)' if pred == 1 else '❌ Rejected (0)'}")

        st.info(
            "You can use this panel to explore counterfactuals: "
            "e.g., how changes in income, debt, or employment affect credit decisions."
        )

if __name__ == "__main__":
    main()
