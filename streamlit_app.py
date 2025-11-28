# streamlit_app.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import streamlit as st
import scipy.sparse as sp

from src.modeling import train_model
from src.fairness import demographic_parity, equalized_odds
from src.data_utils import clean_data
from src.config import NUMERIC_COLS, CATEGORICAL_COLS, SENSITIVE_COL
from src.explainability import compute_shap_values

# ---------- Page & styling ----------

st.set_page_config(
    page_title="Credit Approval Fairness Dashboard",
    layout="wide",
)

shap.initjs()

# Simple custom CSS for a cleaner UI
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #ffffff 40%, #f5f7fa 100%);
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem 1.5rem;
        border-radius: 0.75rem;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.08);
        border: 1px solid #e5e7eb;
    }
    .section-title {
        font-weight: 600;
        font-size: 1.15rem;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Cached model training ----------

@st.cache_resource
def get_trained_model():
    clf, eval_data = train_model()
    return clf, eval_data


def main():
    st.markdown(
        "<h1 style='margin-bottom:0'>💳 Credit Approval – Fairness & Explainability</h1>"
        "<p style='color:#4b5563;margin-top:0.25rem;'>"
        "Interactive governance dashboard for monitoring model performance, fairness, and explanations."
        "</p>",
        unsafe_allow_html=True,
    )

    clf, (X_test, y_test, s_test, y_pred, y_proba, metrics) = get_trained_model()

    tab_perf, tab_fair, tab_shap, tab_whatif = st.tabs(
        ["📈 Performance", "⚖️ Fairness", "🔍 Explainability", "🧪 What-if Analysis"]
    )

    # ---------- 1) Performance ----------
    with tab_perf:
        st.markdown("<div class='section-title'>Model Performance Overview</div>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("ROC–AUC", f"{metrics['roc_auc']:.3f}")
            st.markdown("</div>", unsafe_allow_html=True)
        with c3:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Test Sample Size", f"{len(X_test)}")
            st.markdown("</div>", unsafe_allow_html=True)

        st.write("### Confusion Matrix")
        cm = metrics["confusion_matrix"]
        cm_df = pd.DataFrame(
            cm,
            index=["Actual 0", "Actual 1"],
            columns=["Pred 0", "Pred 1"],
        )
        st.dataframe(cm_df, use_container_width=True)

        st.write("### Score Distribution (P(Approved = 1))")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(y_proba, bins=20, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Predicted Probability of Approval")
        ax.set_ylabel("Count")
        st.pyplot(fig, use_container_width=True)

    # ---------- 2) Fairness ----------
    with tab_fair:
        st.markdown("<div class='section-title'>Fairness Metrics</div>", unsafe_allow_html=True)

        df_eval = X_test.copy()
        df_eval["y_true"] = y_test.values
        df_eval["y_pred"] = y_pred
        df_eval[SENSITIVE_COL] = s_test.values

        st.write(f"**Sensitive attribute:** `{SENSITIVE_COL}`")

        col_a, col_b = st.columns(2)

        with col_a:
            st.write("#### Demographic Parity")
            rates, dp_diff = demographic_parity(df_eval["y_pred"], df_eval[SENSITIVE_COL])
            rates_df = pd.DataFrame(
                {"group": list(rates.keys()), "approval_rate": list(rates.values())}
            )
            st.dataframe(rates_df, use_container_width=True)
            st.info(f"DP difference (max approval rate gap): **{dp_diff:.3f}**")

        with col_b:
            st.write("#### Equalized Odds")
            tpr, fpr, tpr_gap, fpr_gap = equalized_odds(
                df_eval["y_true"], df_eval["y_pred"], df_eval[SENSITIVE_COL]
            )
            eo_df = pd.DataFrame({
                "group": list(tpr.keys()),
                "TPR": list(tpr.values()),
                "FPR": list(fpr.values()),
            })
            st.dataframe(eo_df, use_container_width=True)
            st.info(f"TPR gap: **{tpr_gap:.3f}**, FPR gap: **{fpr_gap:.3f}**")

        st.caption(
            "Smaller gaps indicate better fairness under Demographic Parity and Equalized Odds."
        )

    # ---------- 3) SHAP Explainability ----------
    with tab_shap:
        st.markdown("<div class='section-title'>Global & Local Explainability (SHAP)</div>", unsafe_allow_html=True)

        df_full = clean_data()
        st.write("We sample a subset for SHAP to keep it responsive.")
        sample = df_full.sample(n=min(200, len(df_full)), random_state=42)
        feature_cols = NUMERIC_COLS + CATEGORICAL_COLS
        X_sample = sample[feature_cols]

        shap_vals, expected_value, X_trans = compute_shap_values(clf, X_sample)

        # Handle sparse X_trans
        if sp.issparse(X_trans):
            X_trans_plot = X_trans.toarray()
        else:
            X_trans_plot = X_trans

        st.write("### Global Feature Importance (SHAP Summary Plot)")
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.summary_plot(shap_vals, X_trans_plot, show=False)
        st.pyplot(fig, use_container_width=True)

        st.write("### Local Explanation for a Single Applicant")
        idx = st.slider("Pick an index from the SHAP sample", 0, len(X_sample) - 1, 0)
        shap_vals_row = shap_vals[idx : idx + 1]

        # Force plot (matplotlib mode)
        fig2, ax2 = plt.subplots(figsize=(8, 2.5))
        shap.force_plot(
            expected_value,
            shap_vals_row,
            matplotlib=True,
            show=False,
        )
        st.pyplot(fig2, use_container_width=True)

        st.caption(
            "SHAP provides both global feature importance (summary plot) "
            "and local explanations for individual decisions (force plot)."
        )

    # ---------- 4) What-if Analysis ----------
    with tab_whatif:
        st.markdown("<div class='section-title'>What-if Analysis – Individual-level Governance</div>", unsafe_allow_html=True)

        df = clean_data()
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
                value=float(round(base[col], 2)),
            )

        st.write("### Categorical Features")
        for col in CATEGORICAL_COLS:
            options = df[col].value_counts().index.tolist()
            user_input[col] = st.selectbox(col, options, index=0)

        input_df = pd.DataFrame([user_input])

        prob = clf.predict_proba(input_df)[0, 1]
        pred = int(prob >= 0.5)

        st.write("### Model Output")
        st.markdown(
            f"**Approval probability:** `{prob:.3f}`  &nbsp;&nbsp; "
            f"**Decision:** {'✅ Approved (1)' if pred == 1 else '❌ Rejected (0)'}"
        )

        st.info(
            "Use this panel to explore counterfactuals – how changes in features affect approval."
        )


if __name__ == "__main__":
    main()
