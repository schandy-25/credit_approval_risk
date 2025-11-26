# src/modeling.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import shap

from .data_utils import clean_data
from .preprocessing import build_preprocessor
from .config import TARGET_COL, SENSITIVE_COL, NUMERIC_COLS, CATEGORICAL_COLS

def train_model(test_size=0.2, random_state=42):
    """
    Train a RandomForest classifier with preprocessing pipeline.
    Returns the fitted pipeline and evaluation data.
    """
    df = clean_data()

    feature_cols = NUMERIC_COLS + CATEGORICAL_COLS
    X = df[feature_cols]
    y = df[TARGET_COL].astype(int)
    sensitive = df[SENSITIVE_COL]

    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, sensitive,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    preprocessor = build_preprocessor()

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1
    )

    clf = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ])

    clf.fit(X_train, y_train)

    # Predictions on test set
    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }

    eval_data = (X_test, y_test, s_test, y_pred, y_proba, metrics)
    return clf, eval_data


# ===== Additional functions for fairness and SHAP =====
def compute_shap_values(clf, X, nsamples=100):
    """
    Compute SHAP values for the given model and dataset.
    """
    preprocessor = clf.named_steps["preprocess"]
    model = clf.named_steps["model"]

    X_trans = preprocessor.transform(X)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_trans)[1]  # class 1

    return shap_values, explainer.expected_value[1]


def predict_with_threshold(clf, X, threshold=0.5):
    """
    Predict class labels based on a custom probability threshold.
    """
    proba = clf.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return pred, proba


def evaluate_model(clf, X_test, y_test, s_test=None):
    """
    Convenience function to compute metrics for any dataset.
    """
    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }

    return y_pred, y_proba, metrics
