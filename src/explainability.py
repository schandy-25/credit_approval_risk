# src/explainability.py

import shap

def compute_shap_values(pipeline, X_sample):
    """
    pipeline: sklearn Pipeline(preprocess, model)
    X_sample: small subset of original (untransformed) feature DataFrame
    """
    preprocess = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]

    # Features as seen by the model
    X_trans = preprocess.transform(X_sample)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_trans)

    # For RandomForestClassifier (binary), shap_values is a list [class0, class1]
    shap_vals_class1 = shap_values[1]
    expected_value = explainer.expected_value[1]

    return shap_vals_class1, expected_value, X_trans
