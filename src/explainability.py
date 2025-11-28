# src/explainability.py
import shap

def compute_shap_values(pipeline, X_sample):
    """
    pipeline: sklearn Pipeline(preprocess, model)
    X_sample: small subset of original (untransformed) feature DataFrame
    """
    preprocess = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]

    # Transform features (this is what the model actually sees)
    X_trans = preprocess.transform(X_sample)

    # TreeExplainer for RandomForest
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_trans)

    # Focus on positive class (approval = 1)
    shap_vals_class1 = shap_values[1]
    expected_value = explainer.expected_value[1]

    
    return shap_vals_class1, expected_value, X_trans
