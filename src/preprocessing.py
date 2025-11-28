# src/preprocessing.py

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from .config import NUMERIC_COLS, CATEGORICAL_COLS

def build_preprocessor():
    numeric_tf = Pipeline(
        steps=[("scaler", StandardScaler())]
    )

    categorical_tf = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_tf, NUMERIC_COLS),
            ("cat", categorical_tf, CATEGORICAL_COLS),
        ]
    )

    return preprocessor
