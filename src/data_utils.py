# src/data_utils.py
import pandas as pd
import numpy as np
from .config import DATA_PATH, TARGET_COL, COLUMN_NAMES

def load_raw_data():
    # No header in file → header=None, names=...
    df = pd.read_csv(
        DATA_PATH,
        header=None,
        names=COLUMN_NAMES
    )

    # In original UCI, missing values are '?'
    df = df.replace("?", np.nan)

    return df

def clean_data():
    df = load_raw_data()

    # Map '+' / '-' to 1 / 0
    if df[TARGET_COL].dtype == "object":
        df[TARGET_COL] = df[TARGET_COL].map({"+": 1, "-": 0})

    # Drop rows with missing target
    df = df.dropna(subset=[TARGET_COL])

    # Simple cleaning: drop rows with any NaNs (you can switch to imputation later)
    df = df.dropna().reset_index(drop=True)

    # Ensure int labels
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    return df

