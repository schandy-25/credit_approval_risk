# src/data_utils.py

import pandas as pd
from .config import COL_NAMES, TARGET_COL, NUMERIC_COLS

def clean_data():
   
    df = pd.read_csv(
        "data/uci_credit_approval.csv",
        header=None,
        names=COL_NAMES,
        na_values="?",
    )

    # Drop rows with missing values for simplicity
    df = df.dropna().reset_index(drop=True)

    # Convert numeric columns to float
    for col in NUMERIC_COLS:
        df[col] = df[col].astype(float)

    # Convert target '+'/'-' to 1/0
    df[TARGET_COL] = df[TARGET_COL].map({"+": 1, "-": 0}).astype(int)

    return df
