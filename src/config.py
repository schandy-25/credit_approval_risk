# src/config.py

# Column names for the UCI Credit Approval (crx) dataset
# https://archive.ics.uci.edu/ml/datasets/credit+approval
COL_NAMES = [
    "A1",  # categorical
    "A2",  # numeric
    "A3",  # numeric
    "A4",  # categorical
    "A5",  # categorical
    "A6",  # categorical
    "A7",  # categorical
    "A8",  # numeric
    "A9",  # categorical
    "A10", # categorical
    "A11", # numeric
    "A12", # categorical
    "A13", # categorical
    "A14", # numeric
    "A15", # numeric
    "A16", # target: '+' or '-'
]

TARGET_COL = "A16"
SENSITIVE_COL = "A1"  # we'll treat A1 as sensitive attribute

# Numeric columns based on dataset description
NUMERIC_COLS = ["A2", "A3", "A8", "A11", "A14", "A15"]

# All other feature columns (except target) are categorical
CATEGORICAL_COLS = [
    "A1", "A4", "A5", "A6", "A7", "A9", "A10", "A12", "A13"
]
