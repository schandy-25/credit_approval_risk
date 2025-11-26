# src/config.py

DATA_PATH = "data/uci_credit_approval.csv"  # your raw file

# Give proper names to the 16 columns (15 attrs + class)
COLUMN_NAMES = [
    "A1", "A2", "A3", "A4", "A5",
    "A6", "A7", "A8", "A9", "A10",
    "A11", "A12", "A13", "A14", "A15",
    "Approved"
]

TARGET_COL = "Approved"   # last column, '+' or '-'

# Let's treat A1 as the sensitive attribute (it's often used as a gender-like attr)
SENSITIVE_COL = "A1"

# Numeric attributes in original UCI documentation
NUMERIC_COLS = [
    "A2",   # e.g., Age
    "A3",   # Debt
    "A8",   # YearsEmployed
    "A11",  # CreditScore-ish
    "A14",  # Zip/income-ish
    "A15"   # Income
]

# Remaining as categorical
CATEGORICAL_COLS = [
    "A1", "A4", "A5", "A6", "A7",
    "A9", "A10", "A12", "A13"
]
