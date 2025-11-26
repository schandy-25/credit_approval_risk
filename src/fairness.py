# src/fairness.py
import numpy as np
import pandas as pd

def _group_mask(sensitive_series, group_value):
    return (sensitive_series == group_value)

def demographic_parity(y_pred, sensitive):
    """
    Returns approval rates per group + Demographic Parity difference.
    """
    sensitive = pd.Series(sensitive)
    y_pred = np.array(y_pred)

    groups = sensitive.unique()
    rates = {}

    for g in groups:
        mask = _group_mask(sensitive, g)
        if mask.sum() == 0:
            rates[g] = np.nan
        else:
            rates[g] = y_pred[mask].mean()

    vals = [v for v in rates.values() if not np.isnan(v)]
    dp_diff = max(vals) - min(vals) if len(vals) > 1 else 0.0

    return rates, dp_diff

def equalized_odds(y_true, y_pred, sensitive):
    """
    Computes TPR/FPR per group and EO gaps.
    """
    sensitive = pd.Series(sensitive)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    groups = sensitive.unique()
    tpr, fpr = {}, {}

    for g in groups:
        mask = _group_mask(sensitive, g)
        if mask.sum() == 0:
            tpr[g] = fpr[g] = np.nan
            continue

        yt = y_true[mask]
        yp = y_pred[mask]

        pos = (yt == 1)
        tpr[g] = yp[pos].mean() if pos.sum() > 0 else np.nan

        neg = (yt == 0)
        fpr[g] = yp[neg].mean() if neg.sum() > 0 else np.nan

    def _gap(d):
        vals = [v for v in d.values() if not np.isnan(v)]
        return max(vals) - min(vals) if len(vals) > 1 else 0.0

    tpr_gap = _gap(tpr)
    fpr_gap = _gap(fpr)

    return tpr, fpr, tpr_gap, fpr_gap
