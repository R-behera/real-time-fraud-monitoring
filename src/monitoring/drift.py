from __future__ import annotations

import pandas as pd


def ks_like_drift(reference: pd.Series, current: pd.Series, zscore: float = 3.0):
    r_mean, r_std = reference.mean(), reference.std() or 1.0
    c_mean = current.mean()
    score = abs(c_mean - r_mean) / r_std
    return {"drift": score >= zscore, "score": float(score)}
