from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix
import joblib


def evaluate(feature_path="data/processed/test.csv", threshold=0.5):
    model = joblib.load("models/model.joblib")
    df = pd.read_csv(feature_path)
    if "target" not in df.columns:
        raise ValueError("Expected target column missing")

    X = df.drop(columns=["target"])
    y = df["target"]
    probs = model.predict_proba(X)[:, 1]
    y_pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

    summary = {
        "threshold": threshold,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    Path("models").mkdir(exist_ok=True, parents=True)
    Path("models/eval_results.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    print(evaluate())
