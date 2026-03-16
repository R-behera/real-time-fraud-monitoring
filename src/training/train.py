from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support


def main():
    data_path = Path("data/processed/train.csv")
    if not data_path.exists():
        raise FileNotFoundError("data/processed/train.csv not found; run pipeline first.")

    df = pd.read_csv(data_path)
    if "target" not in df.columns:
        raise ValueError("Expected target column missing.")

    X = df.drop(columns=["target"])
    y = df["target"]

    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    model = Pipeline(
        steps=[
            ("pre", preprocessor),
            ("clf", RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=-1)),
        ]
    )

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_val)[:, 1]
    y_pred = (probs >= 0.5).astype(int)
    precision, recall, _, _ = precision_recall_fscore_support(y_val, y_pred, average="binary", zero_division=0)

    metrics = {
        "auc": float(roc_auc_score(y_val, probs)),
        "precision": float(precision),
        "recall": float(recall),
    }

    print("validation_metrics", metrics)
    Path("models").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, Path("models/model.joblib"))
    pd.Series(metrics).to_json(Path("models/metrics.json"))


if __name__ == "__main__":
    main()
