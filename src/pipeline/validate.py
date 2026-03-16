from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass
class ValidationResult:
    valid: bool
    issues: list[str]


def validate_columns(df: pd.DataFrame, required_columns: list[str]) -> ValidationResult:
    missing = [c for c in required_columns if c not in df.columns]
    valid = len(missing) == 0
    issues = [f"missing_columns={missing}"] if not valid else []
    return ValidationResult(valid=valid, issues=issues)


def validate_non_empty(df: pd.DataFrame, min_rows: int = 100) -> ValidationResult:
    if df is None or len(df) < min_rows:
        return ValidationResult(False, [f"rows_below_threshold:{0 if df is None else len(df)}"])
    return ValidationResult(True, [])
