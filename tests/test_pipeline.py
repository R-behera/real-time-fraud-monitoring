from __future__ import annotations

import pandas as pd
from src.pipeline.validate import validate_non_empty, validate_columns


def test_validate_empty_dataframe():
    assert not validate_non_empty(pd.DataFrame(), min_rows=10).valid


def test_validate_columns_missing():
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    result = validate_columns(df, ['a', 'c'])
    assert not result.valid
    assert result.issues
