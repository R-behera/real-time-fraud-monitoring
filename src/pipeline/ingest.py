from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class IngestConfig:
    raw_path: str = "data/raw"
    file_name: str = "input.csv"


def ingest_csv(config: Optional[IngestConfig] = None) -> pd.DataFrame:
    cfg = config or IngestConfig()
    path = Path(cfg.raw_path) / cfg.file_name
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_csv(path)


def ingest_records(records):
    """Accept stream-like records and return DataFrame."""
    return pd.DataFrame.from_records(records)
