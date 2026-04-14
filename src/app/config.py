
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def load_project_config() -> dict:
    root = Path(__file__).resolve().parents[2]
    config_path = root / "configs" / "project.json"
    return json.loads(config_path.read_text(encoding="utf-8"))
