
from __future__ import annotations

import json
from pathlib import Path

from src.app.config import load_project_config
from src.app.engine import analyze_payload, query_payload, recommend_payload, score_payload


def main() -> None:
    config = load_project_config()
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "data" / "mock"
    output_dir.mkdir(parents=True, exist_ok=True)

    score_example = score_payload(config, config["sample_features"])
    analysis_example = analyze_payload(config, config["sample_metrics"], {})
    query_example = query_payload(config, "What is the top priority recommendation?")
    recommend_example = recommend_payload(
        config,
        "Recommend the next best action for this project.",
        config["sample_features"],
        config["sample_metrics"],
    )

    (output_dir / "score_example.json").write_text(json.dumps(score_example, indent=2), encoding="utf-8")
    (output_dir / "analysis_example.json").write_text(json.dumps(analysis_example, indent=2), encoding="utf-8")
    (output_dir / "query_example.json").write_text(json.dumps(query_example, indent=2), encoding="utf-8")
    (output_dir / "recommend_example.json").write_text(json.dumps(recommend_example, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
