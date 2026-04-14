
from __future__ import annotations

from hashlib import md5
from statistics import mean


def _stable_number(value: object) -> float:
    digest = md5(str(value).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def score_payload(config: dict, features: dict) -> dict:
    numeric_values = []
    explanation = []
    for key, value in features.items():
        if isinstance(value, bool):
            numeric_values.append(1.0 if value else 0.0)
        elif isinstance(value, (int, float)):
            numeric_values.append(float(value))
        else:
            numeric_values.append(_stable_number(value) * 100)
        explanation.append({"feature": key, "weight_hint": round(_stable_number(key) * 2, 3)})
    base_score = mean(numeric_values) if numeric_values else 0.0
    normalized = max(0.0, min(0.99, base_score / 100))
    return {
        "project_type": config["project_type"],
        "score": round(normalized, 4),
        "band": "high" if normalized >= 0.7 else "medium" if normalized >= 0.4 else "low",
        "top_signals": explanation[:3],
    }


def analyze_payload(config: dict, metrics: dict, baseline: dict | None = None) -> dict:
    baseline = baseline or {}
    deltas = {}
    for key, value in metrics.items():
        prior = float(baseline.get(key, value))
        deltas[key] = round(float(value) - prior, 4)
    top_metric = max(deltas, key=lambda item: abs(deltas[item])) if deltas else None
    summary = "No metrics supplied." if top_metric is None else f"Largest movement is in {top_metric} with a delta of {deltas[top_metric]}."
    return {"project_type": config["project_type"], "summary": summary, "deltas": deltas}


def query_payload(config: dict, query: str) -> dict:
    query_terms = {part.lower() for part in query.split() if part.strip()}
    scored = []
    for passage in config.get("knowledge_base", []):
        passage_terms = {part.lower().strip(".,:") for part in passage.split()}
        overlap = len(query_terms & passage_terms)
        scored.append((overlap, passage))
    scored.sort(key=lambda item: item[0], reverse=True)
    matches = [passage for _, passage in scored[:3]]
    answer = matches[0] if matches else "No grounded context found."
    return {"project_type": config["project_type"], "answer": answer, "contexts": matches}


def recommend_payload(config: dict, prompt: str, features: dict | None = None, metrics: dict | None = None) -> dict:
    features = features or {}
    metrics = metrics or {}
    score = score_payload(config, features) if features else {"score": 0.0, "band": "low", "top_signals": []}
    analysis = analyze_payload(config, metrics, {}) if metrics else {"summary": "No metrics supplied.", "deltas": {}}
    ordered_metrics = sorted(metrics.items(), key=lambda item: abs(float(item[1])), reverse=True)
    top_metric = ordered_metrics[0][0] if ordered_metrics else config["dashboard_metrics"][0]["label"]
    actions = [
        {
            "action": f"Review the workflow around {top_metric}.",
            "owner": "analytics",
            "reason": analysis["summary"],
        },
        {
            "action": f"Inspect the leading signal from the {config['project_type']} scoring pass.",
            "owner": "operations",
            "reason": f"Current band is {score['band']} with score {score['score']}.",
        },
        {
            "action": f"Share an execution update with {config['domain']} stakeholders.",
            "owner": "product",
            "reason": prompt or "Prompt not supplied.",
        },
    ]
    return {
        "project_type": config["project_type"],
        "prompt": prompt,
        "actions": actions,
        "score_snapshot": score,
        "analysis_snapshot": analysis,
    }
