
from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.app.config import load_project_config
from src.app.engine import analyze_payload, query_payload, score_payload


app = FastAPI()
CONFIG = load_project_config()


class ScoreRequest(BaseModel):
    features: dict[str, Any] = Field(default_factory=dict)


class AnalyzeRequest(BaseModel):
    metrics: dict[str, float] = Field(default_factory=dict)
    baseline: dict[str, float] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    query: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "project": CONFIG["slug"]}


@app.get("/project")
def project() -> dict[str, Any]:
    return {
        "slug": CONFIG["slug"],
        "title": CONFIG["title"],
        "project_type": CONFIG["project_type"],
        "domain": CONFIG["domain"],
        "tags": CONFIG["tags"],
    }


@app.post("/score")
def score(request: ScoreRequest) -> dict[str, Any]:
    return score_payload(CONFIG, request.features)


@app.post("/analyze")
def analyze(request: AnalyzeRequest) -> dict[str, Any]:
    return analyze_payload(CONFIG, request.metrics, request.baseline)


@app.post("/query")
def query(request: QueryRequest) -> dict[str, Any]:
    return query_payload(CONFIG, request.query)
