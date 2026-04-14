
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.app.config import load_project_config
from src.app.engine import analyze_payload, query_payload, score_payload


app = FastAPI()
CONFIG = load_project_config()
WEB_DIR = Path(__file__).resolve().parent / "web"


class ScoreRequest(BaseModel):
    features: dict[str, Any] = Field(default_factory=dict)


class AnalyzeRequest(BaseModel):
    metrics: dict[str, float] = Field(default_factory=dict)
    baseline: dict[str, float] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    query: str


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/styles.css", include_in_schema=False)
def styles() -> FileResponse:
    return FileResponse(WEB_DIR / "styles.css", media_type="text/css")


@app.get("/app.js", include_in_schema=False)
def app_js() -> FileResponse:
    return FileResponse(WEB_DIR / "app.js", media_type="application/javascript")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "project": CONFIG["slug"]}


@app.get("/bootstrap")
def bootstrap() -> dict[str, Any]:
    return {
        "project": {
            "slug": CONFIG["slug"],
            "title": CONFIG["title"],
            "project_type": CONFIG["project_type"],
            "domain": CONFIG["domain"],
            "summary": CONFIG["summary"],
            "tags": CONFIG["tags"],
        },
        "sample_features": CONFIG["sample_features"],
        "sample_metrics": CONFIG["sample_metrics"],
        "sample_query": "What is the top priority recommendation?",
        "knowledge_base": CONFIG["knowledge_base"],
        "dashboard_metrics": CONFIG["dashboard_metrics"],
    }


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
