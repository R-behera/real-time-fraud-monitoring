from __future__ import annotations

from typing import Any, Dict, List
import os
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response
import joblib

app = FastAPI(title="ML Inference API")

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
MODEL = joblib.load(MODEL_PATH) if Path(MODEL_PATH).exists() else None
REQ_COUNT = Counter("predict_requests_total", "Total prediction requests")
PRED_LATENCY = Histogram("predict_latency_seconds", "Prediction latency")


class Record(BaseModel):
    features: Dict[str, Any] = Field(..., description="Flat feature map")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "model_loaded": str(MODEL is not None)}


@app.post("/predict")
def predict(payload: List[Record]):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    import time

    req = pd.DataFrame([item.features for item in payload])
    REQ_COUNT.inc()
    start = time.perf_counter()
    probs = MODEL.predict_proba(req)[:, 1]
    latency = time.perf_counter() - start
    PRED_LATENCY.observe(latency)

    threshold = float(os.getenv("THRESHOLD", "0.5"))
    out = [{"score": float(s), "risk": "high" if s >= threshold else "low"} for s in probs]
    return {"predictions": out}


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type="text/plain")
