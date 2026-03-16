# Real-Time Fraud Monitoring

## Overview
Streaming transaction fraud detection with end-to-end ingest, validation, training, scoring API, monitoring, and incident runbooks.

## What this project demonstrates
- ML model training, validation, and evaluation.
- Data pipeline for ingest, cleaning, validation, and feature preparation.
- Production API with health and metrics endpoints.
- Monitoring scripts for drift and latency.
- Deployment artifacts with Docker and compose configuration.

## Quick Start

1. Create virtual env and install dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run training
```bash
python -m src.training.train
```

3. Start API
```bash
python -m uvicorn src.serving.api:app --reload --host 0.0.0.0 --port 8000
```

## Endpoints
- GET /health
- POST /predict
- GET /metrics

## Structure
```
.
├── src/
│   ├── pipeline/
│   ├── training/
│   ├── serving/
│   └── monitoring/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── docs/
├── tests/
├── infra/
└── requirements.txt
```
