# Architecture Overview

This template follows a production-friendly flow:

1. Ingest raw data into `data/raw`.
2. Clean and validate in `src/pipeline`.
3. Train and evaluate in `src/training`.
4. Save artifacts in `models/`.
5. Serve with FastAPI in `src/serving`.
6. Monitor drift/latency in `src/monitoring`.
