
# Architecture

Real-Time Fraud Monitoring is structured as a compact production-style starter:

1. `configs/project.json` stores project metadata, sample features, and a small local knowledge base.
2. `src/app/engine.py` holds the scoring, analysis, and retrieval logic.
3. `src/app/main.py` exposes FastAPI endpoints plus the application shell at `/`.
4. `src/app/web/` contains the interactive browser app and `demo/` mirrors those assets for screenshots and repo previews.
5. `.github/workflows/ci.yml` runs a lightweight validation pass in CI.
