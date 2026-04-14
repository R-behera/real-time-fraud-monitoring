
# Runbook

## Local Run
- Install dependencies with `pip install -r requirements.txt`
- Seed mock outputs with `python scripts/bootstrap_data.py`
- Start the API with `uvicorn src.app.main:app --reload`

## Deployment
- Build the image with `docker build -t real-time-fraud-monitoring:latest .`
- Run via `docker compose up --build`

## Monitoring
- Open `/` for the interactive application
- Use `/health` for availability
- Use `/bootstrap` for client-side initialization data
- Use `/project` for version and metadata checks
- Extend API logging and request tracing for production usage
