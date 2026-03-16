# Runbook

## Deployment

1. Install dependencies from `requirements.txt`.
2. Run training with `python -m src.training.train`.
3. Start API with uvicorn.

## Rollback

- Stop API.
- Restore previous artifact from model registry.
- Restart and verify /health.

## Monitoring

- Monitor `/metrics` endpoint.
- Run drift checks on daily windows.
