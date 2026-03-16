# Evaluation Plan

- Hold out 20% of data for validation.
- Track AUC, precision, and recall.
- Run `src/training/evaluate.py` after each retrain.
- Deploy only when metrics pass quality gates and are reviewed.
