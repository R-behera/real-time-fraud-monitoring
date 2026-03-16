from __future__ import annotations


def latency_alert_ms(ms: float, alert_threshold_ms: float = 120.0):
    return {"alert": bool(ms > alert_threshold_ms), "ms": float(ms), "threshold": float(alert_threshold_ms)}
