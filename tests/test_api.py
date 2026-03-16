from __future__ import annotations

from src.serving import api


def test_health_ok():
    status = api.health()
    assert status["status"] == "ok"
