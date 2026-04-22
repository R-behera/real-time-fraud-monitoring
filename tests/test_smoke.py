from src.app.main import app


def test_title_present():
    assert app.title == 'Real-Time Fraud Monitoring'
