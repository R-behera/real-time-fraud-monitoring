
install:
	pip install -r requirements.txt

run:
	uvicorn src.app.main:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest -q

seed:
	python scripts/bootstrap_data.py
