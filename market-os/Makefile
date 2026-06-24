.PHONY: setup infra-up infra-down demo dashboard test lint clean

VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

setup:
	python3 -m venv $(VENV)
	$(PIP) install -U pip
	$(PIP) install -e ".[data,ts,dev]"
	@echo "Base install done. For full stack: $(PIP) install -e '.[all]'"

infra-up:
	docker compose up -d

infra-down:
	docker compose down

demo:
	$(PY) scripts/demo.py

dashboard:
	$(VENV)/bin/streamlit run src/marketos/dashboard/app.py

test:
	$(PY) -m pytest -q

lint:
	$(VENV)/bin/ruff check src tests scripts

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
