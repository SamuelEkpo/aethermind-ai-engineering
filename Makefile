.PHONY: install install-ml dev test lint typecheck format check up down logs clean

install:
	pip install -e ".[dev]"

install-ml:
	pip install -e ".[dev,ml]"

dev:
	uvicorn apps.api.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest tests/unit

test-all:
	pytest

lint:
	ruff check apps packages tests

format:
	ruff format apps packages tests
	ruff check --fix apps packages tests

typecheck:
	mypy apps packages

check: lint typecheck test

up:
	docker compose up -d postgres redis mlflow

down:
	docker compose down

logs:
	docker compose logs -f

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
