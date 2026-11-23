.PHONY: install install-ml dev dev-stable test lint typecheck format check up down logs clean corpus train eval generate

install:
	pip install -e ".[dev]"

install-ml:
	pip install -e ".[dev,ml]"
	python -c "import numpy, torch; print('numpy', numpy.__version__, 'torch', torch.__version__)"

# Reload only app/package source — never watch .venv (avoids restart storms on Windows)
dev:
	uvicorn apps.api.main:app --reload --reload-dir apps --reload-dir packages --host 0.0.0.0 --port 8000

# No reload — stable for manual API testing while installing packages
dev-stable:
	uvicorn apps.api.main:app --host 0.0.0.0 --port 8000

test:
	python -m pytest tests/unit -v

test-all:
	python -m pytest -v

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
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage

# --- Model workflow ---

corpus:
	python scripts/build_seed_corpus.py

train:
	python -m training.scripts.train_lm --config tiny --steps 500 --batch-size 8 --seq-len 64 --eval-every 50

eval:
	python -m training.scripts.evaluate_lm --checkpoint training/checkpoints/lm_tiny

generate:
	python -m training.scripts.generate --prompt "def " --max-new-tokens 40
