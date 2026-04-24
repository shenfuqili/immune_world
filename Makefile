.PHONY: help install dev lint type test smoke docker clean

PY ?= python

help:
	@echo "make install  -- pip install -e ."
	@echo "make dev      -- pip install -e .[dev]"
	@echo "make lint     -- ruff + black --check + isort --check-only"
	@echo "make type     -- mypy --strict on src/immune_world"
	@echo "make test     -- pytest -q (CPU; skips GPU-marked tests)"
	@echo "make smoke    -- run 2-step training on _smoke.yaml"
	@echo "make docker   -- build immune_world:dev image"
	@echo "make clean    -- remove caches"

install:
	$(PY) -m pip install -e .

dev:
	$(PY) -m pip install -e ".[dev]"

lint:
	$(PY) -m ruff check .
	$(PY) -m black --check src tests
	$(PY) -m isort --check-only src tests

type:
	$(PY) -m mypy --strict src/immune_world

test:
	$(PY) -m pytest -q -m "not gpu"

smoke:
	$(PY) -m immune_world.cli.train experiment=_smoke

docker:
	docker build -t immune_world:dev .

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info .coverage htmlcov
	find . -type d -name '__pycache__' -exec rm -rf {} +
