.PHONY: help install dev lint format typecheck test test-all stubs clean

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package (production deps only)
	pip install -e .

dev:  ## Install with dev dependencies and pre-commit hooks
	pip install -e ".[dev]"
	pre-commit install

lint:  ## Run ruff linter
	ruff check src/ tests/

format:  ## Auto-format code with ruff
	ruff format src/ tests/
	ruff check --fix src/ tests/

typecheck:  ## Run mypy type checking
	mypy src/

test:  ## Run tests (excluding GPU tests)
	pytest -m "not gpu" --tb=short

test-all:  ## Run all tests including GPU
	pytest --tb=short

stubs:  ## Regenerate Pylance stubs for untyped packages (run after pip upgrade)
	stubgen -p glfw -p moderngl -o typestubs/
	cp typestubs/glfw/__init__.pyi ~/Developer/typestubs/glfw/__init__.pyi
	cp typestubs/moderngl/__init__.pyi ~/Developer/typestubs/moderngl/__init__.pyi

clean:  ## Remove build artifacts
	rm -rf dist/ build/ *.egg-info .pytest_cache .mypy_cache .coverage htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
