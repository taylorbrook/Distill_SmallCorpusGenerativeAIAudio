# Source: Community best practice for Python ML projects
.PHONY: help setup run test lint format clean benchmark

PYTHON := uv run python
MODULE := distill

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## Install all dependencies and set up environment
	uv sync
	@echo "Setup complete. Run 'make run' to start."

run: ## Run the application
	$(PYTHON) -m $(MODULE)

run-verbose: ## Run with verbose output
	$(PYTHON) -m $(MODULE) --verbose

test: ## Run test suite
	$(PYTHON) -m pytest tests/ -v

lint: ## Run linter
	uv run ruff check src/ tests/

format: ## Auto-format code
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

clean: ## Remove caches and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ *.egg-info

benchmark: ## Run hardware benchmark
	$(PYTHON) -m $(MODULE) --benchmark
