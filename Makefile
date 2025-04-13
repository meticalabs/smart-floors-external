##@ Utility
.PHONY: help
help:  ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make <target>\033[36m\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)


.PHONY: uv
uv:  ## Install uv if it's not present.
	@command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh

.PHONY: dev
dev: uv ## Install dev dependencies
	uv sync --dev

.PHONY: lock
lock: uv ## lock dependencies
	uv lock

.PHONY: install
install: uv ## Install dependencies
	uv sync --frozen

.PHONY: test
test:  ## Run tests
	uv run pytest

.PHONY: ruff
ruff:  ## Run linters
	uv run ruff check ./tests

.PHONY: cov
cov: ## Run tests with coverage
	uv run pytest --cov=bid_optim_etl_py --cov-report=xml --cov-report=html --cov-report=term-missing

.PHONY: doc
doc:  ## Build documentation
	cd docs && uv run make html

.PHONY: build
build:  ## Build package
	uv build

.PHONY: clean
clean:
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache
	rm -f uv.lock

.PHONY: format
format:
	uv run black .

.PHONY: all
all:  clean lock install ruff cov build
