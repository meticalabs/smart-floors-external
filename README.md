# bid-optim-etl-py

<p>
<a href="https://www.python.org/downloads/release/python-3100/"><img alt="python" src="https://img.shields.io/badge/python-3.10+-blue.svg"></a>
<a href="https://github.com/meticalabs/bid-optim-etl-py/actions/workflows/wf-on-push-and-pr.yml"><img alt="Build" src="https://github.com/meticalabs/bid-optim-etl-py/actions/workflows/wf-on-push-and-pr.yml/badge.svg?branch=main"></a>
<a href="https://python-poetry.org/"><img alt="Poetry" src="https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://github.com/astral-sh/ruff"><img alt="ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json"></a>
</p>

## Description
Repository to hold the implementation of the bid floor optimisation ml training and inference logic

## Build
```shell
# Pre-requisite: Setup venv or conda environment before running below commands

# Install dependencies
make install

# Run all tests
make test

# Tests with coverage
make test-coverage

# Build
make build

# Format 
make format

# Lint (Ruff)
make ruff

# All (Install + Test + Build)
make all
```
