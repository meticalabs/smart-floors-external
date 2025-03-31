install:
	poetry lock --no-update
	poetry install --no-root --no-interaction --with=dev,test

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
	rm -f poetry.lock

format:
	poetry run black .

ruff:
	poetry run ruff clean && poetry run ruff check .

test: clean
	poetry run pytest -s .

test-coverage: clean
	poetry run coverage run -m pytest && poetry run coverage report -m && poetry run coverage html && poetry run coverage xml

build: clean install
	poetry build

all: clean install ruff test-coverage build