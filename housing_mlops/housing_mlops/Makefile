# Operator entrypoints for this project. Prefer these over hand-typed
# docker/uv/pytest invocations so the "right" way to run something is
# documented in one place instead of living in someone's shell history.
#
# Local dev tooling (uv, ruff, pytest) is expected to be installed on the
# host -- see README for setup. Docker images themselves never invoke uv
# (see the Dockerfiles); uv is only used here, locally, to manage the
# lockfile and run tests/lint outside a container.

.PHONY: help lock sync lint format test \
        up down train api webapp logs select-features smoke-test clean

help:
	@echo "make lock            - resolve dependencies into uv.lock, export per-service requirements.txt"
	@echo "make sync            - install all dependency groups into the local .venv"
	@echo "make lint            - run ruff over the codebase"
	@echo "make format          - auto-fix lint issues where possible"
	@echo "make test            - run the pytest suite (unit tests, no containers needed)"
	@echo "make up              - start mlflow, run the trainer, then start api + webapp (docker compose)"
	@echo "make train           - (re)build and run just the trainer container"
	@echo "make api             - (re)build and run just the api container"
	@echo "make webapp          - (re)build and run just the webapp container"
	@echo "make select-features - run the feature-selection analysis inside the trainer image"
	@echo "make smoke-test      - hit the running mlflow/api/webapp containers over HTTP (run 'make up' first)"
	@echo "make logs            - tail logs from all running containers"
	@echo "make down            - stop and remove all containers"
	@echo "make clean           - down, plus remove local mlflow data and reports"

# --- Local dependency management (uv; never runs inside Docker) -------------
#
# Note: webapp/requirements.txt is NOT regenerated here. `uv export --extra
# webapp` would also pull in housing-ml's base dependencies (pandas,
# scikit-learn, ...) since webapp is an extra of the same pyproject.toml --
# but the webapp is a standalone HTTP client that needs none of that, so its
# requirements.txt is hand-maintained instead (see its own header comment).

lock:
	uv lock
	uv export --format requirements.txt --extra trainer --no-hashes --no-header -o trainer/requirements.txt
	uv export --format requirements.txt --extra api --no-hashes --no-header -o api/requirements.txt
	@# uv export always includes "-e ." (the local package) as its first line;
	@# strip it -- both Dockerfiles install the local package separately, in
	@# their own `pip install --no-deps -e .` step, after these pinned deps.
	sed -i '/^-e \.$$/d' trainer/requirements.txt api/requirements.txt

sync:
	uv sync --extra trainer --extra api --extra webapp --extra dev

lint:
	uv run ruff check .

format:
	uv run ruff check . --fix

test:
	uv run pytest

# --- Docker compose orchestration -------------------------------------------

up:
	docker compose up -d --build mlflow trainer
	docker compose up -d --build api webapp

train:
	docker compose up -d --build trainer

api:
	docker compose up -d --build api

webapp:
	docker compose up -d --build webapp

# Runs the feature-selection script inside a one-off trainer container. The
# container mounts ./config and ./reports read-write (see docker-compose.yml),
# so the resulting features.yaml and plots land back on the host.
select-features:
	docker compose run --rm trainer python scripts/select_features.py

# Assumes `make up` (or equivalent) already has mlflow/api/webapp running --
# this does not start anything itself, it only checks what's there.
smoke-test:
	uv run python scripts/smoke_test.py

logs:
	docker compose logs -f

down:
	docker compose down

clean: down
	rm -rf mlflow-data reports/figures/*.png
