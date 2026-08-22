"""Smoke test against an already-running docker-compose stack.

Unlike tests/ (pure unit tests -- no network calls, no containers), this
hits the actual running services over HTTP. It exists to catch the class of
bug that unit tests structurally can't see: wrong Dockerfile COPY paths,
docker-compose volume/network misconfiguration, or the trainer and API
disagreeing on the model registry name -- exactly the kind of thing that
broke the original project (see registry_name in config/train.yaml) and
that no amount of mocking would have caught.

Not run in CI: CI has no docker-compose stack to test against. Run this by
hand (`make smoke-test`) after `make up`, or as a manual gate before
shipping a change that touches Dockerfiles, docker-compose.yml, or the
trainer/API wiring.
"""

from __future__ import annotations

import sys

import requests

MLFLOW_URL = "http://localhost:5000"
API_URL = "http://localhost:8000"
WEBAPP_URL = "http://localhost:5001"

SAMPLE_PAYLOAD = {
    "MedInc": 8.3,
    "HouseAge": 41,
    "AveRooms": 6.9,
    "AveBedrms": 1.0,
    "Population": 322,
    "AveOccup": 2.5,
    "Latitude": 37.9,
    "Longitude": -122.2,
}


def check_mlflow_reachable() -> None:
    response = requests.get(MLFLOW_URL, timeout=5)
    response.raise_for_status()


def check_api_health() -> None:
    response = requests.get(f"{API_URL}/health", timeout=5)
    response.raise_for_status()
    assert "model_uri" in response.json()


def check_api_predicts() -> None:
    response = requests.post(f"{API_URL}/predict", json=SAMPLE_PAYLOAD, timeout=15)
    response.raise_for_status()
    prediction = response.json().get("prediction")
    assert isinstance(prediction, (int, float)), f"unexpected response: {response.json()}"


def check_webapp_health() -> None:
    response = requests.get(f"{WEBAPP_URL}/health", timeout=5)
    response.raise_for_status()


def check_webapp_form_round_trips_a_prediction() -> None:
    response = requests.post(WEBAPP_URL + "/", data=SAMPLE_PAYLOAD, timeout=15)
    response.raise_for_status()
    assert "Predicted median house value" in response.text


CHECKS = [
    ("mlflow reachable", check_mlflow_reachable),
    ("api /health", check_api_health),
    ("api /predict returns a number", check_api_predicts),
    ("webapp /health", check_webapp_health),
    ("webapp form round-trips a prediction", check_webapp_form_round_trips_a_prediction),
]


def main() -> None:
    failures = []
    for name, check in CHECKS:
        try:
            check()
        except Exception as e:
            print(f"FAIL  {name}: {e}")
            failures.append(name)
        else:
            print(f"OK    {name}")

    if failures:
        print(f"\n{len(failures)}/{len(CHECKS)} checks failed: {', '.join(failures)}")
        print("Is the stack up? Run `make up` first.")
        sys.exit(1)

    print(f"\nAll {len(CHECKS)} smoke checks passed.")


if __name__ == "__main__":
    main()
