"""API tests.

These never talk to a real MLflow server -- `get_model` is monkeypatched to
return a stub with a fixed `.predict()`, so the test suite can verify the
API's request/response contract and error handling without needing the
`mlflow` + `api` services running.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))

import main as api_main  # noqa: E402

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


class _StubModel:
    def predict(self, df):
        return [3.14]


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(api_main, "get_model", lambda: _StubModel())
    # Without this, /predict's logging call would append real entries into
    # the repo's logs/ directory as a side effect of running the test suite.
    monkeypatch.setattr(api_main, "PREDICTION_LOG_PATH", tmp_path / "predictions.jsonl")
    return TestClient(api_main.app)


def test_predict_returns_prediction(client):
    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    assert response.status_code == 200
    assert response.json() == {"prediction": pytest.approx(3.14)}


def test_predict_rejects_missing_field(client):
    bad_payload = dict(SAMPLE_PAYLOAD)
    del bad_payload["MedInc"]
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422  # pydantic validation error


def test_predict_returns_503_when_model_unavailable(monkeypatch):
    def _raise():
        raise RuntimeError("registry unreachable")

    monkeypatch.setattr(api_main, "get_model", _raise)
    client = TestClient(api_main.app)
    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    assert response.status_code == 503


def test_health_endpoint_reports_model_uri(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_uri"] == api_main.MODEL_URI


def test_predict_logs_the_request_and_prediction(client):
    client.post("/predict", json=SAMPLE_PAYLOAD)

    lines = api_main.PREDICTION_LOG_PATH.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1

    record = json.loads(lines[0])
    assert record["features"] == SAMPLE_PAYLOAD
    assert record["prediction"] == pytest.approx(3.14)
