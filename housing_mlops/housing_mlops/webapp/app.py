"""Minimal Flask front-end for the housing price API.

This is deliberately thin: it has no model logic, no MLflow, no database --
it just renders an HTML form and forwards the submitted values to the
FastAPI service's /predict endpoint over plain HTTP. It exists to demonstrate
consuming the API from a separate client, not as a second way to do
prediction.
"""

from __future__ import annotations

import logging
import os

import requests
from flask import Flask, render_template, request

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

# The FastAPI service's base URL. Defaults to localhost for running this
# app bare on a laptop; docker-compose overrides it to "http://api:8000" so
# the two containers can reach each other by service name.
API_URL = os.environ.get("API_URL", "http://localhost:8000")

# One entry per HousingFeatures field the API expects (see api/main.py).
# Keeping this list here (rather than hardcoding 8 form fields in the
# template) means adding/removing a field is a one-place change.
FEATURE_FIELDS = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]

# Pre-filled with a real row from the dataset purely so the form isn't blank
# on first load -- convenient for a demo, not a default prediction of anything.
EXAMPLE_VALUES = {
    "MedInc": 8.3,
    "HouseAge": 41,
    "AveRooms": 6.9,
    "AveBedrms": 1.0,
    "Population": 322,
    "AveOccup": 2.5,
    "Latitude": 37.9,
    "Longitude": -122.2,
}


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    # Fall back to the example values so the form re-populates with
    # whatever the user last submitted, or the example on first load.
    values = EXAMPLE_VALUES.copy()

    if request.method == "POST":
        values = {field: request.form.get(field, "") for field in FEATURE_FIELDS}
        try:
            payload = {field: float(values[field]) for field in FEATURE_FIELDS}
        except ValueError:
            error = "All fields must be numbers."
        else:
            try:
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
                response.raise_for_status()
                prediction = response.json()["prediction"]
            except requests.RequestException as e:
                logger.exception("Call to %s/predict failed", API_URL)
                error = f"Could not reach the prediction API at {API_URL} ({e})"

    return render_template(
        "index.html",
        fields=FEATURE_FIELDS,
        values=values,
        prediction=prediction,
        error=error,
        api_url=API_URL,
    )


@app.route("/health")
def health():
    return {"status": "ok", "api_url": API_URL}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
