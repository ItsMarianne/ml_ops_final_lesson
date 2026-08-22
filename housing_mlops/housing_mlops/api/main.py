"""FastAPI service that serves predictions from the model registered by the
trainer.

The model URI is derived from config/train.yaml's `registry_name` (the same
file the trainer reads) rather than being a separately hardcoded string in
this file. This is a direct fix for a bug in the original project: the
trainer registered under "housing-regression" while this API defaulted to
"housing-regressor" -- two different names that happened to look similar --
so the API could never actually find the model the trainer produced. With
both sides reading `registry_name` from the same config file, that class of
drift can't happen again.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from housing_ml.config import TrainConfig
from housing_ml.prediction_log import log_prediction

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Housing Price Regressor")

_config = TrainConfig.load()
mlflow.set_tracking_uri(_config.mlflow_tracking_uri)

# Which registry alias to serve (dev/uat/prod -- see promote_aliases in
# train.yaml). Defaults to "prod" but can be overridden per-deployment
# (e.g. a staging API instance serving the "uat" alias) without a code change.
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "prod")
MODEL_URI = f"models:/{_config.registry_name}@{MODEL_ALIAS}"

# Where served predictions get logged (see prediction_log.py). docker-compose
# mounts ./logs:/app/logs so these persist on the host across container
# restarts; this is the raw material a future data-drift check would need.
PREDICTION_LOG_PATH = Path(os.environ.get("PREDICTION_LOG_PATH", "logs/predictions.jsonl"))

_model = None  # loaded lazily -- see get_model() below


class HousingFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float


def get_model():
    """Lazily load the model on first request rather than at container
    startup. At startup time the trainer job may not have run yet (or may
    still be running) -- failing every /predict call with a clear 503 is
    much friendlier to operate than crash-looping the whole container."""
    global _model
    if _model is None:
        logger.info("Loading model from %s", MODEL_URI)
        _model = mlflow.pyfunc.load_model(MODEL_URI)
    return _model


@app.post("/predict")
def predict(inp: HousingFeatures):
    try:
        row = pd.DataFrame([inp.model_dump()])
        pred = get_model().predict(row)
    except Exception as e:
        raise HTTPException(
            503, f"model not available at '{MODEL_URI}' — run the trainer first ({e})"
        )

    prediction = float(pred[0])
    try:
        log_prediction(PREDICTION_LOG_PATH, inp.model_dump(), prediction)
    except OSError:
        # A logging failure (e.g. read-only filesystem) shouldn't take down
        # an otherwise-successful prediction -- but it should be visible in
        # the server logs, not silently swallowed.
        logger.exception("Failed to write prediction log entry to %s", PREDICTION_LOG_PATH)

    return {"prediction": prediction}


@app.post("/reload")
def reload_model():
    """Call this after retraining to pick up the newest version promoted to
    this API instance's alias, without restarting the container."""
    global _model
    _model = None
    return {"status": "model will reload on next /predict", "model_uri": MODEL_URI}


@app.get("/health")
def health():
    return {"status": "ok", "model_uri": MODEL_URI, "model_loaded": _model is not None}
