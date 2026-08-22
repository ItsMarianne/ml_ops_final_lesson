"""Typed loaders for the YAML files under config/.

Why this exists: the original script had experiment names, registry names,
and hyperparameter grids hardcoded inline in Python. Changing any of them
meant editing code and rebuilding a Docker image. Putting them in YAML and
loading them through pydantic models gives us two things for free:

1. A single source of truth that both the trainer and the API read (fixes
   the registry-name mismatch bug from the original project).
2. Fail-fast validation -- a typo'd or missing config key raises a clear
   pydantic ValidationError at startup, instead of a confusing KeyError or
   AttributeError deep inside a training run.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

# Repo root, computed from this file's location, so config paths in YAML
# (e.g. "data/raw/california_housing.csv") resolve the same way whether this
# runs from a laptop shell, a Docker container, or pytest.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_CONFIG_PATH = REPO_ROOT / "config" / "train.yaml"
DEFAULT_FEATURES_CONFIG_PATH = REPO_ROOT / "config" / "features.yaml"


class DataConfig(BaseModel):
    raw_path: Path
    target: str
    test_size: float = Field(gt=0.0, lt=1.0)
    random_state: int


class ModelSpec(BaseModel):
    """One entry under `models:` in train.yaml -- an estimator name plus the
    hyperparameter grid GridSearchCV should search for it."""

    estimator: str
    param_grid: dict[str, list[Any]]


class TrainConfig(BaseModel):
    experiment_name: str
    registry_name: str
    mlflow_tracking_uri: str
    data: DataConfig
    cv_folds: int = Field(gt=1)
    scoring: str
    promote_aliases: list[str]
    promotion_metric: str
    lower_is_better: bool
    models: dict[str, ModelSpec]

    @classmethod
    def load(cls, path: Path | None = None) -> TrainConfig:
        path = path or DEFAULT_TRAIN_CONFIG_PATH
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        # MLFLOW_TRACKING_URI is set by docker-compose per-environment (it
        # needs to be "http://mlflow:5000" inside the compose network, but
        # might be "http://localhost:5000" when running the trainer bare on
        # a laptop). Let the env var win over the YAML default rather than
        # forcing every environment to maintain its own copy of the file.
        env_uri = os.environ.get("MLFLOW_TRACKING_URI")
        if env_uri:
            raw["mlflow_tracking_uri"] = env_uri

        return cls(**raw)

    def resolved_raw_path(self) -> Path:
        """`data.raw_path` in YAML is repo-relative; resolve it against the
        repo root so callers don't need to know or care about cwd."""
        if self.data.raw_path.is_absolute():
            return self.data.raw_path
        return REPO_ROOT / self.data.raw_path


class FeaturesConfig(BaseModel):
    """The output of scripts/select_features.py: which raw columns to feed
    into the model, and why the rest were dropped."""

    target: str
    selected: list[str]
    dropped: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | None = None) -> FeaturesConfig:
        path = path or DEFAULT_FEATURES_CONFIG_PATH
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(**raw)
