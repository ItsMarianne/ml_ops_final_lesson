"""All MLflow interaction: experiment run logging and model registry
promotion.

Isolating this from train.py means the training loop (which decides *what*
to train) doesn't need to know the details of *how* a run gets logged or
*how* the winning model gets promoted -- and it means these functions can be
tested independently against a local/temp MLflow tracking URI without
running a full GridSearchCV.
"""

from __future__ import annotations

import logging

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def flatten_grid_search_params(param_dict: dict) -> dict:
    """GridSearchCV's `best_params_` keys look like "model__alpha" (the
    sklearn Pipeline step-name prefix). Strip the prefix so MLflow's UI
    shows the plain hyperparameter name ("alpha") instead of the
    implementation detail of how the pipeline step happens to be named."""
    return {key.split("__", 1)[-1]: value for key, value in param_dict.items()}


def to_mlflow_metric_names(metrics: dict[str, float]) -> dict[str, float]:
    """Map evaluate.py's metric keys (RMSE/MAE/R2) to the names they're
    logged under in MLflow (test_rmse/test_mae/test_r2).

    Shared by log_run (writing metrics) and train.py (comparing a candidate
    against what's currently promoted) so both sides agree on the name a
    given metric is stored under -- config/train.yaml's `promotion_metric`
    only makes sense if it's spelled the same way here as in MLflow.
    """
    return {
        "test_rmse": metrics["RMSE"],
        "test_mae": metrics["MAE"],
        "test_r2": metrics["R2"],
    }


def log_run(
    *,
    run_name: str,
    pipeline: Pipeline,
    best_params: dict,
    metrics: dict[str, float],
    X_train: pd.DataFrame,
    target: str,
    dataset_name: str = "california_housing",
) -> tuple[str, str]:
    """Log one completed training run to MLflow: tags, params, metrics, and
    the fitted model artifact. Returns (run_id, model_uri) so the caller can
    later decide whether this run's model is the one to register."""
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({"dataset": dataset_name, "task": "regression", "target": target})
        mlflow.log_params(flatten_grid_search_params(best_params))
        mlflow.log_metrics(to_mlflow_metric_names(metrics))
        signature = infer_signature(X_train, pipeline.predict(X_train))
        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            signature=signature,
            input_example=X_train.iloc[:1],
        )
        return run.info.run_id, model_info.model_uri


def is_improvement(
    candidate_metric: float, current_metric: float | None, lower_is_better: bool
) -> bool:
    """Pure decision logic behind promotion, pulled out of
    register_and_promote so it can be unit tested without a real MLflow
    tracking server: is `candidate_metric` good enough to take over an
    alias currently scored at `current_metric`?

    `current_metric=None` means the alias has nothing registered yet --
    always promote in that case, since there's nothing to regress against.
    """
    if current_metric is None:
        return True
    if lower_is_better:
        return candidate_metric < current_metric
    return candidate_metric > current_metric


def get_alias_metric(
    client: MlflowClient, registry_name: str, alias: str, metric_key: str
) -> float | None:
    """Look up the metric value logged for whatever model version is
    currently aliased `alias`. Returns None if the alias doesn't exist yet
    (a brand-new registry) or its run doesn't have that metric logged."""
    try:
        current_version = client.get_model_version_by_alias(registry_name, alias)
    except MlflowException:
        return None
    run = client.get_run(current_version.run_id)
    return run.data.metrics.get(metric_key)


def register_and_promote(
    *,
    run_id: str,
    registry_name: str,
    aliases: list[str],
    candidate_metrics: dict[str, float],
    promotion_metric: str,
    lower_is_better: bool,
) -> tuple[int, list[str], list[tuple[str, float]]]:
    """Register the model produced by `run_id`, then advance each alias in
    `aliases` to it -- but only where the candidate actually beats whatever
    is currently aliased there (see is_improvement). This is what stops a
    noisy or unlucky training run from silently regressing an alias, most
    importantly "prod".

    Returns (registered_version, promoted_aliases, skipped_aliases) where
    skipped_aliases is [(alias, current_metric_it_lost_to), ...] so the
    caller can log exactly why an alias didn't move.
    """
    registered = mlflow.register_model(f"runs:/{run_id}/model", registry_name)
    client = MlflowClient()
    candidate_metric = candidate_metrics[promotion_metric]

    promoted: list[str] = []
    skipped: list[tuple[str, float]] = []
    for alias in aliases:
        current_metric = get_alias_metric(client, registry_name, alias, promotion_metric)
        if is_improvement(candidate_metric, current_metric, lower_is_better):
            client.set_registered_model_alias(registry_name, alias, registered.version)
            promoted.append(alias)
        else:
            skipped.append((alias, current_metric))

    return int(registered.version), promoted, skipped
