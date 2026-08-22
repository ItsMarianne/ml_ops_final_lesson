"""Training entrypoint: for each model in config/train.yaml, run
GridSearchCV, log the result to MLflow, then register + promote whichever
model scored best on the held-out test set.

This module intentionally contains no model definitions, no metric
formulas, and no MLflow API calls of its own -- it only orchestrates calls
into data.py / models.py / evaluate.py / registry.py. That means this file
reads as "the sequence of steps", and any of those steps can be tested or
changed in isolation without touching this file.
"""

from __future__ import annotations

import logging

import mlflow
from sklearn.model_selection import GridSearchCV

from housing_ml.config import FeaturesConfig, ModelSpec, TrainConfig
from housing_ml.data import load_raw, split_features_target, train_test_split_df
from housing_ml.evaluate import evaluate_regression
from housing_ml.features import selected_columns
from housing_ml.models import build_pipeline
from housing_ml.registry import log_run, register_and_promote, to_mlflow_metric_names

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_search(
    name: str, spec: ModelSpec, config: TrainConfig, X_train, y_train
) -> GridSearchCV:
    """Build and fit one GridSearchCV for a single model spec from
    train.yaml."""
    pipeline = build_pipeline(spec.estimator, random_state=config.data.random_state)
    search = GridSearchCV(
        pipeline,
        param_grid=spec.param_grid,
        scoring=config.scoring,
        cv=config.cv_folds,
        n_jobs=-1,
    )
    logger.info("Fitting '%s' (estimator=%s)...", name, spec.estimator)
    search.fit(X_train, y_train)
    return search


def main() -> None:
    config = TrainConfig.load()
    features = FeaturesConfig.load()

    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.experiment_name)

    df = load_raw(config.resolved_raw_path())
    feature_columns = selected_columns(features, df)
    logger.info(
        "Training on %d/%d columns (%d dropped by feature selection)",
        len(feature_columns),
        len(df.columns) - 1,
        len(features.dropped),
    )

    X, y = split_features_target(df, feature_columns, config.data.target)
    X_train, X_test, y_train, y_test = train_test_split_df(
        X, y, config.data.test_size, config.data.random_state
    )

    results = {}
    for name, spec in config.models.items():
        search = run_search(name, spec, config, X_train, y_train)
        best_pipeline = search.best_estimator_
        metrics = evaluate_regression(y_test, best_pipeline.predict(X_test))

        run_id, model_uri = log_run(
            run_name=name,
            pipeline=best_pipeline,
            best_params=search.best_params_,
            metrics=metrics,
            X_train=X_train,
            target=config.data.target,
        )
        results[name] = {"metrics": metrics, "run_id": run_id, "model_uri": model_uri}
        logger.info("'%s' test RMSE=%.4f", name, metrics["RMSE"])

    best_name = min(results, key=lambda n: results[n]["metrics"]["RMSE"])
    best = results[best_name]
    logger.info("Best model: %s (test_rmse=%.4f)", best_name, best["metrics"]["RMSE"])

    candidate_metrics = to_mlflow_metric_names(best["metrics"])
    version, promoted, skipped = register_and_promote(
        run_id=best["run_id"],
        registry_name=config.registry_name,
        aliases=config.promote_aliases,
        candidate_metrics=candidate_metrics,
        promotion_metric=config.promotion_metric,
        lower_is_better=config.lower_is_better,
    )
    logger.info("Registered %s v%d", config.registry_name, version)
    if promoted:
        logger.info("Promoted to: %s", ", ".join(promoted))
    for alias, current_metric in skipped:
        logger.warning(
            "Did NOT promote to '%s': candidate %s=%.4f did not beat current %s=%.4f",
            alias,
            config.promotion_metric,
            candidate_metrics[config.promotion_metric],
            config.promotion_metric,
            current_metric,
        )


if __name__ == "__main__":
    main()
