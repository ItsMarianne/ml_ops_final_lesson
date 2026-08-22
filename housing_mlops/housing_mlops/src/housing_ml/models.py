"""Model definitions: which estimators exist, and how each is wrapped into a
preprocessing + model sklearn Pipeline.

Splitting this out of train.py means the feature-selection script
(scripts/select_features.py) can reuse the exact same pipeline-building logic
to fit its own baseline model, instead of duplicating "RobustScaler + Ridge"
in two places that could quietly drift apart.
"""

from __future__ import annotations

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

# Maps the `estimator` name used in config/train.yaml to a zero-argument
# factory that builds a fresh, unfitted estimator instance. A factory (not a
# shared instance) is important -- GridSearchCV clones estimators internally,
# but building a fresh one per pipeline avoids any risk of shared mutable
# state across models when this dict is reused across multiple runs/tests.
ESTIMATOR_FACTORIES = {
    "ridge": lambda random_state: Ridge(random_state=random_state),
    "random_forest": lambda random_state: RandomForestRegressor(
        random_state=random_state, n_jobs=-1
    ),
    "gradient_boosting": lambda random_state: GradientBoostingRegressor(
        random_state=random_state
    ),
}


def build_pipeline(estimator_name: str, random_state: int) -> Pipeline:
    """Build the standard "scale then model" pipeline for one estimator.

    RobustScaler (median/IQR-based) is used instead of StandardScaler
    because median home values and some of the housing features (e.g.
    AveOccup) have long right tails -- RobustScaler is less thrown off by
    those outliers than a mean/std-based scaler would be.
    """
    if estimator_name not in ESTIMATOR_FACTORIES:
        raise ValueError(
            f"Unknown estimator '{estimator_name}'. "
            f"Available: {sorted(ESTIMATOR_FACTORIES)}"
        )
    model = ESTIMATOR_FACTORIES[estimator_name](random_state)
    return Pipeline([("scaler", RobustScaler()), ("model", model)])
