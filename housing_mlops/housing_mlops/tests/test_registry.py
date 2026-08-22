"""Tests for the promotion-safeguard decision logic.

Only `is_improvement` and `to_mlflow_metric_names` are tested here -- pure
functions with no MLflow calls. `register_and_promote` and `get_alias_metric`
talk to a real MLflow tracking server/registry and are exercised indirectly
via the manual end-to-end check (`make smoke-test`) instead of being mocked
here, which would mostly just test the mock.
"""

from housing_ml.registry import is_improvement, to_mlflow_metric_names


def test_first_promotion_always_happens_when_alias_is_empty():
    assert is_improvement(candidate_metric=999.0, current_metric=None, lower_is_better=True)


def test_lower_is_better_promotes_only_when_candidate_is_smaller():
    assert is_improvement(0.40, current_metric=0.50, lower_is_better=True)
    assert not is_improvement(0.60, current_metric=0.50, lower_is_better=True)


def test_higher_is_better_promotes_only_when_candidate_is_larger():
    assert is_improvement(0.90, current_metric=0.80, lower_is_better=False)
    assert not is_improvement(0.70, current_metric=0.80, lower_is_better=False)


def test_tie_does_not_promote():
    # Neither strictly better -- a tie shouldn't trigger a version bump for
    # no reason.
    assert not is_improvement(0.50, current_metric=0.50, lower_is_better=True)
    assert not is_improvement(0.50, current_metric=0.50, lower_is_better=False)


def test_to_mlflow_metric_names_maps_evaluate_keys():
    mapped = to_mlflow_metric_names({"RMSE": 0.5, "MAE": 0.4, "R2": 0.9})
    assert mapped == {"test_rmse": 0.5, "test_mae": 0.4, "test_r2": 0.9}
