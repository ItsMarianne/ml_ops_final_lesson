import numpy as np

from housing_ml.evaluate import evaluate_regression


def test_perfect_predictions_give_zero_error_and_r2_one():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    metrics = evaluate_regression(y_true, y_true.copy())

    assert metrics["RMSE"] == 0.0
    assert metrics["MAE"] == 0.0
    assert metrics["R2"] == 1.0


def test_known_rmse_and_mae_for_constant_offset():
    # Every prediction is off by exactly 1 -> MAE = 1, RMSE = 1.
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 3.0, 4.0])
    metrics = evaluate_regression(y_true, y_pred)

    assert metrics["MAE"] == 1.0
    assert metrics["RMSE"] == 1.0


def test_metrics_are_returned_as_plain_floats():
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([1.5, 1.5])
    metrics = evaluate_regression(y_true, y_pred)

    for value in metrics.values():
        assert isinstance(value, float)
