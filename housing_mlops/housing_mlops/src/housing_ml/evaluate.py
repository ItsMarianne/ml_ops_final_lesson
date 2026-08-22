"""Regression evaluation metrics.

A one-function module on purpose: this is the kind of tiny, pure piece of
logic that's easy to lose inside a big training script but trivial (and
valuable) to unit test on its own once it's separated out.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_regression(y_true, y_pred) -> dict[str, float]:
    """Return the standard set of regression metrics we track in MLflow.

    RMSE is in the same units as the target (used for model comparison /
    selecting the best model); MAE is more robust to outliers and easier to
    explain to non-technical stakeholders ("average dollar error"); R^2
    gives a scale-free sense of how much variance is explained.
    """
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }
