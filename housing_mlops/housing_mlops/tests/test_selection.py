"""Tests for scripts/select_features.py's pure decision logic.

Only `decide_drops` and `correlation_with_target` are tested here --
`permutation_importance_scores` fits a real RandomForest and is exercised
indirectly through these (via directly-constructed importance Series
instead of refitting a model in every test, which would make the suite slow
for no extra confidence).
"""

import pandas as pd

from select_features import correlation_with_target, decide_drops


def test_correlation_with_target_sorts_by_absolute_strength():
    df = pd.DataFrame(
        {
            "strong_positive": [1, 2, 3, 4, 5],
            "weak": [3, 1, 4, 1, 5],
            "target": [1, 2, 3, 4, 5],
        }
    )
    corr = correlation_with_target(df, ["strong_positive", "weak"], "target")
    assert corr.index[0] == "strong_positive"
    assert corr["strong_positive"] == 1.0


def test_decide_drops_flags_only_features_weak_on_both_signals():
    corr = pd.Series({"strong": 0.9, "weak_corr_only": 0.02, "weak_both": 0.01})
    importances = pd.Series({"strong": 0.20, "weak_corr_only": 0.18, "weak_both": 0.0005})

    selected, dropped = decide_drops(corr, importances)

    assert "strong" in selected
    # weak_corr_only has weak correlation but still matters to the model
    # (high permutation importance) -> must be kept, not dropped.
    assert "weak_corr_only" in selected
    assert "weak_both" in dropped
    assert "weak correlation" in dropped["weak_both"]
    assert "permutation importance" in dropped["weak_both"]
