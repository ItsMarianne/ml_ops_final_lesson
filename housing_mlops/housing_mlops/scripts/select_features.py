"""Feature-selection analysis for the housing regressor.

What this does and why it's a separate script (not part of train.py):
picking which features to drop is a judgment call that a human should look
at, not something that should silently change every time the training job
runs. So this script is a standalone, run-on-demand analysis
(`make select-features`) that:

  1. Screens every raw feature's linear correlation with the target.
  2. Fits a baseline RandomForest and computes permutation importance for
     every feature on a held-out test split.
  3. Flags a feature as droppable only when BOTH signals agree it contributes
     little (weak correlation AND weak permutation importance) -- relying on
     either signal alone is risky: a feature can have near-zero linear
     correlation but still matter to a nonlinear model (or vice versa).
  4. Writes the resulting selected/dropped lists -- with the actual numbers
     that justified each decision -- to config/features.yaml.
  5. Saves two plots (correlation, permutation importance) to
     reports/figures/ so the decision can be sanity-checked visually.

Run it with: `make select-features` (or `python scripts/select_features.py`
from the repo root with the project's dependencies installed).

Nothing here is wired into the training pipeline automatically -- you
re-run this script, look at the output, and only then does the training
pipeline (which reads config/features.yaml) start using the new feature
list.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
from sklearn.inspection import permutation_importance

# Allow running this script directly (`python scripts/select_features.py`)
# without having installed the package first, by adding src/ to the path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from housing_ml.config import DEFAULT_FEATURES_CONFIG_PATH, REPO_ROOT, TrainConfig
from housing_ml.data import load_raw, train_test_split_df
from housing_ml.models import build_pipeline

# --- Thresholds for the "weak signal" flags below ---------------------------
# These are deliberately simple, human-reviewable heuristics rather than a
# single opaque score. A feature is only a *candidate* for dropping if it
# trips both:
#   - CORRELATION_THRESHOLD: |Pearson r| with the target below this counts
#     as "weak linear relationship".
#   - IMPORTANCE_RATIO_THRESHOLD: permutation importance below this fraction
#     of the single most important feature's importance counts as
#     "contributes little to this model's predictions".
CORRELATION_THRESHOLD = 0.10
IMPORTANCE_RATIO_THRESHOLD = 0.05

FIGURES_DIR = REPO_ROOT / "reports" / "figures"


def correlation_with_target(df: pd.DataFrame, feature_columns: list[str], target: str) -> pd.Series:
    """Pearson correlation of each raw feature with the target, sorted by
    absolute strength (strongest first)."""
    corr = df[feature_columns + [target]].corr()[target].drop(target)
    return corr.reindex(corr.abs().sort_values(ascending=False).index)


def permutation_importance_scores(
    df: pd.DataFrame,
    feature_columns: list[str],
    target: str,
    test_size: float,
    random_state: int,
) -> pd.Series:
    """Fit a RandomForest baseline and compute permutation importance on the
    held-out test split.

    RandomForest is used as the baseline here (rather than the eventual
    winning model, which isn't known yet) because it captures non-linear
    feature/target relationships without needing feature scaling to be
    tuned first, making it a reasonable general-purpose importance probe.
    """
    X = df[feature_columns]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split_df(
        X, y, test_size=test_size, random_state=random_state
    )

    pipeline = build_pipeline("random_forest", random_state=random_state)
    pipeline.fit(X_train, y_train)

    result = permutation_importance(
        pipeline,
        X_test,
        y_test,
        n_repeats=10,
        random_state=random_state,
        n_jobs=-1,
        scoring="neg_root_mean_squared_error",
    )
    importances = pd.Series(result.importances_mean, index=feature_columns)
    return importances.sort_values(ascending=False)


def plot_correlation(corr: pd.Series, path: Path) -> None:
    plt.figure(figsize=(8, 5))
    sns.barplot(x=corr.values, y=corr.index, hue=corr.index, palette="vlag", legend=False)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title("Correlation of each feature with MedHouseVal")
    plt.xlabel("Pearson correlation")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def plot_importance(importances: pd.Series, path: Path) -> None:
    plt.figure(figsize=(8, 5))
    sns.barplot(
        x=importances.values, y=importances.index, hue=importances.index,
        palette="viridis", legend=False,
    )
    plt.title("Permutation importance (RandomForest baseline, drop in -RMSE)")
    plt.xlabel("Mean importance")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def decide_drops(
    corr: pd.Series, importances: pd.Series
) -> tuple[list[str], dict[str, str]]:
    """Apply the "weak on both signals" rule and build a human-readable
    reason string for every dropped feature."""
    max_importance = importances.abs().max()
    dropped: dict[str, str] = {}

    for feature in corr.index:
        weak_corr = abs(corr[feature]) < CORRELATION_THRESHOLD
        weak_importance = (
            importances[feature] < IMPORTANCE_RATIO_THRESHOLD * max_importance
        )
        if weak_corr and weak_importance:
            dropped[feature] = (
                f"weak correlation with target (r={corr[feature]:.3f}) and "
                f"low permutation importance ({importances[feature]:.4f}, "
                f"{importances[feature] / max_importance:.1%} of the top feature)"
            )

    selected = [f for f in corr.index if f not in dropped]
    return selected, dropped


def main() -> None:
    config = TrainConfig.load()
    df = load_raw(config.resolved_raw_path())
    target = config.data.target
    all_features = [c for c in df.columns if c != target]

    print(f"Analyzing {len(all_features)} candidate features against target '{target}'...\n")

    corr = correlation_with_target(df, all_features, target)
    print("Correlation with target:")
    print(corr.to_string(float_format=lambda v: f"{v:+.3f}"))

    importances = permutation_importance_scores(
        df, all_features, target, config.data.test_size, config.data.random_state
    )
    print("\nPermutation importance (RandomForest baseline):")
    print(importances.to_string(float_format=lambda v: f"{v:.4f}"))

    selected, dropped = decide_drops(corr, importances)

    plot_correlation(corr, FIGURES_DIR / "correlation_with_target.png")
    plot_importance(importances, FIGURES_DIR / "permutation_importance.png")

    features_yaml = {
        "target": target,
        "selected": selected,
        "dropped": dropped,
    }
    with open(DEFAULT_FEATURES_CONFIG_PATH, "w", encoding="utf-8") as f:
        f.write(
            "# Auto-generated by scripts/select_features.py -- re-run that script to\n"
            "# regenerate rather than hand-editing the selected/dropped lists.\n"
        )
        yaml.safe_dump(features_yaml, f, sort_keys=False)

    print(f"\nSelected {len(selected)}/{len(all_features)} features, dropped {len(dropped)}:")
    for feature, reason in dropped.items():
        print(f"  - {feature}: {reason}")
    print(f"\nWrote {DEFAULT_FEATURES_CONFIG_PATH}")
    print(f"Wrote plots to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
