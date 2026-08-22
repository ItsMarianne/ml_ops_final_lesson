"""Append-only JSONL logging of served predictions.

This is intentionally the minimum needed to seed future data-drift
monitoring, not a monitoring system itself: it durably records what the API
was actually asked to predict on and what it returned, one line per
request. That's the raw material a later drift check (e.g. comparing the
distribution of `features` here against data/raw/california_housing.csv
with a KS test or PSI) would need -- this module doesn't compute or flag
anything about drift on its own.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def log_prediction(log_path: Path, features: dict, prediction: float) -> None:
    """Append one record: UTC timestamp, the input features, and the
    prediction returned for them.

    Creates the parent directory on first use rather than requiring it to
    already exist -- convenient locally; in Docker, docker-compose mounts
    logs/ as a volume so these records land on the host and survive
    container restarts.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "features": features,
        "prediction": prediction,
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
