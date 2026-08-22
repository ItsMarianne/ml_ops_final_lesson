import json

from housing_ml.prediction_log import log_prediction


def test_log_prediction_appends_one_json_line(tmp_path):
    log_path = tmp_path / "logs" / "predictions.jsonl"

    log_prediction(log_path, {"MedInc": 8.3, "HouseAge": 41}, 4.1)
    log_prediction(log_path, {"MedInc": 3.1, "HouseAge": 12}, 1.9)

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2

    first = json.loads(lines[0])
    assert first["features"] == {"MedInc": 8.3, "HouseAge": 41}
    assert first["prediction"] == 4.1
    assert "timestamp" in first


def test_log_prediction_creates_parent_directory(tmp_path):
    log_path = tmp_path / "does" / "not" / "exist" / "predictions.jsonl"
    log_prediction(log_path, {"MedInc": 1.0}, 2.0)
    assert log_path.exists()
