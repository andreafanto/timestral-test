"""Failing tests that reproduce the four confirmed bugs."""
import numpy as np
import pandas as pd
import pytest

from core import compute_metrics, load_series, save_predictions


def _write_csv(path, text):
    path.write_text(text)
    return str(path)


def test_load_series_missing_timestamp_column_exits_with_message(tmp_path):
    """A CSV without 'timestamp' must exit cleanly, not raise KeyError."""
    csv = _write_csv(tmp_path / "bad.csv", "value\n1\n2\n")
    with pytest.raises(SystemExit):
        load_series(csv)


def test_load_series_deduplicates_duplicate_timestamps(tmp_path):
    """DST fall-back duplicates must not be fed to the model twice."""
    csv = _write_csv(
        tmp_path / "dup.csv",
        "timestamp,value\n"
        "2020-01-01 00:00:00,1\n"
        "2020-01-01 01:00:00,2\n"
        "2020-01-01 01:00:00,3\n"
        "2020-01-01 02:00:00,4\n",
    )
    df = load_series(csv)
    assert not df["timestamp"].duplicated().any()


def test_save_predictions_generates_hourly_timestamps(tmp_path):
    """Hourly data with DST gaps must yield hourly forecast timestamps.

    Reproduces the real data shape: one 2h spring-forward gap and one
    duplicated fall-back timestamp, which make pd.infer_freq return None.
    """
    ts = pd.date_range("2020-01-01", periods=48, freq="h").delete(20)
    ts = ts.append(pd.DatetimeIndex([ts[5]]))  # duplicate (fall-back)
    ts = ts.sort_values()
    history = pd.DataFrame({"timestamp": ts, "value": np.arange(len(ts), dtype=float)})

    payload = {
        "model": "test-model",
        "median": list(range(48)),
        "q10": list(range(48)),
        "q90": list(range(48)),
    }
    save_predictions(history, payload, output_dir=str(tmp_path))

    out = pd.read_csv(tmp_path / "forecast.csv", parse_dates=["timestamp"])
    assert len(out) == 48
    steps = out["timestamp"].diff().dropna()
    assert (steps == pd.Timedelta("1h")).all(), f"not hourly: {steps.unique()}"
    assert out["timestamp"].iloc[0] == history["timestamp"].iloc[-1] + pd.Timedelta("1h")


def test_compute_metrics_zero_actual_produces_finite_mape():
    """MAPE must not become NaN when an actual value is zero."""
    actual = pd.DataFrame({"value": [10.0, 0.0, 20.0]})
    median = np.array([11.0, 1.0, 18.0])
    metrics = compute_metrics(actual, median)
    assert not np.isnan(metrics["mape_percent"])
