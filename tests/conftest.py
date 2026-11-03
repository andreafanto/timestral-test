import pathlib
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "app"))


@pytest.fixture
def tiny_csv(tmp_path):
    ts = pd.date_range("2020-01-01", periods=40, freq="h")
    path = tmp_path / "series.csv"
    pd.DataFrame({"timestamp": ts, "value": range(40)}).to_csv(path, index=False)
    return str(path)


@pytest.fixture
def fake_forecast():
    def _forecast(api, values, horizon):
        return {"model": "test-model", "median": list(range(horizon)),
                "q10": list(range(horizon)), "q90": list(range(horizon))}
    return _forecast
