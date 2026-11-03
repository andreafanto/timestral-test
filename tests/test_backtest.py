import importlib

import backtest
import predict


def test_backtest_and_predict_outputs_coexist(tmp_path, monkeypatch, tiny_csv, fake_forecast):
    monkeypatch.setattr(predict, "CSV", tiny_csv)
    monkeypatch.setattr(predict, "OUTPUT", str(tmp_path))
    monkeypatch.setattr(predict, "forecast", fake_forecast)
    monkeypatch.setattr(backtest, "CSV", tiny_csv)
    monkeypatch.setattr(backtest, "OUTPUT", str(tmp_path))
    monkeypatch.setattr(backtest, "forecast", fake_forecast)

    predict.main(48)
    backtest.main(2)

    pred_runs = list((tmp_path / "prediction").glob("????-??-??_??-??"))
    back_runs = list((tmp_path / "backtest").glob("????-??-??_??-??"))
    assert len(pred_runs) == 1 and len(back_runs) == 1
    assert (pred_runs[0] / "metrics.json").exists()     # not clobbered
    assert (back_runs[0] / "metrics.json").exists()
    assert (back_runs[0] / "forecast_vs_actual.csv").exists()
    assert (back_runs[0] / "forecast.png").exists()


def test_backtest_output_default_is_container_path(monkeypatch):
    monkeypatch.delenv("OUTPUT", raising=False)
    importlib.reload(backtest)
    assert backtest.OUTPUT == "/output"
