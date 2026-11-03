import importlib

import predict


def test_predict_writes_to_timestamped_prediction_folder(tmp_path, monkeypatch, tiny_csv, fake_forecast):
    fresh = tmp_path / "fresh"          # does not exist yet — pins auto-mkdir
    monkeypatch.setattr(predict, "CSV", tiny_csv)
    monkeypatch.setattr(predict, "OUTPUT", str(fresh))
    monkeypatch.setattr(predict, "forecast", fake_forecast)
    predict.main(48)
    runs = list((fresh / "prediction").glob("????-??-??_??-??"))
    assert len(runs) == 1
    assert (runs[0] / "forecast.csv").exists()
    assert (runs[0] / "metrics.json").exists()


def test_predict_output_default_is_container_path(monkeypatch):
    monkeypatch.delenv("OUTPUT", raising=False)
    importlib.reload(predict)
    assert predict.OUTPUT == "/output"


def test_predict_writes_prediction_plot(tmp_path, monkeypatch, tiny_csv, fake_forecast):
    fresh = tmp_path / "fresh"
    monkeypatch.setattr(predict, "CSV", tiny_csv)
    monkeypatch.setattr(predict, "OUTPUT", str(fresh))
    monkeypatch.setattr(predict, "forecast", fake_forecast)
    predict.main(48)
    runs = list((fresh / "prediction").glob("????-??-??_??-??"))
    png = runs[0] / "prediction.png"
    assert png.exists() and png.stat().st_size > 0
