import os
import sys
import json
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

API = os.getenv("API", "http://localhost:8000")
CSV = os.getenv("CSV", "/data/series.csv")
HORIZON = int(os.getenv("HORIZON", "48"))
OUT = os.getenv("OUT", "/output/forecast.png")


def main():
    df = pd.read_csv(CSV, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    if "value" not in df.columns:
        sys.exit("Il CSV deve contenere le colonne 'timestamp' e 'value'.")

    if len(df) <= HORIZON + 32:
        sys.exit("Servono almeno HORIZON + 32 righe nel CSV.")

    split = len(df) - HORIZON
    history = df.iloc[:split]
    actual = df.iloc[split:]

    response = requests.post(
        f"{API}/forecast",
        json={
            "values": history["value"].astype(float).tolist(),
            "prediction_length": HORIZON,
        },
        timeout=300,
    )
    response.raise_for_status()
    payload = response.json()

    median = np.asarray(payload["median"], dtype=float)
    q10 = np.asarray(payload["q10"], dtype=float)
    q90 = np.asarray(payload["q90"], dtype=float)
    y = actual["value"].to_numpy(dtype=float)

    mae = mean_absolute_error(y, median)
    rmse = float(np.sqrt(mean_squared_error(y, median)))
    mape = float(np.mean(np.abs((y - median) / np.where(y == 0, np.nan, y))) * 100)

    metrics = {
        "model": payload["model"],
        "horizon": HORIZON,
        "history_points": len(history),
        "mae": mae,
        "rmse": rmse,
        "mape_percent": mape,
    }

    with open("/output/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))

    plt.figure(figsize=(14, 7))
    tail = df.iloc[max(0, split - 200):]

    plt.plot(
        tail["timestamp"], tail["value"],
        label="Storico", color="#2563eb", linewidth=1.8,
    )
    plt.plot(
        actual["timestamp"], y,
        label="Valore reale", color="#111827", linewidth=2,
    )
    plt.plot(
        actual["timestamp"], median,
        label="Timestral mediana", color="#dc2626", linewidth=2,
    )
    plt.fill_between(
        actual["timestamp"], q10, q90,
        color="#dc2626", alpha=0.18, label="Intervallo 10-90%",
    )

    plt.axvline(history["timestamp"].iloc[-1], color="gray", linestyle="--", linewidth=1)
    plt.title(f"Timestral backtest — MAE {mae:.3f}, RMSE {rmse:.3f}, MAPE {mape:.2f}%")
    plt.xlabel("Tempo")
    plt.ylabel("Valore")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT, dpi=160)

    pd.DataFrame({
        "timestamp": actual["timestamp"],
        "actual": y,
        "forecast_median": median,
        "q10": q10,
        "q90": q90,
    }).to_csv("/output/forecast_vs_actual.csv", index=False)


if __name__ == "__main__":
    main()