"""Backtest: split the series and evaluate forecast accuracy."""
import os
import sys
import argparse
import numpy as np
from core import load_series, compute_metrics, save_outputs
from timestral_client import forecast

API = os.getenv("API", "http://localhost:8000")
CSV = os.getenv("CSV", "/data/series.csv")


def main(horizon):
    df = load_series(CSV)

    if len(df) <= horizon + 32:
        sys.exit("CSV must have at least HORIZON + 32 rows.")

    split = len(df) - horizon
    history = df.iloc[:split]
    actual = df.iloc[split:]

    payload = forecast(
        API,
        history["value"].astype(float).tolist(),
        horizon,
    )

    median = np.asarray(payload["median"], dtype=float)
    metrics = compute_metrics(actual, median)
    save_outputs(payload, history, actual, metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest time series forecast using Timestral")
    parser.add_argument(
        "--horizon", required=True, type=int,
        help="Prediction length, minimum 1 maximum 1024",
    )
    args = parser.parse_args()
    main(args.horizon)