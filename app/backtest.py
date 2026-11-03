"""Backtest: split the series and evaluate forecast accuracy."""
import os
import sys
import argparse
from datetime import datetime
import numpy as np
from core import load_series, compute_metrics, save_outputs
from timestral_client import forecast

API = os.getenv("API", "http://localhost:8000")
CSV = os.getenv("CSV", "/data/series.csv")
OUTPUT = os.getenv("OUTPUT", "/output")


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
    run_dir = os.path.join(OUTPUT, "backtest", datetime.now().strftime("%Y-%m-%d_%H-%M"))
    os.makedirs(run_dir, exist_ok=True)
    save_outputs(payload, history, actual, metrics, output_dir=run_dir)
    print(f"Outputs saved to: {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest time series forecast using Timestral")
    parser.add_argument(
        "--horizon", required=True, type=int,
        help="Prediction length, minimum 1 maximum 1024",
    )
    args = parser.parse_args()
    main(args.horizon)