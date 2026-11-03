"""Predict: use all available history and forecast future points."""
import os
import sys
import argparse
from datetime import datetime
from core import load_series, save_predictions
from timestral_client import forecast

API = os.getenv("API", "http://localhost:8000")
CSV = os.getenv("CSV", "/data/series.csv")
OUTPUT = os.getenv("OUTPUT", "/output")


def main(horizon):
    df = load_series(CSV)

    if len(df) < 32:
        sys.exit("The number of datapoints in CSV should be >= 32")

    payload = forecast(
        API,
        df["value"].astype(float).tolist(),
        horizon,
    )

    run_dir = os.path.join(OUTPUT, "prediction", datetime.now().strftime("%Y-%m-%d_%H-%M"))
    os.makedirs(run_dir, exist_ok=True)
    save_predictions(df, payload, output_dir=run_dir)
    print(f"Outputs saved to: {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict future time series data using Timestral")
    parser.add_argument(
        "--horizon", required=True, type=int,
        help="Prediction length, minimum 1 maximum 1024",
    )
    args = parser.parse_args()
    main(args.horizon)