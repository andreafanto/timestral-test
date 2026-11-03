import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error


def load_series(csv_path):
    """Load and validate the CSV time series."""
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    if "value" not in df.columns:
        sys.exit("CSV must contain 'timestamp' and 'value' columns.")
    return df


def compute_metrics(actual, median):
    """Compute MAE, RMSE, MAPE from actual and predicted values."""
    y = actual["value"].to_numpy(dtype=float)
    mae = mean_absolute_error(y, median)
    rmse = float(np.sqrt(mean_squared_error(y, median)))
    mape = float(np.mean(np.abs((y - median) / np.where(y == 0, np.nan, y))) * 100)
    return {
        "mae": mae,
        "rmse": rmse,
        "mape_percent": mape,
    }


def plot_forecast(df, split, actual, median, q10, q90, metrics, out_path):
    """Render and save the forecast vs actual plot."""
    plt.figure(figsize=(14, 7))
    tail = df.iloc[max(0, split - 200):]

    plt.plot(
        tail["timestamp"], tail["value"],
        label="History", color="#2563eb", linewidth=1.8,
    )
    plt.plot(
        actual["timestamp"], actual["value"].to_numpy(dtype=float),
        label="Actual value", color="#111827", linewidth=2,
    )
    plt.plot(
        actual["timestamp"], median,
        label="Timestral median", color="#dc2626", linewidth=2,
    )
    plt.fill_between(
        actual["timestamp"], q10, q90,
        color="#dc2626", alpha=0.18, label="10-90% interval",
    )

    plt.axvline(df["timestamp"].iloc[split - 1], color="gray", linestyle="--", linewidth=1)
    plt.title(
        f"Timestral — MAE {metrics['mae']:.3f}, "
        f"RMSE {metrics['rmse']:.3f}, MAPE {metrics['mape_percent']:.2f}%"
    )
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_forecast_csv(actual, median, q10, q90, path):
    """Save forecast vs actual comparison as CSV."""
    pd.DataFrame({
        "timestamp": actual["timestamp"],
        "actual": actual["value"].to_numpy(dtype=float),
        "forecast_median": median,
        "q10": q10,
        "q90": q90,
    }).to_csv(path, index=False)


def save_predictions(history_df, payload, output_dir="/output"):
    """Save prediction outputs: metrics JSON and forecast CSV (no actuals comparison)."""
    import pandas as pd

    median = np.asarray(payload["median"], dtype=float)
    q10 = np.asarray(payload["q10"], dtype=float)
    q90 = np.asarray(payload["q90"], dtype=float)
    horizon = len(median)

    last_ts = history_df["timestamp"].iloc[-1]
    freq = _infer_freq(history_df["timestamp"])
    future_ts = pd.date_range(start=last_ts, periods=horizon + 1, freq=freq)[1:]

    full_metrics = {
        "model": payload["model"],
        "horizon": horizon,
        "history_points": len(history_df),
    }

    with open(f"{output_dir}/metrics.json", "w") as f:
        json.dump(full_metrics, f, indent=2)
    print(json.dumps(full_metrics, indent=2))

    pd.DataFrame({
        "timestamp": future_ts,
        "forecast_median": median,
        "q10": q10,
        "q90": q90,
    }).to_csv(f"{output_dir}/forecast.csv", index=False)


def _infer_freq(timestamps):
    """Infer the frequency of a pandas DatetimeIndex."""
    inferred = pd.infer_freq(timestamps)
    return inferred if inferred else "D"


def save_outputs(payload, history, actual, metrics, output_dir="/output"):
    """Save metrics JSON, forecast vs actual CSV, and plot."""
    median = np.asarray(payload["median"], dtype=float)
    q10 = np.asarray(payload["q10"], dtype=float)
    q90 = np.asarray(payload["q90"], dtype=float)

    full_metrics = {
        "model": payload["model"],
        "horizon": len(actual),
        "history_points": len(history),
        **metrics,
    }

    with open(f"{output_dir}/metrics.json", "w") as f:
        json.dump(full_metrics, f, indent=2)

    print(json.dumps(full_metrics, indent=2))

    plot_forecast(
        pd.concat([history, actual], ignore_index=True),
        len(history),
        actual,
        median, q10, q90,
        metrics,
        f"{output_dir}/forecast.png",
    )
    save_forecast_csv(
        actual, median, q10, q90, f"{output_dir}/forecast_vs_actual.csv"
    )
