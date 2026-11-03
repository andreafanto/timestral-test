# Timestral Test

Example integration of the [Timestral](https://github.com/mistralai/timestral) time series forecasting model.

## Quick start

You need to have Docker installed.

### 1. Create required folders

```bash
mkdir -p {data,models,output}
```

### 2. Prepare your data

Place a CSV file at `data/series.csv` with two columns: `timestamp` and `value`.

### 3. Build and run

```bash
docker compose up -d --build
```

### 4. Verify the model is loaded

```bash
curl http://localhost:8000/health
```

## Usage

### Predict future values

Forecast the next `N` future data points using all available history:

```bash
docker compose exec timestral python app/predict.py --horizon 48
```

Outputs are saved to the `output/` directory:
- `forecast.csv` — predicted values with median, q10, and q90 columns
- `metrics.json` — model and horizon information

### Backtest forecast accuracy

Evaluate forecast accuracy by comparing predictions against known actual values:

```bash
docker compose exec timestral python app/backtest.py --horizon 48
```

Outputs are saved to the `output/` directory:
- `forecast_vs_actual.csv` — predicted vs actual values
- `forecast.png` — visual comparison plot
- `metrics.json` — MAE, RMSE, and MAPE metrics

## Configuration

| Environment variable | Default | Description |
|----|-------|------|
| `MODEL_ID` | `mistralai/Timestral-28M` | Hugging Face model ID |
| `DEVICE` | `cpu` | Device to run the model on |
| `API` | `http://localhost:8000` | API base URL (for scripts) |
| `CSV` | `/data/series.csv` | Path to input CSV (inside container) |