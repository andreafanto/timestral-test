# Timestral Test

Example integration of the [Timestral](https://github.com/mistralai/timestral) time series forecasting model.

## Quick start

You need to have Docker installed (or see [Run without Docker](#run-without-docker) to run everything with local Python instead).

### 1. Create required folders

```bash
mkdir -p {data,models,output}
```

### 2. Prepare your data

Place a CSV file at `data/series.csv` with two columns: `timestamp` and `value`.

### 3. Build and run

```bash
docker compose up -d --build --force-recreate
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

Outputs are saved to a timestamped `output/prediction/<YYYY-MM-DD_HH-MM>/` folder (the command prints the exact path):
- `forecast.csv` — predicted values with median, q10, and q90 columns
- `metrics.json` — model and horizon information
- `prediction.png` — history and forecast plot

### Backtest forecast accuracy

Evaluate forecast accuracy by comparing predictions against known actual values:

```bash
docker compose exec timestral python app/backtest.py --horizon 48
```

Outputs are saved to a timestamped `output/backtest/<YYYY-MM-DD_HH-MM>/` folder (the command prints the exact path):
- `forecast_vs_actual.csv` — predicted vs actual values
- `forecast.png` — visual comparison plot
- `metrics.json` — MAE, RMSE, and MAPE metrics

## Run without Docker

Prefer local Python? Everything also runs without Docker. You need Python 3.12+, `git` (the `timestral` dependency installs from GitHub), and about 2 GB of disk for the model weights, which are downloaded automatically on the first run. All commands run from the repository root.

### 1. Create required folders

```bash
mkdir -p data output
```

### 2. Prepare your data

Place a CSV file at `data/series.csv` with two columns: `timestamp` and `value`.

### 3. Set up the environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Start the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Wait until the log prints `Model loaded.`, then check from another terminal:

```bash
curl http://localhost:8000/health
```

### 5. Predict future values

With the API running and the virtualenv activated:

```bash
CSV=data/series.csv OUTPUT=output python app/predict.py --horizon 48
```

The command prints its output folder, e.g. `output/prediction/2026-11-03_18-00/`.

### 6. Backtest forecast accuracy

```bash
CSV=data/series.csv OUTPUT=output python app/backtest.py --horizon 48
```

The command prints its output folder, e.g. `output/backtest/2026-11-03_18-00/`.

### 7. Run the tests (optional)

```bash
pip install -r requirements-dev.txt
pytest
```

## Configuration

| Environment variable | Default | Description |
|----|-------|------|
| `MODEL_ID` | `mistralai/Timestral-28M` | Hugging Face model ID |
| `DEVICE` | `cpu` | Device to run the model on |
| `API` | `http://localhost:8000` | API base URL (for scripts) |
| `CSV` | `/data/series.csv` | Path to input CSV (inside container) |
| `OUTPUT` | `/output` | Output base directory (use `output` locally) |