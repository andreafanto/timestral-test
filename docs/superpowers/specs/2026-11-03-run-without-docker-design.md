# Spec: Local run, output folders, and prediction plot (2026-11-03)

**Approved:** 2026-11-03 (approach A, then amended twice: timestamped folders, prediction plot)

**Intent / why:** Let someone run the full flow (API → predict → backtest) with only Python installed, documented in the existing README — and stop the two flows from overwriting each other's outputs (both currently write `metrics.json` to the same folder), with a graph for predictions to match the backtest graph.

## Goals

- New README section: venv setup → install → start API → health check → predict → backtest, all from repo root, bash-style like the rest of the README.
- Local runs read `data/series.csv` and write under `output/`.
- `OUTPUT` env var controls the output base dir (default `/output` keeps container behavior).
- Outputs land in timestamped per-flow folders: `prediction/<YYYY-MM-DD_HH-MM>/` and `backtest/<YYYY-MM-DD_HH-MM>/`, created automatically; scripts print the run folder (timestamped paths aren't guessable).
- `predict` writes a graph (`prediction.png`): history (last 200 points), forecast start line, red median line, q10–q90 band — no actuals, no MAE/RMSE/MAPE (there are no actuals in predict). Backtest's `forecast.png` stays as-is.
- Container flow unchanged apart from outputs being nested under the mounted `/output`.

## Non-goals / anti-goals

- No Windows/PowerShell instructions (README is bash-style throughout).
- No restructuring of the Docker sections; no separate `LOCAL.md`.
- No `--output` CLI flags — env var only, matching the existing `API`/`CSV` pattern.
- No run history cleanup/rotation; old timestamped folders accumulate.

## How (the decided parts)

1. `app/predict.py` + `app/backtest.py`: read `OUTPUT = os.getenv("OUTPUT", "/output")` (module top, same as `API`/`CSV`), build `os.path.join(OUTPUT, "prediction"|"backtest", timestamp)` with format `%Y-%m-%d_%H-%M`, `os.makedirs(..., exist_ok=True)`, pass as `output_dir`, print the folder.
2. `app/core.py`: new `plot_prediction(history_df, future_ts, median, q10, q90, out_path)`; `save_predictions` writes `prediction.png` alongside `forecast.csv` and `metrics.json`. `plot_forecast` untouched.
3. README: new "Run without Docker" section; `OUTPUT` row in the Configuration table; predict/backtest output bullets updated to the nested timestamped paths; note first run downloads model weights (~2 GB) and that `git` is required by `requirements.txt`.
4. TDD: failing tests first — predict writes to `prediction/<ts>/` under `OUTPUT` (dir auto-created), predict then backtest coexist without clobbering, `prediction.png` exists — then the fixes, then green.

## Accepted tradeoff

Timestamps are minute-level (`%Y-%m-%d_%H-%M`, as chosen): two runs in the same minute share a folder and overwrite. Acceptable for manual runs.

## Approaches considered

- **A (chosen):** Add section + `OUTPUT` env var. Least duplication, nothing breaks.
- **B (rejected):** Two equal paths throughout — doubles README size, paths drift apart.
- **C (rejected):** Separate `LOCAL.md` — split docs get forgotten.
