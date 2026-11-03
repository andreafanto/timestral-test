# Local Run, Output Folders & Prediction Plot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make predict/backtest runnable without Docker via an `OUTPUT` env var, split their outputs into auto-created timestamped folders (`prediction/`, `backtest/`), add a prediction graph, and document the local workflow in the README.

**Architecture:** Scripts gain a module-top `OUTPUT` constant (same pattern as `API`/`CSV`), build a timestamped run folder under it, and pass that folder to the existing `save_predictions`/`save_outputs` writers. `core.py` gains one new plotting function; backtest plotting is untouched. README gains one section plus updated output paths.

**Tech Stack:** Python 3.12+, pandas, matplotlib, pytest (`.venv`), FastAPI/uvicorn for the API.

**Spec:** `docs/superpowers/specs/2026-11-03-run-without-docker-design.md`

## Global Constraints

- `OUTPUT` default must be exactly `/output` (container parity); `CSV` default stays `/data/series.csv`.
- Env read at module top: `os.getenv("OUTPUT", "/output")` — no CLI flags.
- Folder names exactly `prediction` and `backtest`; timestamp format exactly `%Y-%m-%d_%H-%M`.
- Run folders are created by the scripts (`os.makedirs(..., exist_ok=True)`); scripts print the folder path.
- README stays bash-only; Docker quick-start structure unchanged (output bullets updated to nested paths).
- Tests run with `.venv/bin/pytest tests/ -v`; test suite currently 4 tests, must stay green.
- Commits deferred — user has not requested commits (prior fixes also uncommitted).

## Review Focus

1. Fresh clone with no `output/` dir → scripts must still work (auto-mkdir): pinned by Task 1 Step 1 test, which points `OUTPUT` at a non-existent directory.
2. Container parity — `OUTPUT` unset must still default to `/output` in both scripts: pinned by `test_predict_output_default_is_container_path` (Task 1) and `test_backtest_output_default_is_container_path` (Task 2).
3. Predict and backtest must not clobber each other (the `metrics.json` collision): pinned by Task 2's test, which runs both flows against one `OUTPUT` and asserts both folders hold their own `metrics.json`.
4. Prediction graph must not claim metrics/actuals it doesn't have: pinned by Task 3's test asserting `prediction.png` exists; `plot_prediction`'s signature (no `actual`, no `metrics` params) makes the wrong title structurally impossible.
5. README commands must actually work as written: pinned by Task 4's verification step, which runs each documented command verbatim.

---

### Task 1: `OUTPUT` env var + timestamped `prediction/` folder (predict)

**Files:**
- Modify: `app/predict.py`
- Test: `tests/test_predict.py` (new); `tests/conftest.py` (add fixtures)

**Interfaces:**
- Consumes: `save_predictions(history_df, payload, output_dir="/output")` (existing, unchanged).
- Produces: module global `predict.OUTPUT: str` (default `/output`); `predict.main(horizon: int)` writes all outputs to `{OUTPUT}/prediction/<YYYY-MM-DD_HH-MM>/` and prints that path.

- [ ] **Step 1: Write the failing tests**

Add fixtures to `tests/conftest.py` (needs `import pandas as pd`, `import pytest`):

```python
@pytest.fixture
def tiny_csv(tmp_path):
    ts = pd.date_range("2020-01-01", periods=40, freq="h")
    path = tmp_path / "series.csv"
    pd.DataFrame({"timestamp": ts, "value": range(40)}).to_csv(path, index=False)
    return str(path)

@pytest.fixture
def fake_forecast():
    def _forecast(api, values, horizon):
        return {"model": "test-model", "median": list(range(horizon)),
                "q10": list(range(horizon)), "q90": list(range(horizon))}
    return _forecast
```

New `tests/test_predict.py`:

```python
import importlib
import os

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_predict.py -v`
Expected: FAIL — first test errors with `AttributeError: module 'predict' has no attribute 'OUTPUT'` (feature missing).

- [ ] **Step 3: Implement in `app/predict.py`**

Add after the `CSV = ...` line: `OUTPUT = os.getenv("OUTPUT", "/output")` (plus `from datetime import datetime`). In `main`, after `forecast(...)`: build `run_dir = os.path.join(OUTPUT, "prediction", datetime.now().strftime("%Y-%m-%d_%H-%M"))`, `os.makedirs(run_dir, exist_ok=True)`, call `save_predictions(df, payload, output_dir=run_dir)`, then `print(f"Outputs saved to: {run_dir}")`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_predict.py -v` → 2 passed.
Run: `.venv/bin/pytest tests/ -v` → all green (4 old + 2 new).

---

### Task 2: `OUTPUT` env var + timestamped `backtest/` folder (backtest)

**Files:**
- Modify: `app/backtest.py`
- Test: `tests/test_backtest.py` (new)

**Interfaces:**
- Consumes: `save_outputs(payload, history, actual, metrics, output_dir="/output")` (existing, unchanged).
- Produces: module global `backtest.OUTPUT: str` (default `/output`); `backtest.main(horizon: int)` writes to `{OUTPUT}/backtest/<YYYY-MM-DD_HH-MM>/` and prints that path.

- [ ] **Step 1: Write the failing test**

New `tests/test_backtest.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_backtest.py -v`
Expected: FAIL — `AttributeError: module 'backtest' has no attribute 'OUTPUT'`.

- [ ] **Step 3: Implement in `app/backtest.py`**

Same pattern as Task 1: `OUTPUT = os.getenv("OUTPUT", "/output")` after `CSV`, `from datetime import datetime`; build `run_dir = os.path.join(OUTPUT, "backtest", datetime.now().strftime("%Y-%m-%d_%H-%M"))`, `os.makedirs(run_dir, exist_ok=True)`, `save_outputs(..., output_dir=run_dir)`, print the folder.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/ -v` → all green (4 old + 4 new).

---

### Task 3: Prediction plot (`prediction.png`)

**Files:**
- Modify: `app/core.py` (new `plot_prediction`; `save_predictions` calls it)
- Test: `tests/test_predict.py` (add one test)

**Interfaces:**
- Consumes: `save_predictions` already computes `future_ts` (line building `pd.date_range(...)`) — pass it through.
- Produces: `plot_prediction(history_df, future_ts, median, q10, q90, out_path) -> None` in `app/core.py`; `save_predictions` writes `prediction.png` into `output_dir`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_predict.py`:

```python
def test_predict_writes_prediction_plot(tmp_path, monkeypatch, tiny_csv, fake_forecast):
    fresh = tmp_path / "fresh"
    monkeypatch.setattr(predict, "CSV", tiny_csv)
    monkeypatch.setattr(predict, "OUTPUT", str(fresh))
    monkeypatch.setattr(predict, "forecast", fake_forecast)
    predict.main(48)
    runs = list((fresh / "prediction").glob("????-??-??_??-??"))
    png = runs[0] / "prediction.png"
    assert png.exists() and png.stat().st_size > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_predict.py::test_predict_writes_prediction_plot -v`
Expected: FAIL — `assert (… / "prediction.png").exists()` is False.

- [ ] **Step 3: Implement `plot_prediction` in `app/core.py`**

Signature: `def plot_prediction(history_df, future_ts, median, q10, q90, out_path):`. Mirror `plot_forecast`'s styling (figsize 14×7, history color `#2563eb`, median `#dc2626` linewidth 2, band alpha 0.18, gray dashed `axvline` at `history_df["timestamp"].iloc[-1]`, grid alpha 0.25, dpi 160) but: history is `history_df.iloc[-200:]`, x-axis for median/band is `future_ts`, title is `f"Timestral — {len(median)}-step forecast"`, no Actual line, no metrics in the title, `plt.close()` at the end. Call it from `save_predictions` right after the CSV write: `plot_prediction(history_df, future_ts, median, q10, q90, f"{output_dir}/prediction.png")`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/ -v` → all green (4 old + 5 new).

---

### Task 4: README — "Run without Docker" section + updated output paths

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: the behaviors of Tasks 1–3 (folder names, defaults, printed paths).
- Produces: documented local workflow; `OUTPUT` row in Configuration table.

- [ ] **Step 1: Write the section and edits**

Three edits to `README.md`:
1. Quick start intro: after "You need to have Docker installed." add "(or see [Run without Docker](#run-without-docker))".
2. New `## Run without Docker` section inserted between `## Usage` and `## Configuration`, with: prerequisites (Python 3.12+, `git` for the `timestral @ git+https` requirement, ~2 GB for model weights downloaded on first run); `mkdir -p data output`; CSV at `data/series.csv`; `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`; `uvicorn app.main:app --host 0.0.0.0 --port 8000` (wait for `Model loaded.`); `curl http://localhost:8000/health`; then `CSV=data/series.csv OUTPUT=output python app/predict.py --horizon 48` and `CSV=data/series.csv OUTPUT=output python app/backtest.py --horizon 48`, each noting it prints its output folder; tests via `pip install -r requirements-dev.txt && pytest`.
3. Update both Docker usage bullets to the nested timestamped paths and add `prediction.png` to predict's outputs: predict → `output/prediction/<YYYY-MM-DD_HH-MM>/` (`forecast.csv`, `metrics.json`, `prediction.png`); backtest → `output/backtest/<YYYY-MM-DD_HH-MM>/` (`forecast_vs_actual.csv`, `forecast.png`, `metrics.json`). Configuration table gains `| OUTPUT | /output | Output base directory (use `output` locally) |`.

- [ ] **Step 2: Run every documented local command verbatim**

Precondition: API already running (health check green); if `output/` is root-owned from container runs, run `docker compose exec timestral chown -R 1000:1000 /output` first.
Run each command exactly as written in the new section; expected: both flows exit 0, print their `Outputs saved to:` path, and the named files exist under `output/prediction/<ts>/` and `output/backtest/<ts>/`.

- [ ] **Step 3: Full suite**

Run: `.venv/bin/pytest tests/ -v` → 9 passed.

---

## Self-review (writing-plans)

1. **Spec coverage:** OUTPUT env (T1/T2) ✓, timestamped folders + auto-mkdir + printed path (T1/T2) ✓, plot + exact styling/signature (T3) ✓, README section + config row + nested-path bullets + git/model notes (T4) ✓, accepted minute-level tradeoff (spec) ✓.
2. **Step scan:** every step = one test, one run, one implementation, one verify; README step lists exact edits; no TBD lines.
3. **Type consistency:** `OUTPUT: str` in both scripts; `plot_prediction(history_df, future_ts, median, q10, q90, out_path)` matches its call site and test; folder names/timestamp format identical across tasks and glob patterns.
4. **Review Focus:** all five lines pinned to a task/step above.
5. **Proportion:** code blocks limited to tests and the one new function's styling spec; bodies otherwise left to the implementer.
