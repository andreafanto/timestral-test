import os
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from timestral import TimestralForecaster

MODEL_ID = os.getenv("MODEL_ID", "mistralai/Timestral-28M")
DEVICE = os.getenv("DEVICE", "cpu")

app = FastAPI(title="Timestral Forecast API")

print(f"Loading {MODEL_ID} on {DEVICE}...")
forecaster = TimestralForecaster.from_pretrained(MODEL_ID, device=DEVICE)
print("Model loaded.")


class ForecastRequest(BaseModel):
    values: list[float]
    prediction_length: int = 24


@app.get("/healthz")
def health():
    return {"status": "ok", "model": MODEL_ID, "device": DEVICE}


@app.post("/forecast")
def forecast(req: ForecastRequest):
    if len(req.values) < 32:
        raise HTTPException(400, "Fornisci almeno 32 osservazioni storiche.")

    if req.prediction_length < 1 or req.prediction_length > 1024:
        raise HTTPException(400, "prediction_length deve essere tra 1 e 1024.")

    # Timestral si aspetta: (serie, variabili, tempo)
    context = np.asarray(req.values, dtype=np.float32)[None, None, :]

    result = forecaster.forecast(
        context,
        prediction_length=req.prediction_length,
    )

    return {
        "model": MODEL_ID,
        "context_length": len(req.values),
        "prediction_length": req.prediction_length,
        "median": result.median[0, 0].tolist(),
        "q10": result.quantile(0.10)[0, 0].tolist(),
        "q90": result.quantile(0.90)[0, 0].tolist(),
    }