from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from joblib import load
import os
from typing import Literal, List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Configuration ===
PAIRS = [
    'AUDUSD', 'EURUSD', 'GBPUSD', 'NZDUSD', 'USDCAD',
    'USDCHF', 'USDHKD', 'USDNOK', 'USDSEK'
]
MODEL_PATH = "models"
FEATURE_COLS = [
    'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume',
    'sma_14', 'adx_14', 'stoch_k', 'rsi_14', 'cci_20', 'roc_10', 'atr_14',
    'bb_width', 'obv', 'mfi_14', 'macd_line', 'macd_hist', 'candle_body', 'candle_range'
]

trend_map = {-1: "Downtrend", 0: "Ranging", 1: "Uptrend"}
vol_map = {0: "Low", 1: "Medium", 2: "High"}

# === FastAPI App ===
app = FastAPI(title="Forex Prediction API (With Features Provided)")

# === Load Models at Startup ===
trend_models = {}
vol_models = {}

@app.on_event("startup")
def load_models():
    for pair in PAIRS:
        try:
            trend_models[pair] = load(os.path.join(MODEL_PATH, f"{pair}_model.joblib"))
            vol_models[pair] = load(os.path.join(MODEL_PATH, f"{pair}_vol_model.joblib"))
            logger.info(f"Loaded type for {pair}: {type(trend_models[pair])}")
            logger.info(f"Loaded type for {pair}: {type(vol_models[pair])}")


        except Exception as e:
            print(f"[ERROR] Could not load model for {pair}: {e}")

# === Request Schema ===
class FeatureInput(BaseModel):
    pair: Literal[
        'AUDUSD', 'EURUSD', 'GBPUSD', 'NZDUSD', 'USDCAD',
        'USDCHF', 'USDHKD', 'USDNOK', 'USDSEK'
    ]
    data: List[Dict]  # Already processed with all feature columns

@app.get("/health")
def health_check():
    return {"status": "ok"}

# === Prediction Endpoint ===
@app.post("/predict")
def predict(request: FeatureInput):
    pair = request.pair
    if pair not in trend_models or pair not in vol_models:
        raise HTTPException(status_code=404, detail=f"Models not found for {pair}")

    try:
        df = pd.DataFrame(request.data)

        # Check all required features are present
        missing_cols = [col for col in FEATURE_COLS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols}")

        df = df.dropna(subset=FEATURE_COLS)
        if df.empty:
            raise ValueError("No complete row with all features available.")

        latest = df.iloc[-1]
        X = latest[FEATURE_COLS].values.reshape(1, -1)

        trend_pred = trend_models[pair].predict(X)[0]
        vol_pred = vol_models[pair].predict(X)[0]

        return {
            "pair": pair,
            "trend_class": int(trend_pred),
            "trend_label": trend_map.get(trend_pred, "Unknown"),
            "vol_class": int(vol_pred),
            "vol_label": vol_map.get(vol_pred, "Unknown")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
