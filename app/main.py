"""
Trading Opportunity Rating (TOR) API - Student D
Multiclass classification using LightGBM to predict trading opportunities
for cryptocurrency assets 7 days into the future.
"""

import os
import json
from datetime import datetime, timedelta, timezone

import joblib
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Trading Opportunity Rating API",
    description="Predicts the Trading Opportunity Rating (TOR) of cryptocurrencies 7 days into the future.",
    version="1.0.0",
)

SUPPORTED_TOKENS = ["BTC", "ETH", "XRP", "SOL", "DOGE", "XMR", "XLM", "TRX"]

CLASS_LABELS = ["Strong Sell", "Sell", "Hold", "Buy", "Strong Buy"]

STUDENT_A_URL = os.getenv("STUDENT_A_API_URL", "")
STUDENT_B_URL = os.getenv("STUDENT_B_API_URL", "")
STUDENT_C_URL = os.getenv("STUDENT_C_API_URL", "")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "tor_champion.joblib")
METADATA_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "tor_model_metadata.json")

model_bundle = None
model_metadata = None


def load_model():
    global model_bundle, model_metadata
    if os.path.exists(MODEL_PATH):
        model_bundle = joblib.load(MODEL_PATH)
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r") as f:
            model_metadata = json.load(f)


@app.on_event("startup")
def startup():
    load_model()


class FeaturesRequest(BaseModel):
    token: str
    forecast_horizon: int = Field(default=7, ge=1)


class PredictRequest(BaseModel):
    token: str
    forecast_horizon: int = Field(default=7, ge=1)


class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    forecast_horizon: int
    as_of_date: str
    prediction_date: str
    probabilities: dict[str, float] = {}
    warnings: list[str] = []


class FeaturesResponse(BaseModel):
    token: str
    features: dict


def fetch_ohlcv(token: str, days: int = 90) -> pd.DataFrame:
    """Fetch daily OHLCV data from Kraken public API."""
    pair_map = {
        "BTC": "XBTUSD", "ETH": "ETHUSD", "XRP": "XRPUSD",
        "SOL": "SOLUSD", "DOGE": "DOGEUSD", "XMR": "XMRUSD",
        "XLM": "XLMUSD", "TRX": "TRXUSD",
    }
    pair = pair_map.get(token)
    if not pair:
        return pd.DataFrame()

    since = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())
    try:
        resp = requests.get(
            "https://api.kraken.com/0/public/OHLC",
            params={"pair": pair, "interval": 1440, "since": since},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("error"):
            return pd.DataFrame()
        result = data["result"]
        pair_key = [k for k in result if k != "last"][0]
        candles = result[pair_key]
        df = pd.DataFrame(candles, columns=[
            "timestamp", "open", "high", "low", "close", "vwap", "volume", "count"
        ])
        df = df.drop(columns=["vwap", "count"])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s")
        df = df.set_index("timestamp").resample("D").last().dropna()
        today_midnight = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
        df = df[df.index < today_midnight]
        return df
    except (requests.RequestException, KeyError, ValueError):
        return pd.DataFrame()


def compute_sma20_volume(volume: pd.Series) -> pd.Series:
    return volume.rolling(window=20, min_periods=20).mean()


def compute_relative_volume(volume: pd.Series, sma20_vol: pd.Series) -> pd.Series:
    return volume / sma20_vol


def fetch_features_from_api(url: str, token: str) -> dict:
    """Call another student's /features endpoint with timeout and graceful fallback."""
    if not url:
        return {}
    try:
        resp = requests.post(
            f"{url}/features",
            json={"token": token, "forecast_horizon": 7},
            timeout=90,
        )
        if resp.status_code == 200:
            return resp.json().get("features", {})
    except Exception:
        pass
    return {}


@app.get("/")
def root():
    return {
        "project": "Trading Opportunity Rating API",
        "description": "Predicts the Trading Opportunity Rating (TOR) of cryptocurrencies 7 days into the future.",
        "algorithm": "LightGBM Classifier",
        "version": "1.0.0",
        "github": "https://github.com/KunalMistry000/36120-26SP-Group_2-Student_d-API",
        "endpoints": ["/", "/health", "/features", "/predict", "/model-metadata"],
    }


@app.get("/health")
def health():
    status = "healthy" if model_bundle is not None else "degraded"
    return {
        "status": status,
        "model_loaded": model_bundle is not None,
        "student_a_url": STUDENT_A_URL[:30] + "..." if STUDENT_A_URL else "",
        "student_b_url": STUDENT_B_URL[:30] + "..." if STUDENT_B_URL else "",
        "student_c_url": STUDENT_C_URL[:30] + "..." if STUDENT_C_URL else "",
    }


@app.post("/features")
def features(req: FeaturesRequest):
    if req.token not in SUPPORTED_TOKENS:
        raise HTTPException(
            status_code=400,
            detail={"error": "Unsupported cryptocurrency", "supported_tokens": SUPPORTED_TOKENS},
        )

    ohlcv = fetch_ohlcv(req.token, days=90)
    if ohlcv.empty:
        raise HTTPException(status_code=503, detail="Unable to fetch market data")

    sma20vol = compute_sma20_volume(ohlcv["volume"])
    rel_vol = compute_relative_volume(ohlcv["volume"], sma20vol)
    result = {
        "SMA20Volume": None if pd.isna(sma20vol.iloc[-1]) else float(sma20vol.iloc[-1]),
        "RelativeVolume": None if pd.isna(rel_vol.iloc[-1]) else float(rel_vol.iloc[-1]),
    }

    return FeaturesResponse(token=req.token, features=result)


@app.post("/predict")
def predict(req: PredictRequest):
    if req.token not in SUPPORTED_TOKENS:
        raise HTTPException(
            status_code=400,
            detail={"error": "Unsupported cryptocurrency", "supported_tokens": SUPPORTED_TOKENS},
        )

    if model_bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    a_feats = fetch_features_from_api(STUDENT_A_URL, req.token)
    b_feats = fetch_features_from_api(STUDENT_B_URL, req.token)
    c_feats = fetch_features_from_api(STUDENT_C_URL, req.token)

    return20 = a_feats.get("Return20")
    ma5 = b_feats.get("MA5")
    ma20 = b_feats.get("MA20")
    volatility20 = c_feats.get("Volatility20")

    ohlcv = fetch_ohlcv(req.token, days=90)

    if ohlcv.empty:
        relative_volume = None
        as_of_date = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        sma20vol = compute_sma20_volume(ohlcv["volume"])
        rel_vol = compute_relative_volume(ohlcv["volume"], sma20vol)
        relative_volume = None if pd.isna(rel_vol.iloc[-1]) else float(rel_vol.iloc[-1])
        as_of_date = ohlcv.index[-1].strftime("%Y-%m-%d")

    feature_names = model_bundle.get("feature_names", [
        "Return20", "RelativeVolume", "Volatility20", "MA5", "MA20",
    ])
    feature_values = {
        "Return20": return20,
        "RelativeVolume": relative_volume,
        "Volatility20": volatility20,
        "MA5": ma5,
        "MA20": ma20,
    }

    missing = [k for k in feature_names if feature_values.get(k) is None]
    warnings = [f"Missing feature: {m}" for m in missing] if missing else []

    X = pd.DataFrame([{fn: feature_values.get(fn) for fn in feature_names}]).astype(float)

    model = model_bundle["model"]
    proba = model.predict(X)
    if proba.ndim == 1:
        proba = proba.reshape(1, -1)
    proba = proba[0]
    pred_idx = int(np.argmax(proba))
    confidence = float(proba[pred_idx])

    label_map = model_bundle.get("label_map", CLASS_LABELS)
    prediction_label = label_map[pred_idx] if pred_idx < len(label_map) else "Hold"

    probabilities = {label_map[i]: round(float(proba[i]), 4) for i in range(min(len(proba), len(label_map)))}

    prediction_date = (pd.Timestamp(as_of_date) + timedelta(days=req.forecast_horizon)).strftime("%Y-%m-%d")

    return PredictResponse(
        prediction=prediction_label,
        confidence=round(confidence, 4),
        forecast_horizon=req.forecast_horizon,
        as_of_date=as_of_date,
        prediction_date=prediction_date,
        probabilities=probabilities,
        warnings=warnings,
    )


@app.get("/model-metadata")
def get_model_metadata():
    if model_metadata is not None:
        return model_metadata

    return {
        "target": "Trading Opportunity Rating",
        "algorithm": "LightGBM Classifier",
        "forecast_horizon": 7,
        "features": ["Return20", "RelativeVolume", "Volatility20", "MA5", "MA20"],
        "feature_importance": {},
        "classification_labels": CLASS_LABELS,
    }
