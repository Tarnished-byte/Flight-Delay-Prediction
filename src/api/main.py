from __future__ import annotations
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from src.features.build_features import (
    engineer_deterministic_features,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)

MODEL_PATH = "models/xgb_flight_delay.joblib"

app = FastAPI(title="Flight Delay Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # fine for local/demo use; restrict in real production
    allow_methods=["*"],
    allow_headers=["*"],
)

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
preprocessor = artifact["preprocessor"]
rate_maps = artifact["rate_maps"]          # dict: {"carrier_delay_rate": Series, ...}
global_mean = artifact["global_mean"]


class FlightRequest(BaseModel):
    FL_DATE: str            # "2024-06-15"
    AIRLINE: str
    ORIGIN: str
    DEST: str
    CRS_DEP_TIME: int       # e.g. 1430 for 2:30pm
    CRS_ARR_TIME: int
    CRS_ELAPSED_TIME: float
    DISTANCE: float


def apply_saved_rate_maps(df: pd.DataFrame) -> pd.DataFrame:
    """Mirrors fit_apply_target_encodings from training, but using
    already-fitted maps instead of recomputing them."""
    df = df.copy()
    df["route"] = df["ORIGIN"] + "_" + df["DEST"]
    col_map = {
        "AIRLINE": "carrier_delay_rate",
        "ORIGIN": "origin_delay_rate",
        "DEST": "dest_delay_rate",
        "route": "route_delay_rate",
    }
    for raw_col, new_col in col_map.items():
        df[new_col] = df[raw_col].map(rate_maps[new_col]).fillna(global_mean)
    return df


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(flight: FlightRequest):
    try:
        df = pd.DataFrame([flight.dict()])
        df = engineer_deterministic_features(df)   # sched_hour, month, cyclical, distance_bucket, etc.
        df = apply_saved_rate_maps(df)              # carrier/origin/dest/route delay rates

        X = preprocessor.transform(df[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
        prob = float(model.predict_proba(X)[0][1])

        return {
            "delay_probability": round(prob, 4),
            "predicted_delayed": prob > 0.5,
        }
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing or invalid field: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))