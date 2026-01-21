import numpy as np
import xgboost as xgb
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
from pathlib import Path

# --------------------------------------------------
# LOAD MODEL FROM MLFLOW MODEL REGISTRY ARTIFACT
# --------------------------------------------------
MODEL_PATH = Path(
    "mlruns/0/models/m-f71f30c69d4645cbbfc62fd53075df6d/artifacts/model.ubj"
)

if not MODEL_PATH.exists():
    raise RuntimeError(f"Model file not found at {MODEL_PATH}")

booster = xgb.Booster()
booster.load_model(str(MODEL_PATH))

# --------------------------------------------------
# FASTAPI APP
# --------------------------------------------------
app = FastAPI(
    title="MediFlow Sepsis Risk API",
    description="Predicts sepsis risk using XGBoost (MLflow Registry)",
    version="1.0"
)

class PatientFeatures(BaseModel):
    features: Dict[str, float]

# --------------------------------------------------
# PREDICTION ENDPOINT
# --------------------------------------------------
@app.post("/predict")
def predict_risk(data: PatientFeatures):
    X = np.array(list(data.features.values())).reshape(1, -1)
    dmatrix = xgb.DMatrix(X)
    prob = float(booster.predict(dmatrix)[0])

    return {
        "sepsis_risk_score": round(prob * 100, 2)
    }
