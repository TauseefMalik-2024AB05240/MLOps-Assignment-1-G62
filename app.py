from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib

app = FastAPI(
    title="Heart Disease Prediction API",
    version="1.0.0",
    description="Prediction API trained on UCI Heart Disease dataset"
)

# =========================
# Load trained pipeline
# =========================
MODEL_PATH = "best_random_forest_pipeline.joblib"
model = joblib.load(MODEL_PATH)

# =========================
# Input schema (MATCHES DATASET)
# =========================
class Patient(BaseModel):
    age: float
    sex: str                # e.g. "male", "female"
    cp: str                 # chest pain type
    trestbps: float
    chol: float
    fbs: int                # 0 or 1
    restecg: str
    thalach: float
    exang: int              # 0 or 1
    oldpeak: float
    slope: str
    ca: float
    thal: str


class PredictionResult(BaseModel):
    prediction: int         # 0 or 1
    probability: float      # confidence


# =========================
# Root
# =========================
@app.get("/")
def root():
    return {
        "message": "Heart Disease Prediction API is running",
        "docs": "/docs"
    }


# =========================
# Health check
# =========================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }


# =========================
# Model metadata
# =========================
@app.get("/model/info")
def model_info():
    return {
        "model_type": type(model).__name__,
        "uses_pipeline": True,
        "supports_proba": hasattr(model, "predict_proba")
    }


# =========================
# Expected features
# =========================
@app.get("/model/features")
def model_features():
    return {
        "features": list(Patient.model_fields.keys())
    }


# =========================
# Batch prediction
# =========================
@app.post("/predict", response_model=List[PredictionResult])
def predict(patients: List[Patient]):
    df = pd.DataFrame([p.model_dump() for p in patients])

    probabilities = model.predict_proba(df)[:, 1]
    predictions = model.predict(df)

    return [
        {
            "prediction": int(pred),
            "probability": float(prob)
        }
        for pred, prob in zip(predictions, probabilities)
    ]


# =========================
# Single prediction (UI-friendly)
# =========================
@app.post("/predict/single", response_model=PredictionResult)
def predict_single(patient: Patient):
    df = pd.DataFrame([patient.model_dump()])

    prob = model.predict_proba(df)[0][1]
    pred = model.predict(df)[0]

    return {
        "prediction": int(pred),
        "probability": float(prob)
    }
