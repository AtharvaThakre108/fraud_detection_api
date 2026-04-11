from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import time
from predict import predict

app = FastAPI(
    title="Fraud Detection API",
    description="Detects fraudulent credit card transactions using LightGBM with SHAP explainability",
    version="1.0.0"
)

class TransactionRequest(BaseModel):
    Time: float = Field(..., description="Seconds elapsed since first transaction")
    Amount: float = Field(..., description="Transaction amount")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    risk_level: str
    top_contributing_features: dict
    processing_time_ms: float

@app.get("/")
def root():
    return {"status": "running", "model": "LightGBM Fraud Detector v1.0"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
def predict_fraud(transaction: TransactionRequest):
    try:
        start = time.time()
        result = predict(transaction.dict())
        elapsed = round((time.time() - start) * 1000, 2)
        return {**result, "processing_time_ms": elapsed}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))