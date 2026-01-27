"""
OP-ECOM: Online Shoppers Purchase Prediction API
FastAPI backend for fast CPU inference using ONNX
"""

import time
import os
import numpy as np
import joblib
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any

app = FastAPI(
    title="OP-ECOM Prediction API",
    description="Fast CPU inference for online shoppers purchase prediction",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths to model artifacts
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
ONNX_MODEL_PATH = os.path.join(MODELS_DIR, "tabm_best.onnx")
SCALER_PATH = os.path.join(MODELS_DIR, "onnx_scaler.joblib")
ENCODERS_PATH = os.path.join(MODELS_DIR, "onnx_label_encoders.joblib")

# Global variables for model artifacts
session = None
scaler = None
label_encoders = None

@app.on_event("startup")
async def load_model_artifacts():
    """Load ONNX session and preprocessing artifacts on startup"""
    global session, scaler, label_encoders
    try:
        # Check if files exist
        for path in [ONNX_MODEL_PATH, SCALER_PATH, ENCODERS_PATH]:
            if not os.path.exists(path):
                print(f"Warning: Artifact not found at {path}")
        
        # Load ONNX session
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        session = ort.InferenceSession(ONNX_MODEL_PATH, sess_options, providers=['CPUExecutionProvider'])
        
        # Load scaler and encoders
        scaler = joblib.load(SCALER_PATH)
        label_encoders = joblib.load(ENCODERS_PATH)
        
        print(f"Successfully loaded model from {ONNX_MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model artifacts: {e}")

class PredictRequest(BaseModel):
    """Input features for prediction"""
    administrative: int = 0
    administrative_duration: float = 0.0
    informational: int = 0
    informational_duration: float = 0.0
    product_related: int = 0
    product_related_duration: float = 0.0
    bounce_rates: float = 0.0
    exit_rates: float = 0.0
    page_values: float = 0.0
    special_day: float = 0.0
    month: str = "Feb"
    operating_systems: int = 1
    browser: int = 1
    region: int = 1
    traffic_type: int = 1
    visitor_type: str = "Returning_Visitor"
    weekend: bool = False

class PredictResponse(BaseModel):
    """Prediction result"""
    label: str  # YES or NO
    probability: float  # 0-1
    inference_latency_ms: float  # pure model time
    total_latency_ms: float  # preprocess + model time

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    version: str

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=session is not None,
        version="1.0.0"
    )

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Predict purchase intent from session features.
    Returns label (YES/NO), probability, and inference latency.
    """
    if session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_total = time.perf_counter()
    
    try:
        # 1. Map request to feature vector
        raw_features = [
            request.administrative,
            request.administrative_duration,
            request.informational,
            request.informational_duration,
            request.product_related,
            request.product_related_duration,
            request.bounce_rates,
            request.exit_rates,
            request.page_values,
            request.special_day,
            request.month,
            request.operating_systems,
            request.browser,
            request.region,
            request.traffic_type,
            request.visitor_type,
            request.weekend
        ]
        
        # 2. Encode categorical features
        processed_features = list(raw_features)
        cat_indices = {10: "Month", 15: "VisitorType", 16: "Weekend"}
        for idx, col_name in cat_indices.items():
            val = str(processed_features[idx])
            le = label_encoders[col_name]
            try:
                processed_features[idx] = le.transform([val])[0]
            except ValueError:
                processed_features[idx] = 0
        
        # 3. Scale features
        features_array = np.array(processed_features).reshape(1, -1)
        scaled_features = scaler.transform(features_array).astype(np.float32)
        
        # 4. Run ONNX inference (MEASURE THIS SEPARATELY)
        start_inference = time.perf_counter()
        input_name = session.get_inputs()[0].name
        onnx_result = session.run(None, {input_name: scaled_features})
        inference_latency_ms = (time.perf_counter() - start_inference) * 1000
        
        probability = float(onnx_result[0][0])
        label = "YES" if probability >= 0.5 else "NO"
        
        total_latency_ms = (time.perf_counter() - start_total) * 1000
        
        return PredictResponse(
            label=label,
            probability=round(probability, 4),
            inference_latency_ms=round(inference_latency_ms, 2),
            total_latency_ms=round(total_latency_ms, 2)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "OP-ECOM Prediction API",
        "docs": "/docs",
        "health": "/health"
    }
