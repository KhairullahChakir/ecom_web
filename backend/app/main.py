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
        # Define feature groups structure (must match model training order)
        # Numerical: [Admin, AdminDur, Info, InfoDur, Prod, ProdDur, Bounce, Exit, PageVal, SpecDay]
        num_features = [
            request.administrative,
            request.administrative_duration,
            request.informational,
            request.informational_duration,
            request.product_related,
            request.product_related_duration,
            request.bounce_rates,
            request.exit_rates,
            request.page_values,
            request.special_day
        ]

        # Categorical: [Month, VisitorType, Weekend, OS, Browser, Region, TrafficType]
        # Note: Order matches load_and_preprocess in training notebook
        cat_features_raw = {
            'Month': request.month,
            'VisitorType': request.visitor_type,
            'Weekend': str(request.weekend), # boolean to string
            'OperatingSystems': str(request.operating_systems),
            'Browser': str(request.browser),
            'Region': str(request.region),
            'TrafficType': str(request.traffic_type)
        }
        
        # 1. Process Categorical (Label Encode)
        # Order: ['Month', 'VisitorType', 'Weekend', 'OperatingSystems', 'Browser', 'Region', 'TrafficType']
        # Note: Only Month, VisitorType, and Weekend match the label_encoders keys. 
        # The others (OS, Browser, Region, Traffic) are already ints and don't need encoding.
        
        cols_needing_encoding = ['Month', 'VisitorType', 'Weekend']
        cat_order = ['Month', 'VisitorType', 'Weekend', 'OperatingSystems', 'Browser', 'Region', 'TrafficType']
        x_cat = []
        
        for col in cat_order:
            val = cat_features_raw[col]
            
            if col in cols_needing_encoding:
                le = label_encoders[col]
                try:
                    encoded_val = le.transform([val])[0]
                except ValueError:
                    encoded_val = 0
            else:
                # For OS, Browser, etc., just use the integer value directly
                try:
                    encoded_val = int(request.operating_systems if col == 'OperatingSystems' else 
                                      request.browser if col == 'Browser' else
                                      request.region if col == 'Region' else
                                      request.traffic_type)
                except:
                    encoded_val = 0
                    
            x_cat.append(encoded_val)
            
        x_cat_np = np.array(x_cat, dtype=np.int64).reshape(1, -1)

        # 2. Process Numerical (Scale)
        # The scaler expects ALL 17 features (Num + Cat) because it was fitted on the full dataset.
        # We must construct the full vector, scale it, extract the numerical part, 
        # but keep the original INTEGER categorical values for the TabM embedding layers.
        
        x_num_np = np.array(num_features, dtype=np.float32).reshape(1, -1)
        
        # Combine [Num, Cat] to match the 17 features expected by StandardScaler
        # Note: We assume the order is Numerical (10) + Categorical (7) based on standard pipeline
        x_full = np.concatenate([x_num_np, x_cat_np], axis=1)
        
        # Transform everything
        x_full_scaled = scaler.transform(x_full).astype(np.float32)
        
        # Extract ONLY the numerical columns (first 10) for the model input
        # The model wants Scaled Numerical + Integer Categorical
        x_num_scaled = x_full_scaled[:, :10]
        
        # 3. Run ONNX inference
        start_inference = time.perf_counter()
        
        # Inputs matching the pruned model export
        ort_inputs = {
            'input_cat': x_cat_np,
            'input_num': x_num_scaled
        }
        
        onnx_result = session.run(None, ort_inputs)
        inference_latency_ms = (time.perf_counter() - start_inference) * 1000
        
        # Output is logits or probability depending on model; TabM usually outputs logits, we apply Sigmoid
        output_val = float(onnx_result[0][0])
        
        # Apply Sigmoid since the model output is likely raw logits (BCEWithLogitsLoss was used)
        probability = 1 / (1 + np.exp(-output_val))
        
        label = "YES" if probability >= 0.5 else "NO"
        
        total_latency_ms = (time.perf_counter() - start_total) * 1000
        
        return PredictResponse(
            label=label,
            probability=probability,
            inference_latency_ms=round(inference_latency_ms, 4),
            total_latency_ms=round(total_latency_ms, 2)
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "OP-ECOM Prediction API",
        "docs": "/docs",
        "health": "/health"
    }
