"""
OP-ECOM: Online Shoppers Purchase Prediction API
FastAPI backend for fast CPU inference
"""

import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

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
    latency_ms: float  # measured time


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    version: str


# Global model placeholder
model = None


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Predict purchase intent from session features.
    Returns label (YES/NO), probability, and inference latency.
    """
    start_time = time.perf_counter()
    
    # TODO: Replace with actual model inference
    # For now, return mock response
    probability = 0.65  # Mock probability
    label = "YES" if probability >= 0.5 else "NO"
    
    latency_ms = (time.perf_counter() - start_time) * 1000
    
    return PredictResponse(
        label=label,
        probability=round(probability, 4),
        latency_ms=round(latency_ms, 2)
    )


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "OP-ECOM Prediction API",
        "docs": "/docs",
        "health": "/health"
    }
