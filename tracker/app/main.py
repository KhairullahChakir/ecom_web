"""
OP-ECOM Analytics Tracker - FastAPI Main Application
Self-hosted analytics for collecting real user behavior data
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .database import engine, Base
from .tracker_router import router as tracker_router
from .admin_router import router as admin_router

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="OP-ECOM Analytics Tracker",
    description="Self-hosted analytics API for collecting user behavior data",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(tracker_router)
app.include_router(admin_router)

@app.get("/")
async def root():
    return {
        "service": "OP-ECOM Analytics Tracker",
        "version": "1.0.0",
        "endpoints": {
            "tracker": "/tracker/session/start, /tracker/pageview, /tracker/event, /tracker/purchase",
            "admin": "/admin/sessions, /admin/export, /admin/stats"
        }
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "tracker"}
