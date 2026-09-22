"""ClaimVision AI — FastAPI application entry point.

Task ID : BE-001
Phase   : 13 (Backend Foundation & Assessment APIs)

Run locally:
    uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
"""

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.api.routes import (
    assessment_router,
    claims_router,
    dashboard_router,
    health_router,
    images_router,
    reviews_router,
)
from backend.app.services.upload import ensure_uploads_dir

app = FastAPI(
    title="ClaimVision AI API",
    description="Explainable vehicle-damage insurance claim assessment and triage API.",
    version="1.0.0",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
    openapi_url="/api/v1/openapi.json",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure uploads directory exists and mount static files
uploads_path = ensure_uploads_dir()
app.mount("/uploads", StaticFiles(directory=str(uploads_path)), name="uploads")

# Mount API routers under /api/v1
app.include_router(health_router, prefix="/api/v1")
app.include_router(claims_router, prefix="/api/v1")
app.include_router(images_router, prefix="/api/v1")
app.include_router(assessment_router, prefix="/api/v1")
app.include_router(reviews_router, prefix="/api/v1")
app.include_router(dashboard_router, prefix="/api/v1")
