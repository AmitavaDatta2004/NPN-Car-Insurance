"""ClaimVision AI — FastAPI application entry point.

Phase 0: placeholder with health endpoint only.
Full implementation begins in Phase 12 (Backend foundation).

Run locally:
    uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
Or from inside the backend/ directory:
    uvicorn main:app --reload --host 127.0.0.1 --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="ClaimVision AI",
    description="AI-assisted vehicle insurance claim triage system.",
    version="0.1.0",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
    openapi_url="/api/v1/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/v1/health", tags=["system"])
def health() -> dict:
    """Return service health and readiness status.

    Used by the CI pipeline and the demo runbook pre-demo verification
    to confirm the backend started correctly.
    """
    return {
        "status": "ok",
        "service": "claimvision-api",
        "version": "0.1.0",
        "phase": "0 — skeleton",
        "models_loaded": False,
        "database_connected": False,
    }
