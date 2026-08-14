"""
MediSynth.AI — FastAPI Application Entry Point
Privacy-Preserving Synthetic Healthcare Data Generation Platform
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.api.routes import router
from backend.config import FRONTEND_DIR
from backend.utils.logging_config import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown events."""
    logger = setup_logging()
    logger.info("MediSynth.AI starting up...")
    yield
    logger.info("MediSynth.AI shutting down...")


app = FastAPI(
    title="MediSynth.AI",
    description=(
        "Privacy-Preserving Synthetic Healthcare Data Generation Platform. "
        "Generate HIPAA/GDPR-compliant synthetic data with differential privacy "
        "guarantees, federated learning, and comprehensive privacy validation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(router)

# Serve frontend static files
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
async def serve_frontend():
    """Serve the main dashboard."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {
        "message": "MediSynth.AI API",
        "docs": "/docs",
        "version": "1.0.0",
    }
