"""
Insurance AI Assistant - FastAPI Backend
=========================================
Main application entry point. Registers all routers and serves the frontend.
"""

import sys
import os

# Ensure project root is in Python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from routers import fraud, claims, data_quality, analytics

# -------------------------------------------------------------------
# App setup
# -------------------------------------------------------------------
app = FastAPI(
    title="Insurance AI Assistant",
    description=(
        "AI-powered insurance data assistant with fraud detection, "
        "claims processing, data quality analysis, and predictive analytics."
    ),
    version="1.0.0",
)

# CORS - allow frontend to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# Register routers
# -------------------------------------------------------------------
app.include_router(fraud.router)
app.include_router(claims.router)
app.include_router(data_quality.router)
app.include_router(analytics.router)

# -------------------------------------------------------------------
# Serve frontend static files
# -------------------------------------------------------------------
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
app.mount("/css", StaticFiles(directory=os.path.join(FRONTEND_DIR, "css")), name="css")
app.mount("/js", StaticFiles(directory=os.path.join(FRONTEND_DIR, "js")), name="js")


@app.get("/")
def serve_frontend():
    """Serve the main frontend page."""
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/health")
def health_check():
    """API health check endpoint."""
    return {"status": "ok", "service": "Insurance AI Assistant", "version": "1.0.0"}
