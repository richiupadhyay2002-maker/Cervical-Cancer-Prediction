#!/usr/bin/env python
"""
Entry point for the Cervical Cancer Prediction Service.

Usage:
    python run.py

This starts the FastAPI server with Uvicorn. The model is loaded from
the MLflow Model Registry at startup (see app/config.py to change
which model or stage to use).
"""
import uvicorn
from app.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
    )