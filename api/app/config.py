"""
Configuration settings for the Cervical Cancer Prediction Service.

Centralises all configurable values in one place using Pydantic Settings
(12-Factor App best practice). Values can be overridden via environment
variables or a ``.env`` file, making the service portable across machines.
"""
import os
from pathlib import Path

from pydantic_settings import BaseSettings

# ---------------------------------------------------------------------------
# Project root: <project>/api/app/config.py -> parents[0]=app,
# parents[1]=api, parents[2]=cervical-cancer-mlops
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application configuration, validated at import time."""

    # --- MLflow Model Registry backend (SQLite) ---------------------------
    MLFLOW_TRACKING_URI: str = (
        "sqlite:///" + PROJECT_ROOT.as_posix() + "/notebooks/mlflow.db"
    )

    # --- Default model served by the API -----------------------------------
    # Use the exact registered MLflow model name, not the display label.
    MODEL_NAME: str = "Gradient_Boosting"
    MODEL_STAGE: str = "Production"  # Gradient_Boosting v4 promoted to Production

    # --- Feature columns used during training (order matters!) -------------
    FEATURE_COLUMNS_PATH: str = str(
        PROJECT_ROOT / "processed_data" / "feature_columns.json"
    )

    # --- Classification decision threshold ---------------------------------
    PREDICTION_THRESHOLD: float = 0.5

    # --- Uvicorn server settings --------------------------------------------
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Shared singleton instance (imported everywhere: `from app.config import settings`)
settings = Settings()
