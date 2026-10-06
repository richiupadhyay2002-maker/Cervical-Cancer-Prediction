"""
Health check endpoint.

Returns the current status of the service including metadata about the
loaded model. Use this endpoint for monitoring and readiness probes.
"""
import logging
from fastapi import APIRouter

from app.models import HealthResponse
from app.model_loader import get_default_model
from app.preprocessor import load_feature_columns
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    description=(
        "Returns the current health status of the prediction service "
        "along with metadata about the loaded model (name, version, stage)."
    ),
)
async def health_check():
    """
    GET /health

    Use this endpoint to verify the service is running and the model
    was loaded successfully.
    """
    features = load_feature_columns()
    
    # Get default model info
    _, model_version, model_stage, _ = get_default_model()

    return HealthResponse(
        status="ok",
        model_name=settings.MODEL_NAME,
        model_version=model_version,
        model_stage=model_stage,
        features_count=len(features),
    )
