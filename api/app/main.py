"""
FastAPI application factory for the Cervical Cancer Prediction Service.

Creates the FastAPI app, registers CORS middleware and the health/predict/
report routers, adds global exception handlers, and loads the default model
from the MLflow Model Registry at startup via a lifespan context manager.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.model_loader import load_default_model
from app.models import ErrorResponse
from app.routers import health, predict, report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.

    - **Startup:** Loads the model from the MLflow Model Registry once.
      If loading fails, the server still starts but /predict returns 500s,
      and the reason is logged clearly.
    """
    logger.info("=" * 60)
    logger.info("  Cervical Cancer Prediction Service — Starting up")
    try:
        load_default_model()
        logger.info("✓ Model loaded successfully from registry")
    except RuntimeError as exc:
        logger.error("✗ Failed to load model: %s", exc)
        logger.error(
            "The server will start but /predict will return 500 errors. "
            "Check that model '%s' exists and has a version in the '%s' stage.",
            settings.MODEL_NAME,
            settings.MODEL_STAGE,
        )
    logger.info("Ready to serve predictions on http://%s:%d", settings.HOST, settings.PORT)
    logger.info("Swagger UI: http://localhost:%d/docs", settings.PORT)
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Cervical Cancer Risk Prediction API",
    description=(
        "Serve predictions from a registered MLflow model. Accepts 35 patient "
        "features and returns a binary risk assessment (0 = Negative, "
        "1 = Positive) with a confidence score and optional clinical report."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers: /health, /models, /predict, /predict/report
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(report.router)


@app.exception_handler(422)
async def validation_exception_handler(request: Request, exc):
    """Handle Pydantic validation errors (422 Unprocessable Entity)."""
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(error="Validation Error", detail=str(exc)).model_dump(),
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc):
    """Catch-all handler for unexpected server errors."""
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error", detail=str(exc)
        ).model_dump(),
    )


@app.get("/", include_in_schema=False)
async def root():
    """Service root — returns metadata about the API and its endpoints."""
    return {
        "service": "Cervical Cancer Risk Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "endpoints": {
            "health": "/health",
            "models": "/models",
            "predict": "/predict",
            "report": "/predict/report",
        },
    }
