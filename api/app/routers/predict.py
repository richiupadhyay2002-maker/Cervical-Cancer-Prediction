"""
Prediction endpoint with dynamic model selection.

Accepts patient features as JSON, validates them, runs the model,
and returns the prediction with confidence score (if available).

You can specify which model to use via the `model` query parameter.
If not specified, uses the default model from config.
"""
import logging
from fastapi import APIRouter, HTTPException, Query

from app.models import PredictionInput, PredictionOutput, ErrorResponse
from app.model_loader import get_model, list_available_models, get_prediction_confidence
from app.preprocessor import preprocess
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Prediction"])


@router.get(
    "/models",
    summary="List all available models",
    description="Returns a list of all registered models in the MLflow Model Registry.",
)
async def list_models():
    """
    GET /models
    
    Returns all available models that can be used for prediction.
    """
    try:
        models = list_available_models()
        return {
            "models": models,
            "default_model": settings.MODEL_NAME,
            "total": len(models)
        }
    except Exception as exc:
        logger.error("Failed to list models: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve model list: {exc}"
        )


@router.post(
    "/predict",
    response_model=PredictionOutput,
    responses={
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Prediction error"},
    },
    summary="Predict cervical cancer risk",
    description=(
        "Submit 35 patient features and receive a binary prediction "
        "(0 = Negative, 1 = Positive) with confidence score. "
        "Use the 'model' query parameter to select which model to use. "
        f"Default: {settings.MODEL_NAME}"
    ),
)
async def predict(
    input_data: PredictionInput,
    model: str = Query(
        default=None,
        description="Name of the model to use for prediction. If not specified, uses the default model.",
        examples=["Linear_SVM", "RBF_SVM", "Random_Forest"]
    )
):
    """
    POST /predict
    
    Predicts cervical cancer risk using the specified model.
    
    Query Parameters:
    - model: (optional) Name of the model to use. Examples: Linear_SVM, RBF_SVM, Random_Forest, etc.
    
    Request Body:
    - All 35 patient features (see PredictionInput schema)
    
    Returns:
    - prediction: 0 (Negative) or 1 (Positive)
    - confidence: Probability of being Positive (0-1)
    - model_name: Which model was used
    - model_version: Version of the model used
    """
    try:
        # ── 1. Load the specified model (or default) ─────────────────────────────
        model_name = model if model else settings.MODEL_NAME
        logger.info("Using model: %s", model_name)
        
        try:
            pred_model, model_version, model_stage, _ = get_model(model_name)
        except RuntimeError as exc:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found. Use GET /models to see available models. Error: {exc}"
            )
        
        # ── 2. Convert validated Pydantic model to dict with original column names ──
        raw_features = input_data.model_dump(by_alias=True)
        
        # ── 3. Preprocess: dict → NumPy array in correct column order ──────────────
        features_array = preprocess(raw_features)
        
        # ── 4. Run inference ────────────────────────────────────────────────────────
        prediction = pred_model.predict(features_array)
        pred_class = int(prediction[0])

        # ── 5. Try to get confidence (probability) if the model supports it ─────────
        confidence = get_prediction_confidence(pred_model, features_array)

        # ── 6. Apply the decision threshold ─────────────────────────────────────────
        # The threshold determines the final class. If a confidence score is
        # available, use it; otherwise fall back to the raw model prediction.
        if confidence is not None:
            final_class = 1 if confidence >= settings.PREDICTION_THRESHOLD else 0
        else:
            final_class = pred_class

        # ── 7. Build response ───────────────────────────────────────────────────────
        return PredictionOutput(
            model_name=model_name,
            model_version=model_version,
            prediction=final_class,
            prediction_label="Positive" if final_class == 1 else "Negative",
            confidence=confidence,
            threshold=settings.PREDICTION_THRESHOLD,
        )
    
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        logger.exception("Prediction failed unexpectedly")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")
