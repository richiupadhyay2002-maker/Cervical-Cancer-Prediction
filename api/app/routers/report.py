"""
Report generation endpoint.

Accepts the same 35 patient features as /predict, runs the model,
and returns a full Markdown clinical decision support report.
"""
import logging
from fastapi import APIRouter, HTTPException, Query

from app.models import PredictionInput, ReportOutput, ErrorResponse
from app.model_loader import get_feature_importance, get_model, get_prediction_confidence
from app.preprocessor import load_feature_columns, preprocess
from app.config import settings
from app.report_generator import generate_risk_report

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Report"])


@router.post(
    "/predict/report",
    response_model=ReportOutput,
    responses={
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Prediction error"},
    },
    summary="Predict cervical cancer risk and generate clinical report",
    description=(
        "Submit 35 patient features and receive a binary prediction "
        "(0 = Negative, 1 = Positive) with confidence score, plus a "
        "Markdown decision-support report listing the patient's recorded risk "
        "factors, the model's global top features (when the model exposes "
        "feature importances or coefficients), and general recommendations."
    ),
)
async def predict_with_report(
    input_data: PredictionInput,
    model: str = Query(
        default=None,
        description="Name of the model to use for prediction. If not specified, uses the default model.",
        examples=["Gradient_Boosting", "RBF_SVM", "Random_Forest"]
    )
):
    """
    POST /predict/report

    Predicts cervical cancer risk and generates a clinical report.

    Query Parameters:
    - model: (optional) Name of the model to use.

    Request Body:
    - All 35 patient features (see PredictionInput schema)

    Returns:
    - prediction: 0 (Negative) or 1 (Positive)
    - confidence: Probability of being Positive (0-1)
    - risk_level: High Risk or Low Risk
    - report_markdown: Full Markdown clinical report
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

        # ── 7. Generate the clinical report ────────────────────────────────────────
        # Use a default probability of 0.5 if confidence is not available
        prob_for_report = confidence if confidence is not None else 0.5

        report_markdown = generate_risk_report(
            prediction=final_class,
            probability=prob_for_report,
            patient_features=raw_features,
            feature_importance=get_feature_importance(pred_model, load_feature_columns()),
            model_name=model_name,
        )

        risk_level = "High Risk" if final_class == 1 else "Low Risk"

        # ── 8. Build response ───────────────────────────────────────────────────────
        return ReportOutput(
            model_name=model_name,
            model_version=model_version,
            prediction=final_class,
            prediction_label="Positive" if final_class == 1 else "Negative",
            confidence=confidence,
            risk_level=risk_level,
            report_markdown=report_markdown,
        )

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        logger.exception("Report generation failed unexpectedly")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {exc}")