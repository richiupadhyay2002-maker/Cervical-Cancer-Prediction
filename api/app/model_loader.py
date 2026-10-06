"""
MLflow Model Registry loader with dynamic model switching.

Loads machine-learning models from the MLflow Model Registry, caches them
in memory for performance, and lets the API switch between models at
runtime via query parameters. This is a core MLOps capability: models are
versioned, staged (None -> Staging -> Production), and can be rolled back.
"""
import logging
import re

import mlflow
import mlflow.pyfunc
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from app.config import settings

logger = logging.getLogger(__name__)

# In-memory cache: model_name -> {"model", "version", "stage"}
_model_cache: dict = {}

# Metadata for the default model loaded at startup
_default_model_name: str | None = None
_default_model_version: int | None = None
_default_model_stage: str | None = None


def normalize_model_name(name: str) -> str:
    """Normalize model names to be robust to spacing and underscore differences."""
    return re.sub(r"[^a-z0-9]+", "_", (name or "").strip().lower()).strip("_")


def _resolve_registered_model_name(client, model_name: str) -> str:
    """Resolve human-friendly names like 'Gradient Boosting' to exact registry names."""
    if not model_name:
        return model_name

    registered_models = client.search_registered_models()
    if any(rm.name == model_name for rm in registered_models):
        return model_name

    normalized_name = normalize_model_name(model_name)
    for rm in registered_models:
        if normalize_model_name(rm.name) == normalized_name:
            logger.info("Resolved model alias '%s' to registry name '%s'", model_name, rm.name)
            return rm.name

    return model_name


def _load_model_from_registry(model_name: str, stage: str):
    """
    Load a model from the MLflow Model Registry.

    Prefers a version in the requested ``stage``; if none exists, falls back
    to the latest registered version (logging a warning). Raises RuntimeError
    if the model cannot be found or loaded.
    """
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    client = MlflowClient()
    model_name = _resolve_registered_model_name(client, model_name)

    version: int | None = None
    actual_stage = stage

    # 1) Try the requested stage first
    try:
        versions_in_stage = client.get_latest_versions(model_name, stages=[stage])
    except MlflowException:
        versions_in_stage = []

    if versions_in_stage:
        latest = versions_in_stage[0]
        version = int(latest.version)
        actual_stage = latest.current_stage
    else:
        # 2) Fall back to the latest version of the model
        all_versions = client.search_model_versions(f"name='{model_name}'")
        sorted_versions = sorted(
            all_versions, key=lambda v: int(v.version), reverse=True
        )
        if sorted_versions:
            version = int(sorted_versions[0].version)
            actual_stage = sorted_versions[0].current_stage
            logger.warning(
                "No version in '%s' stage for '%s'. Using v%d (stage=%s).",
                stage, model_name, version, actual_stage,
            )
        else:
            raise RuntimeError(
                f"No versions found for model '{model_name}'."
            )

    model_uri = f"models:/{model_name}/{version}"
    logger.info("Loading model from %s", model_uri)
    try:
        model = mlflow.pyfunc.load_model(model_uri=model_uri)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load model '{model_name}' version {version}. "
            f"Ensure the model is registered. Original error: {exc}"
        ) from exc

    logger.info("Loaded model: %s v%d (stage=%s)", model_name, version, actual_stage)
    return model, version, actual_stage


def load_default_model():
    """Load the default model specified in config at startup (called once)."""
    global _default_model_name, _default_model_version, _default_model_stage
    try:
        model, version, stage = _load_model_from_registry(
            settings.MODEL_NAME, settings.MODEL_STAGE
        )
        _default_model_name = settings.MODEL_NAME
        _model_cache[_default_model_name] = {
            "model": model, "version": version, "stage": stage,
        }
        _default_model_version = version
        _default_model_stage = stage
        logger.info("✓ Default model loaded: %s v%d", settings.MODEL_NAME, version)
        return model
    except Exception as exc:
        logger.error("✗ Failed to load default model: %s", exc)
        raise RuntimeError(str(exc))


def get_model(model_name: str | None = None):
    """
    Get a model by name, loading it from the registry if not cached.

    Returns ``(model, version, stage, native_model)``.
    """
    if model_name is None:
        model_name = _default_model_name

    cached = _model_cache.get(model_name)
    if cached:
        return cached["model"], cached["version"], cached["stage"], None

    logger.info("Model '%s' not in cache, loading from registry...", model_name)
    try:
        model, version, stage = _load_model_from_registry(model_name, "Production")
        _model_cache[model_name] = {
            "model": model, "version": version, "stage": stage,
        }
        return model, version, stage, None
    except Exception as exc:
        logger.error("Failed to load model '%s': %s", model_name, exc)
        raise RuntimeError(str(exc))


def get_default_model():
    """Return the default model loaded at startup: ``(model, version, stage, native)``."""
    cached = _model_cache.get(_default_model_name)
    if cached:
        return cached["model"], cached["version"], cached["stage"], None
    raise RuntimeError("Default model not loaded. Call load_default_model() first.")


def get_native_model(pyfunc_model):
    """
    Safely extract the underlying native scikit-learn model from an MLflow
    pyfunc wrapper (if present), otherwise return ``None``.
    """
    native = getattr(pyfunc_model, "_model_impl", None)
    if native is None:
        return None
    for attr in ("_model", "_flavor_backend", "model_impl"):
        candidate = getattr(native, attr, None)
        if candidate is not None and hasattr(candidate, "predict"):
            native = candidate
    return native


def has_probability_support(pyfunc_model) -> bool:
    """Return True if the loaded model exposes ``predict_proba``."""
    native = get_native_model(pyfunc_model)
    return native is not None and hasattr(native, "predict_proba")


def get_prediction_confidence(pyfunc_model, features_array):
    """
    Return the positive-class probability from a loaded model, or ``None``
    if the model does not support probability predictions.
    """
    if not has_probability_support(pyfunc_model):
        return None
    native = get_native_model(pyfunc_model)
    try:
        proba = native.predict_proba(features_array)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return float(proba[0][1])
        return float(proba[0][0])
    except Exception as exc:
        logger.warning("Could not extract confidence from model: %s", exc)
        return None


def list_available_models() -> list[dict]:
    """
    List all registered models in the MLflow Model Registry.

    Returns a list of dicts with keys: name, version, stage, description.
    """
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    client = MlflowClient()
    models: list[dict] = []
    try:
        registered_models = client.search_registered_models()
        for rm in registered_models:
            name = rm.name
            # Skip MLflow demo models that may exist in the registry
            if name.startswith("mlflow-demo"):
                continue
            versions = client.search_model_versions(f"name='{name}'")
            sorted_versions = sorted(
                versions, key=lambda v: int(v.version), reverse=True
            )
            latest = sorted_versions[0] if sorted_versions else None
            models.append({
                "name": name,
                "version": int(latest.version) if latest else "",
                "stage": latest.current_stage if latest else "",
                "description": rm.description or "",
            })
    except Exception as exc:
        logger.error("Failed to list models: %s", exc)
    return models
