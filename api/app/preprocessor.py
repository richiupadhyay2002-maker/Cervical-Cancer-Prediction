"""
Input preprocessing: converts validated Pydantic input into a model-ready
NumPy array.

The preprocessor guarantees features are assembled in the *exact same order*
as the columns used during model training. Feeding a model features in the
wrong order silently produces garbage predictions, so this order is enforced
from the ``feature_columns.json`` file saved at training time.
"""
import json
import logging

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

_feature_columns: list[str] = []


def load_feature_columns() -> list[str]:
    """
    Load the list of feature column names from the JSON file.

    Returns
    -------
    list[str]
        The 35 feature names, in model-training order.
    """
    global _feature_columns
    path = settings.FEATURE_COLUMNS_PATH
    try:
        with open(path, "r") as f:
            _feature_columns = json.load(f)
    except FileNotFoundError as exc:
        raise ValueError(
            f"Feature columns file not found at: {path}. "
            f"Check 'FEATURE_COLUMNS_PATH' in app/config.py."
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Feature columns file is not valid JSON: {path}. Error: {exc}"
        ) from exc
    logger.info("Loaded %d feature columns", len(_feature_columns))
    return _feature_columns


def preprocess(input_data: dict) -> np.ndarray:
    """
    Convert a validated input dictionary to a 2D NumPy array suitable for
    ``model.predict()``.

    Parameters
    ----------
    input_data : dict
        Payload dumped with ``model_dump(by_alias=True)`` (original column names).

    Returns
    -------
    np.ndarray
        Shape ``(1, n_features)`` float32 array in training column order.
    """
    feature_columns = load_feature_columns()
    expected_count = len(feature_columns)

    if len(input_data) != expected_count:
        raise ValueError(
            f"Expected {expected_count} features, got {len(input_data)}. "
            f"Check that all {expected_count} features are provided."
        )

    feature_vector = [
        float(input_data[col]) if input_data.get(col) is not None else np.nan
        for col in feature_columns
    ]
    return np.array(feature_vector, dtype=np.float32).reshape(1, -1)
