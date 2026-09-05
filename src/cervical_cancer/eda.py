"""Non-plotting analysis helpers from ``eda.ipynb``."""

from __future__ import annotations

import pandas as pd


def detect_outliers_iqr(data: pd.DataFrame, column: str, k: float = 1.5) -> tuple[int, float, float]:
    """Return (outlier count, lower bound, upper bound) using the IQR rule."""
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    mask = (data[column] < lower) | (data[column] > upper)
    return int(mask.sum()), lower, upper


def target_correlations(df: pd.DataFrame, target: str = "Biopsy") -> pd.Series:
    """Correlation of every other numeric column with ``target``, descending."""
    corr = df.corr(numeric_only=True)
    return corr[target].drop(target).sort_values(ascending=False)


def class_balance(y: pd.Series) -> dict[str, float]:
    """Counts and positive-class percentage for a binary target."""
    counts = y.value_counts()
    n = int(len(y))
    pos = int(counts.get(1, 0))
    neg = int(counts.get(0, 0))
    return {
        "negative": neg,
        "positive": pos,
        "positive_pct": (pos / n * 100) if n else 0.0,
    }
