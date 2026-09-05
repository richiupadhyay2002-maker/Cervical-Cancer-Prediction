"""Model construction and metric aggregation from ``model_training_testing_evaluation.ipynb``."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

METRIC_COLUMNS = ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]


def build_models(random_state: int = 42) -> dict:
    """The eight classifiers compared in the notebook, keyed by display name."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "Linear SVM": SVC(kernel="linear", C=1.0, probability=True, random_state=random_state),
        "RBF SVM": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=random_state),
        "Polynomial SVM": SVC(kernel="poly", degree=3, C=1.0, probability=True, random_state=random_state),
        "Bagging": BaggingClassifier(
            estimator=DecisionTreeClassifier(), n_estimators=100, random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_state),
        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=100,
            learning_rate=1.0,
            random_state=random_state,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=random_state
        ),
    }


def compute_metrics(y_true, y_pred, y_prob=None) -> dict[str, float]:
    """Accuracy/precision/recall/F1 (+ ROC-AUC when probabilities are supplied, else 0.0)."""
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else 0.0
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1-Score": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "ROC-AUC": round(auc, 4),
    }


def summarize_results(
    y_true,
    predictions: Mapping[str, np.ndarray],
    probabilities: Mapping[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """One row per model, sorted by F1-Score descending."""
    probabilities = probabilities or {}
    rows = []
    for name, y_pred in predictions.items():
        row = {"Model": name}
        row.update(compute_metrics(y_true, y_pred, probabilities.get(name)))
        rows.append(row)
    df = pd.DataFrame(rows, columns=METRIC_COLUMNS)
    return df.sort_values("F1-Score", ascending=False, kind="stable").reset_index(drop=True)


def best_model(results: pd.DataFrame, metric: str = "F1-Score") -> str:
    if results.empty:
        raise ValueError("results is empty")
    return str(results.loc[results[metric].idxmax(), "Model"])


def feature_importance_table(model, feature_names: list[str]) -> pd.DataFrame:
    """Feature importances of a fitted tree ensemble, descending."""
    importances = model.feature_importances_
    if len(importances) != len(feature_names):
        raise ValueError("feature_names length does not match model importances")
    return (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
