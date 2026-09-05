"""Model training and evaluation helpers shared across the notebooks."""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(y_true, y_pred, y_prob=None):
    """Return a dict of the standard classification metrics."""
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        metrics["ROC-AUC"] = roc_auc_score(y_true, y_prob)
    return metrics


def print_metrics(name, metrics, y_true=None, y_pred=None,
                  show_confusion=False, show_report=False):
    """Print a banner followed by aligned metric values."""
    print("=" * 50)
    print(name.upper())
    print("=" * 50)
    width = max(len(k) for k in metrics) + 1
    for key, value in metrics.items():
        print(f"{key + ':':<{width}} {value:.4f}")
    if show_confusion:
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
    if show_report:
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))


def train_and_evaluate(name, model, X_train, y_train, X_test, y_test,
                       models, predictions, probabilities,
                       show_confusion=False, show_report=False):
    """Fit ``model``, record its outputs in the registries and print metrics.

    ``probabilities`` is only populated when the estimator supports
    ``predict_proba``; ROC-AUC is omitted otherwise.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        probabilities[name] = y_prob

    models[name] = model
    predictions[name] = y_pred

    metrics = evaluate_model(y_test, y_pred, y_prob)
    print_metrics(name, metrics, y_test, y_pred, show_confusion, show_report)
    return model


def compare_models(y_test, predictions, probabilities, sort_by="F1-Score"):
    """Build a metrics table for every model, sorted by ``sort_by`` descending."""
    rows = []
    for name, y_pred in predictions.items():
        metrics = evaluate_model(y_test, y_pred, probabilities.get(name))
        metrics.setdefault("ROC-AUC", 0.0)
        rows.append({"Model": name, **{k: round(v, 4) for k, v in metrics.items()}})
    return (
        pd.DataFrame(rows)
        .sort_values(sort_by, ascending=False)
        .reset_index(drop=True)
    )
