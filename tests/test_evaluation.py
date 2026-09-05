import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from cervical_cancer import evaluation as ev

EXPECTED_MODELS = {
    "Logistic Regression", "Linear SVM", "RBF SVM", "Polynomial SVM",
    "Bagging", "Random Forest", "AdaBoost", "Gradient Boosting",
}


def test_build_models_names_and_seeds():
    models = ev.build_models(random_state=3)
    assert set(models) == EXPECTED_MODELS
    for m in models.values():
        assert m.get_params()["random_state"] == 3
    assert models["Polynomial SVM"].degree == 3
    assert models["AdaBoost"].estimator.max_depth == 1


def test_compute_metrics_perfect_prediction():
    y = np.array([0, 0, 1, 1])
    m = ev.compute_metrics(y, y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert m == {"Accuracy": 1.0, "Precision": 1.0, "Recall": 1.0, "F1-Score": 1.0, "ROC-AUC": 1.0}


def test_compute_metrics_without_probabilities_sets_auc_zero_and_handles_no_positive_preds():
    y = np.array([0, 0, 1, 1])
    pred = np.zeros(4, dtype=int)
    m = ev.compute_metrics(y, pred)
    assert m["ROC-AUC"] == 0.0
    assert m["Precision"] == 0.0  # zero_division=0, no warning/NaN
    assert m["Recall"] == 0.0
    assert m["Accuracy"] == 0.5


def test_compute_metrics_rounds_to_4dp():
    y = np.array([0, 0, 1, 1, 1, 1])
    pred = np.array([0, 1, 1, 1, 1, 0])
    m = ev.compute_metrics(y, pred)
    assert m["Accuracy"] == round(4 / 6, 4)
    assert m["Precision"] == 0.75
    assert m["F1-Score"] == round(2 * 0.75 * 0.75 / 1.5, 4)


def test_summarize_results_sorted_by_f1_and_missing_probs():
    y = np.array([0, 0, 1, 1])
    predictions = {
        "bad": np.array([0, 0, 0, 0]),
        "good": np.array([0, 0, 1, 1]),
        "mid": np.array([0, 1, 1, 1]),
    }
    probabilities = {"good": np.array([0.1, 0.2, 0.9, 0.8])}
    res = ev.summarize_results(y, predictions, probabilities)
    assert list(res.columns) == ev.METRIC_COLUMNS
    assert res["Model"].tolist() == ["good", "mid", "bad"]
    assert res.loc[0, "ROC-AUC"] == 1.0
    assert res.loc[1, "ROC-AUC"] == 0.0
    assert res.index.tolist() == [0, 1, 2]


def test_summarize_results_no_probabilities_argument():
    y = np.array([0, 1])
    res = ev.summarize_results(y, {"m": np.array([0, 1])})
    assert len(res) == 1 and res.loc[0, "ROC-AUC"] == 0.0


def test_summarize_results_empty():
    res = ev.summarize_results(np.array([0, 1]), {})
    assert res.empty and list(res.columns) == ev.METRIC_COLUMNS


def test_best_model_by_metric_and_empty_raises():
    res = pd.DataFrame(
        {"Model": ["a", "b"], "F1-Score": [0.5, 0.9], "ROC-AUC": [0.99, 0.1]},
    )
    assert ev.best_model(res) == "b"
    assert ev.best_model(res, metric="ROC-AUC") == "a"
    with pytest.raises(ValueError):
        ev.best_model(res.iloc[0:0])


def test_feature_importance_table_sorted():
    X = np.array([[0, 1], [0, 2], [1, 3], [1, 4]] * 5, dtype=float)
    y = np.array([0, 0, 1, 1] * 5)
    rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
    table = ev.feature_importance_table(rf, ["f0", "f1"])
    assert list(table.columns) == ["Feature", "Importance"]
    assert table["Importance"].is_monotonic_decreasing
    assert table["Importance"].sum() == pytest.approx(1.0)
    assert set(table["Feature"]) == {"f0", "f1"}


def test_feature_importance_table_length_mismatch_raises():
    rf = RandomForestClassifier(n_estimators=2, random_state=0).fit([[0, 1], [1, 0]], [0, 1])
    with pytest.raises(ValueError):
        ev.feature_importance_table(rf, ["only_one"])


def test_models_train_and_summarize_on_synthetic_data(synthetic_dataset):
    """Smoke test the full model loop with fast models only."""
    from cervical_cancer.preprocessing import prepare_train_test

    X, y = synthetic_dataset
    X_tr, X_te, y_tr, y_te, _ = prepare_train_test(X, y)
    models = ev.build_models()
    preds, probs = {}, {}
    for name in ["Logistic Regression", "Random Forest"]:
        m = models[name].fit(X_tr, y_tr)
        preds[name] = m.predict(X_te)
        probs[name] = m.predict_proba(X_te)[:, 1]
    res = ev.summarize_results(y_te, preds, probs)
    assert len(res) == 2
    scores = res[["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]]
    assert ((scores >= 0) & (scores <= 1)).all().all()
    assert ev.best_model(res) in preds
