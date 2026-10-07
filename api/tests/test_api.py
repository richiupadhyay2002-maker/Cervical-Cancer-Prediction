"""API tests that run without MLflow artifacts.

A small scikit-learn pipeline stands in for the registry model, and
``feature_columns.json`` is generated from ``api/demo_prediction.json`` so the
tests exercise the real routers, schemas, preprocessing, and response handling
without needing a populated MLflow store.
"""
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app import main
from app.config import settings
from app.routers import health, predict, report

DEMO = json.loads(
    (Path(__file__).resolve().parents[1] / "demo_prediction.json").read_text()
)
FEATURES = list(DEMO)


class _Impl:
    def __init__(self, model):
        self._model = model


class FakePyfunc:
    """Mimics the parts of an MLflow pyfunc model the API uses."""

    def __init__(self, model):
        self._model_impl = _Impl(model)
        self._sk = model

    def predict(self, X):
        return self._sk.predict(X)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    columns = tmp_path / "feature_columns.json"
    columns.write_text(json.dumps(FEATURES))
    monkeypatch.setattr(settings, "FEATURE_COLUMNS_PATH", str(columns))

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, len(FEATURES)))
    y = (X[:, 0] > 0).astype(int)
    sk = Pipeline(
        [("imputer", SimpleImputer()), ("model", LogisticRegression())]
    ).fit(X, y)
    fake = FakePyfunc(sk)

    def fake_get_model(name=None):
        if name not in (None, settings.MODEL_NAME):
            raise RuntimeError(f"No versions found for model '{name}'.")
        return fake, 1, "Production", None

    monkeypatch.setattr(main, "load_default_model", lambda: fake)
    monkeypatch.setattr(predict, "get_model", fake_get_model)
    monkeypatch.setattr(report, "get_model", fake_get_model)
    monkeypatch.setattr(
        health, "get_default_model", lambda: (fake, 1, "Production", None)
    )
    with TestClient(main.app) as c:
        yield c


def test_demo_payload_has_35_features():
    assert len(FEATURES) == 35


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_stage"] == "Production"
    assert body["features_count"] == 35


def test_predict(client):
    r = client.post("/predict", json=DEMO)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert body["prediction_label"] == (
        "Positive" if body["prediction"] else "Negative"
    )
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_rejects_unknown_field(client):
    assert client.post("/predict", json={**DEMO, "Unknown": 1}).status_code == 422


def test_predict_rejects_out_of_range(client):
    assert client.post("/predict", json={**DEMO, "Age": 500}).status_code == 422


def test_unknown_model_returns_404(client):
    assert client.post("/predict?model=Does_Not_Exist", json=DEMO).status_code == 404


def test_report_returns_markdown(client):
    r = client.post("/predict/report", json=DEMO)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["risk_level"] in ("High Risk", "Low Risk")
    assert "# Cervical Cancer Risk Assessment Report" in body["report_markdown"]
