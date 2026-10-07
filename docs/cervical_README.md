# 🏥 Cervical Cancer Risk Prediction API — Guide

<div align="center">

![Python](https://img.shields.io/badge/Python-3.13+-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green?style=for-the-badge&logo=fastapi)
![MLflow](https://img.shields.io/badge/MLflow-2.22-orange?style=for-the-badge&logo=mlflow)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?style=for-the-badge&logo=docker)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.5-yellow?style=for-the-badge&logo=scikit-learn)

**A FastAPI service serving 9 registered MLflow models for cervical cancer risk prediction**

</div>

---

## 🎯 Overview

This service exposes a trained, versioned MLflow Model Registry behind a FastAPI
REST API. It accepts 35 patient features and returns a binary risk assessment
with a confidence score and an optional Markdown clinical report.

- **9 registered models** selectable per request
- **Input validation** on all 35 features (Pydantic v2)
- **Model caching** for fast repeat predictions
- **Clinical report generation** via a Markdown template
- **Docker** packaging for the API + MLflow UI

> **Research use only.** Not a medical device; must not be used for diagnosis or
> screening.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.13+
- The pipeline has been run once (so `notebooks/mlflow.db` and
  `processed_data/feature_columns.json` exist)

### Install and run

```bash
cd api
pip install -r requirements.txt
python run.py
```

### Access

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **Health:** http://localhost:8000/health
- **Models:** http://localhost:8000/models

### Docker

```bash
cd docker
docker-compose -f cervical_docker-compose.yml up --build
# API: http://localhost:8000/docs   MLflow UI: http://localhost:5000
```

---

## 📖 API Reference

### 1. Health check — `GET /health`

Returns service status plus metadata about the loaded model.

```json
{
  "status": "ok",
  "model_name": "Gradient_Boosting",
  "model_version": 1,
  "model_stage": "Production",
  "features_count": 35
}
```

### 2. List models — `GET /models`

```json
{
  "models": [
    {"name": "Gradient_Boosting", "version": 1, "stage": "Production", "description": "..."},
    {"name": "RBF_SVM", "version": 1, "stage": "None", "description": "..."}
  ],
  "default_model": "Gradient_Boosting",
  "total": 9
}
```

### 3. Predict — `POST /predict`

Optional query parameter `model` selects a registered model (default:
`Gradient_Boosting`). Request body = the 35 patient features
(see `api/demo_prediction.json` for a complete payload).

```bash
curl -X POST "http://localhost:8000/predict?model=Gradient_Boosting" \
  -H "Content-Type: application/json" \
  -d @demo_prediction.json
```

```json
{
  "model_name": "Gradient_Boosting",
  "model_version": 1,
  "prediction": 1,
  "prediction_label": "Positive",
  "confidence": 0.5945,
  "threshold": 0.5
}
```

### 4. Predict + report — `POST /predict/report`

Same request body; adds `risk_level` and a Markdown `report_markdown` report.

```json
{
  "model_name": "Gradient_Boosting",
  "model_version": 1,
  "prediction": 1,
  "prediction_label": "Positive",
  "confidence": 0.5945,
  "risk_level": "High Risk",
  "report_markdown": "# Cervical Cancer Risk Assessment Report\n..."
}
```

### Response fields

- `prediction` — `0` (Negative) or `1` (Positive)
- `prediction_label` — human-readable label
- `confidence` — probability of the positive class (0–1), when the model supports it
- `model_name` / `model_version` — which registry model answered
- `threshold` — decision threshold used (0.5)

---

## 🏆 Available Models

The 9 registered models are: `Gradient_Boosting`, `Hist_Gradient_Boosting`,
`Random_Forest`, `LDA_Shrinkage`, `Linear_SVM`, `Logistic_ElasticNet`,
`Logistic_Regression`, `RBF_SVM`, `SGD_ElasticNet`.

`Gradient_Boosting` is promoted to **Production** and is the API default; the
others are registered with stage `None` and can be selected per request.

| Model | Test F1 | Test ROC-AUC |
|-------|---------|--------------|
| RBF_SVM | 0.6667 | 0.8450 |
| Gradient_Boosting | 0.6316 | 0.8543 |
| Hist_Gradient_Boosting | 0.6316 | 0.8667 |
| LDA_Shrinkage | 0.6316 | 0.9029 |
| Linear_SVM | 0.6316 | 0.7438 |
| Logistic_ElasticNet | 0.6316 | 0.8533 |
| Logistic_Regression | 0.6316 | 0.7748 |
| Random_Forest | 0.6316 | 0.8554 |
| SGD_ElasticNet | 0.6000 | 0.8223 |

> Different models can give different predictions for the same patient — that is
> expected and useful for comparing model perspectives.

---

## 💡 Examples

### Python

```python
import json
import urllib.request

payload = json.load(open("api/demo_prediction.json"))
req = urllib.request.Request(
    "http://localhost:8000/predict?model=Gradient_Boosting",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
result = json.loads(urllib.request.urlopen(req).read().decode())
print(result["prediction_label"], result["confidence"])
```

### cURL

```bash
curl http://localhost:8000/health
curl http://localhost:8000/models
curl -X POST "http://localhost:8000/predict?model=RBF_SVM" \
  -H "Content-Type: application/json" \
  -d @api/demo_prediction.json
```

---

## 🔧 Configuration

All settings live in `api/app/config.py` and can be overridden via environment
variables (or a `.env` file):

| Setting | Default | Purpose |
|---------|---------|---------|
| `MLFLOW_TRACKING_URI` | `sqlite:///<root>/notebooks/mlflow.db` | Registry backend |
| `MODEL_NAME` | `Gradient_Boosting` | Default model |
| `MODEL_STAGE` | `Production` | Default stage |
| `FEATURE_COLUMNS_PATH` | `<root>/processed_data/feature_columns.json` | Feature order |
| `PREDICTION_THRESHOLD` | `0.5` | Decision threshold |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | Server bind |

---

## 🗂️ Project Structure

```
api/
├── app/
│   ├── main.py            # FastAPI app, CORS, error handlers, lifespan
│   ├── config.py          # Settings
│   ├── model_loader.py    # MLflow loading, caching, model listing
│   ├── models.py          # Pydantic schemas (35 features)
│   ├── preprocessor.py    # Feature-order alignment
│   ├── report_generator.py
│   ├── templates/risk_report_template.md
│   └── routers/{health,predict,report}.py
├── run.py
├── requirements.txt
└── demo_prediction.json
```

---

## 🐳 Docker

The compose stack runs the API (port 8000) and the MLflow UI (port 5000) and
shares `notebooks/mlflow.db` + `mlruns/` as volumes.

```bash
cd docker
docker-compose -f cervical_docker-compose.yml up --build
docker-compose -f cervical_docker-compose.yml down
```

---

## 🧪 Testing

```bash
cd api
pip install -r requirements.txt pytest
pytest tests -q
```

The tests run without MLflow artifacts: they stub the registry with a small
scikit-learn pipeline and generate `feature_columns.json` from
`demo_prediction.json`.

---

## 🔍 Troubleshooting

| Symptom | Fix |
|---------|-----|
| `/predict` returns 500 "No versions found" | Run the pipeline so `notebooks/mlflow.db` is populated, then restart |
| `/predict` returns 404 for a model | Use an exact registered name from `GET /models` (e.g. `Linear_SVM`) |
| `Feature columns file not found` | Run `cervical_02_feature_engineering_advanced.ipynb` to generate `processed_data/feature_columns.json` |
| Port already in use | Change `PORT` in `app/config.py` |
