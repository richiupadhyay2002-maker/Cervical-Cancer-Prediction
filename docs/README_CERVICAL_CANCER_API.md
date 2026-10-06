# Cervical Cancer Risk Prediction API - Complete Guide

## Project Overview

This is a complete MLOps project for cervical cancer risk prediction using machine learning. It includes:
- **8 trained ML models** registered in MLflow Model Registry
- **FastAPI REST API** for making predictions
- **Dynamic model selection** - choose which model to use per request
- **Docker containerization** - deploy anywhere consistently

---

## Quick Start

### Option 1: Run Without Docker (Current - Already Working)

The API is already running! Access it at:
- **Swagger UI**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **MLflow UI**: http://localhost:5000

### Option 2: Run With Docker (After Starting Docker Desktop)

```bash
# Start all services
docker-compose up --build

# Access
# - API: http://localhost:8000/docs
# - MLflow: http://localhost:5000
```

---

## How to Use the API

### 1. Open Swagger UI

Go to: **http://localhost:8000/docs**

You'll see 3 endpoints:
- `GET /health` - Check service status
- `GET /models` - List all available models
- `POST /predict` - Make predictions

### 2. List Available Models

**In Swagger UI:**
1. Click `GET /models`
2. Click "Try it out"
3. Click "Execute"

**Or use curl:**
```bash
curl http://localhost:8000/models
```

**Response:**
```json
{
  "models": [
    {"name": "Linear_SVM", "version": 1, "stage": "Production"},
    {"name": "AdaBoost", "version": 1, "stage": "Production"},
    {"name": "Bagging", "version": 1, "stage": "Production"},
    ...
  ],
  "default_model": "Linear_SVM",
  "total": 9
}
```

### 3. Make a Prediction

**In Swagger UI:**

1. Click `POST /predict`
2. Click "Try it out"
3. **Select a model** (optional):
   - In the "model" field, type a model name like `AdaBoost`
   - Or leave empty to use default (Linear_SVM)
4. **Paste patient data** in the request body:
```json
{
  "age": 25,
  "number_of_sexual_partners": 1,
  "first_sexual_intercourse": 22,
  "num_of_pregnancies": 0,
  "smokes_years": 0,
  "smokes_packs_per_year": 0,
  "hormonal_contraceptives_years": 0,
  "iud_years": 0,
  "stds_number": 0,
  "stds_condylomatosis": 0,
  "stds_cervical_condylomatosis": 0,
  "stds_vaginal_condylomatosis": 0,
  "stds_vulvo_perineal_condylomatosis": 0,
  "stds_syphilis": 0,
  "stds_pelvic_inflammatory_disease": 0,
  "stds_genital_herpes": 0,
  "stds_molluscum_contagiosum": 0,
  "stds_aids": 0,
  "stds_hiv": 0,
  "stds_hepatitis_b": 0,
  "stds_hpv": 0,
  "stds_number_of_diagnosis": 0,
  "dx_cancer": 0,
  "dx_cin": 0,
  "dx_hpv": 0,
  "dx": 0,
  "hinselmann": 0,
  "schiller": 0,
  "citology": 0
}
```
5. Click "Execute"
6. View the prediction result

**Using curl:**
```bash
# Default model (Linear_SVM)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"age\":25,\"number_of_sexual_partners\":1,...}"

# Specific model (AdaBoost)
curl -X POST "http://localhost:8000/predict?model=AdaBoost" \
  -H "Content-Type: application/json" \
  -d "{\"age\":25,\"number_of_sexual_partners\":1,...}"
```

**Using Python:**
```python
import urllib.request
import json

url = "http://localhost:8000/predict?model=AdaBoost"
data = {
    "age": 25,
    "number_of_sexual_partners": 1,
    # ... all 29 features
}

req = urllib.request.Request(
    url,
    data=json.dumps(data).encode(),
    headers={"Content-Type": "application/json"},
    method="POST"
)

response = urllib.request.urlopen(req)
result = json.loads(response.read().decode())
print(result)
```

### 4. Understanding the Response

```json
{
  "model_name": "AdaBoost",
  "model_version": 1,
  "prediction": 1,
  "prediction_label": "Positive",
  "confidence": 0.594546,
  "threshold": 0.5
}
```

**Fields:**
- `model_name`: Which model made the prediction
- `model_version`: Version of the model
- `prediction`: 0 (Negative/low risk) or 1 (Positive/high risk)
- `prediction_label`: Human-readable label
- `confidence`: Probability of being Positive (0-1)
- `threshold`: Decision threshold used (0.5)

---

## Available Models

| Model Name | F1-Score | AUC | Best For |
|------------|----------|-----|----------|
| **Linear_SVM** | 0.8696 | 0.9534 | ⭐ Best overall (default) |
| **AdaBoost** | 0.8696 | 0.9470 | ⭐ Tied for best |
| Bagging | 0.8571 | 0.9476 | Ensemble method |
| Gradient_Boosting | 0.8571 | 0.9307 | Sequential learning |
| Logistic_Regression | 0.8333 | 0.9545 | Interpretable |
| Polynomial_SVM | 0.8182 | 0.9371 | Non-linear patterns |
| Random_Forest | 0.7826 | 0.9263 | Feature importance |
| RBF_SVM | 0.6000 | 0.9604 | Complex boundaries |

**Note:** Different models may give different predictions for the same patient. This is normal and expected!

---

## Project Structure

```
risk-factor/
├── app/
│   ├── __init__.py
│   ├── config.py              # Settings (model name, MLflow URI)
│   ├── models.py              # Pydantic schemas (29 features)
│   ├── model_loader.py        # MLflow model loading with caching
│   ├── preprocessor.py        # Input validation + feature alignment
│   ├── main.py                # FastAPI app, CORS, error handlers
│   └── routers/
│       ├── __init__.py
│       ├── health.py          # GET /health
│       └── predict.py         # POST /predict (with model selection)
├── mlflow.db                  # MLflow tracking database
├── mlruns/                    # Model artifacts
├── feature_columns.json       # Feature names
├── requirements.txt           # Python dependencies
├── run.py                     # Start the API server
├── predict_example.py         # Example script
├── Dockerfile                 # Docker image recipe
├── docker-compose.yml         # Multi-container orchestration
├── .dockerignore              # Docker build exclusions
└── DOCKER_GUIDE.md            # Docker tutorial
```

---

## Docker Setup

### Prerequisites
1. Install Docker Desktop: https://www.docker.com/products/docker-desktop/
2. Start Docker Desktop
3. Wait for whale icon in system tray

### Start Services
```bash
docker-compose up --build
```

### Stop Services
```bash
docker-compose down
```

### Access Points
- API: http://localhost:8000/docs
- MLflow: http://localhost:5000

See **DOCKER_GUIDE.md** for complete Docker documentation.

---

## Testing

### Run Example Script
```bash
python predict_example.py
```

This will:
1. List all available models
2. Make predictions with default model
3. Make predictions with 4 different models
4. Show comparison

### Manual Testing
```bash
# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/models

# Predict
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"age\":25,\"number_of_sexual_partners\":1,...}"
```

---

## Model Registry (MLflow)

### View Models in MLflow UI
1. Go to http://localhost:5000
2. Click "Models" tab
3. See all 8 registered models with versions and stages

### View Experiments
1. Click "Experiments" tab
2. Select "Cervical_Cancer_Model_Training"
3. See all training runs with metrics

---

## Key Features

✅ **8 ML Models** - All registered in MLflow
✅ **Dynamic Model Selection** - Choose model via query parameter
✅ **Input Validation** - Pydantic validates all 29 features
✅ **Confidence Scores** - Get probability estimates
✅ **Error Handling** - Clear error messages
✅ **CORS Enabled** - Can be called from any frontend
✅ **Docker Ready** - Containerize with one command
✅ **Swagger UI** - Interactive API documentation
✅ **Health Checks** - Monitor service status
✅ **Model Caching** - Fast predictions after first load

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Redirect to Swagger UI |
| `/health` | GET | Service health check |
| `/models` | GET | List all available models |
| `/predict` | POST | Make prediction (optional `?model=Name`) |
| `/docs` | GET | Swagger UI documentation |
| `/redoc` | GET | ReDoc documentation |

---

## Example Use Cases

### 1. Use Best Model (Linear_SVM)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{...patient data...}'
```

### 2. Use Specific Model (AdaBoost)
```bash
curl -X POST "http://localhost:8000/predict?model=AdaBoost" \
  -H "Content-Type: application/json" \
  -d '{...patient data...}'
```

### 3. Compare Multiple Models
```python
models = ["Linear_SVM", "AdaBoost", "Random_Forest", "Gradient_Boosting"]
for model in models:
    result = predict_with_model(model, patient_data)
    print(f"{model}: {result['prediction_label']}")
```

### 4. Batch Predictions
```python
patients = [patient1, patient2, patient3]
for patient in patients:
    result = predict_with_model("Linear_SVM", patient)
    print(f"Prediction: {result['prediction_label']}")
```

---

## Troubleshooting

### Server Not Running
```bash
# Start the server
python run.py
```

### Port Already in Use
Change port in `app/config.py`:
```python
PORT: int = 8001
```

### Model Not Found
```bash
# List available models
curl http://localhost:8000/models

# Use correct model name (case-sensitive)
curl -X POST "http://localhost:8000/predict?model=Linear_SVM" ...
```

### Docker Not Starting
1. Start Docker Desktop from Start Menu
2. Wait for whale icon
3. Try again: `docker-compose up --build`

---

## Next Steps

1. ✅ **Use the API** - Go to http://localhost:8000/docs
2. ✅ **Test predictions** - Try different models
3. ✅ **View MLflow** - Check model registry at http://localhost:5000
4. ✅ **Install Docker** - Containerize when ready
5. ✅ **Deploy** - Push to cloud (AWS, GCP, Azure, etc.)

---

## Support

- **Docker Guide**: See DOCKER_GUIDE.md
- **Example Script**: See predict_example.py
- **API Docs**: http://localhost:8000/docs
- **MLflow UI**: http://localhost:5000

---

## Summary

You now have a **complete, production-ready ML prediction service** with:
- 8 trained models in MLflow registry
- FastAPI REST API with dynamic model selection
- Input validation and error handling
- Docker containerization ready
- Comprehensive documentation

**Start using it now:** http://localhost:8000/docs