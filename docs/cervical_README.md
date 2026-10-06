# 🏥 Cervical Cancer Risk Prediction API

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.139.0-green?style=for-the-badge&logo=fastapi)
![MLflow](https://img.shields.io/badge/MLflow-3.14.0-orange?style=for-the-badge&logo=mlflow)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?style=for-the-badge&logo=docker)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.9.0-yellow?style=for-the-badge&logo=scikit-learn)

**A production-ready MLOps pipeline for cervical cancer risk prediction with 8 machine learning models**

[Features](#features) • [Quick Start](#quick-start) • [API Documentation](#api-documentation) • [Models](#available-models) • [Docker](#docker-deployment) • [Project Structure](#project-structure)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [API Documentation](#api-documentation)
- [Available Models](#available-models)
- [Docker Deployment](#docker-deployment)
- [Project Structure](#project-structure)
- [Examples](#examples)
- [Results](#results)
- [Contributing](#contributing)

---

## 🎯 Overview

This project implements a complete **MLOps pipeline** for cervical cancer risk prediction using machine learning. It includes:

- **8 trained ML models** registered in MLflow Model Registry
- **REST API** with dynamic model selection
- **Interactive Swagger UI** for testing
- **Docker containerization** for easy deployment
- **Comprehensive EDA and feature engineering**
- **Production-ready** with error handling and validation

### 🎓 Use Case

Healthcare professionals can use this API to:
- Assess cervical cancer risk based on patient data
- Get predictions from multiple ML models
- Compare different model predictions
- Make informed decisions about further testing

---

## ✨ Features

### 🤖 Machine Learning
- ✅ **8 Different Models** - Logistic Regression, SVM (Linear/RBF/Polynomial), Bagging, Random Forest, AdaBoost, Gradient Boosting
- ✅ **MLflow Integration** - All models tracked and registered
- ✅ **Model Versioning** - Easy rollback and A/B testing
- ✅ **Dynamic Selection** - Choose model per request via query parameter

### 🔌 API Features
- ✅ **RESTful API** - FastAPI with async support
- ✅ **Input Validation** - Pydantic schemas with 29 features
- ✅ **Confidence Scores** - Probability estimates for predictions
- ✅ **Error Handling** - Clear, actionable error messages
- ✅ **CORS Enabled** - Call from any frontend
- ✅ **Health Checks** - Monitor service status
- ✅ **Model Caching** - Fast predictions after first load

### 📊 Data & MLops
- ✅ **EDA Pipeline** - Comprehensive exploratory analysis
- ✅ **Feature Engineering** - SMOTE, scaling, preprocessing
- ✅ **Model Registry** - 8 models in Production stage
- ✅ **Experiment Tracking** - MLflow for metrics and parameters
- ✅ **Docker Ready** - One-command deployment

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- pip package manager
- (Optional) Docker Desktop for containerization

### Installation

```bash
# Clone the repository
cd "risk factor"

# Install dependencies
pip install -r requirements.txt
```

### Run the API

```bash
# Start the FastAPI server
python run.py
```

**Access the API:**
- 🌐 **Swagger UI**: http://localhost:8000/docs
- 🔍 **Health Check**: http://localhost:8000/health
- 📋 **Models List**: http://localhost:8000/models
- 📊 **MLflow UI**: http://localhost:5000

---

## 📖 API Documentation

### 1. Health Check

**Endpoint:** `GET /health`

**Description:** Check if the service is running and get model information

**Response:**
```json
{
  "status": "ok",
  "model_name": "Linear_SVM",
  "model_version": 1,
  "model_stage": "Production",
  "features_count": 29
}
```

### 2. List Available Models

**Endpoint:** `GET /models`

**Description:** Get all registered models available for prediction

**Response:**
```json
{
  "models": [
    {
      "name": "Linear_SVM",
      "version": 1,
      "stage": "Production",
      "description": "..."
    },
    ...
  ],
  "default_model": "Linear_SVM",
  "total": 9
}
```

### 3. Make Prediction

**Endpoint:** `POST /predict`

**Query Parameters:**
- `model` (optional): Name of the model to use (e.g., `AdaBoost`, `Random_Forest`)

**Request Body:**
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

**Response:**
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

**Response Fields:**
- `prediction`: 0 (Negative/low risk) or 1 (Positive/high risk)
- `prediction_label`: Human-readable prediction
- `confidence`: Probability of being Positive (0-1)
- `model_name`: Which model made the prediction
- `model_version`: Version of the model

---

## 🏆 Available Models

| Model | F1-Score | ROC-AUC | Best For |
|-------|----------|---------|----------|
| **Linear_SVM** ⭐ | 0.8696 | 0.9534 | Best overall (default) |
| **AdaBoost** ⭐ | 0.8696 | 0.9470 | Tied for best |
| Bagging | 0.8571 | 0.9476 | Ensemble method |
| Gradient_Boosting | 0.8571 | 0.9307 | Sequential learning |
| Logistic_Regression | 0.8333 | 0.9545 | Interpretable |
| Polynomial_SVM | 0.8182 | 0.9371 | Non-linear patterns |
| Random_Forest | 0.7826 | 0.9263 | Feature importance |
| RBF_SVM | 0.6000 | 0.9604 | Complex boundaries |

**Note:** Different models may give different predictions for the same patient. This is expected and useful for comparing model perspectives!

---

## 🐳 Docker Deployment

### Prerequisites
- Docker Desktop installed and running

### Start All Services

```bash
# Build and start containers
docker-compose up --build

# Or run in background
docker-compose up -d --build
```

### Access Services
- **FastAPI API**: http://localhost:8000/docs
- **MLflow UI**: http://localhost:5000

### Stop Services

```bash
docker-compose down
```

See [DOCKER_GUIDE.md](DOCKER_GUIDE.md) for complete Docker documentation.

---

## 📁 Project Structure

```
risk-factor/
├── 📊 Data & Models
│   ├── kag_risk_factors_cervical_cancer.csv  # Raw dataset
│   ├── mlflow.db                             # MLflow tracking database
│   ├── mlruns/                               # Model artifacts
│   └── feature_columns.json                  # Feature names
│
├── 🔧 Pipeline Scripts
│   ├── 01_eda.py                            # Exploratory data analysis
│   ├── 02_feature_engineering.py            # Feature engineering pipeline
│   ├── 03_model_training.py                 # Model training & evaluation
│   └── 04_mlflow_model_registry.py          # Model registration
│
├── 🚀 FastAPI Application
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py                        # Settings & configuration
│   │   ├── models.py                        # Pydantic schemas
│   │   ├── model_loader.py                  # MLflow model loading
│   │   ├── preprocessor.py                  # Input preprocessing
│   │   ├── main.py                          # FastAPI app factory
│   │   └── routers/
│   │       ├── health.py                    # GET /health
│   │       └── predict.py                   # POST /predict
│   ├── run.py                               # Start server
│   └── requirements.txt
│
├── 🐳 Docker
│   ├── Dockerfile                           # API container recipe
│   ├── docker-compose.yml                   # Multi-container setup
│   ├── .dockerignore                        # Build exclusions
│   └── DOCKER_GUIDE.md                      # Docker tutorial
│
└── 📝 Documentation
    ├── README.md                            # This file
    ├── README_CERVICAL_CANCER_API.md        # Detailed API guide
    └── predict_example.py                   # Example usage script
```

---

## 💡 Examples

### Python Example

```python
import urllib.request
import json

# Patient data
patient = {
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

# Predict with AdaBoost
url = "http://localhost:8000/predict?model=AdaBoost"
req = urllib.request.Request(
    url,
    data=json.dumps(patient).encode(),
    headers={"Content-Type": "application/json"},
    method="POST"
)

response = urllib.request.urlopen(req)
result = json.loads(response.read().decode())
print(f"Prediction: {result['prediction_label']}")
print(f"Confidence: {result['confidence']:.4f}")
```

### cURL Example

```bash
# Predict with default model
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":25,"number_of_sexual_partners":1,...}'

# Predict with specific model
curl -X POST "http://localhost:8000/predict?model=AdaBoost" \
  -H "Content-Type: application/json" \
  -d '{"age":25,"number_of_sexual_partners":1,...}'
```

---

## 📊 Results

### Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Linear_SVM | 0.9820 | 0.8333 | 0.9091 | **0.8696** | 0.9534 |
| AdaBoost | 0.9820 | 0.8333 | 0.9091 | **0.8696** | 0.9470 |
| Bagging | 0.9820 | 0.9000 | 0.8182 | 0.8571 | 0.9476 |
| Gradient_Boosting | 0.9820 | 0.9000 | 0.8182 | 0.8571 | 0.9307 |

### Live API Test Results

```
PREDICTIONS WITH DIFFERENT MODELS:
Linear_SVM               : Negative   (confidence: 0.000000)
AdaBoost                 : Positive   (confidence: 0.594546)
Random_Forest            : Positive   (confidence: 0.780000)
Gradient_Boosting        : Positive   (confidence: 0.990534)
```

**Observation:** Different models give different predictions, demonstrating the value of model selection!

---

## 🛠️ Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| **API Framework** | FastAPI | 0.139.0 |
| **ML Framework** | Scikit-learn | 1.9.0 |
| **Model Registry** | MLflow | 3.14.0 |
| **Data Processing** | Pandas, NumPy | 2.3.3, 2.3.3 |
| **API Server** | Uvicorn | 0.51.0 |
| **Validation** | Pydantic | 2.13.4 |
| **Containerization** | Docker | 29.6.1 |
| **Imbalance Handling** | SMOTE (imbalanced-learn) | Latest |

---

## 🎓 Key Features Explained

### 1. Dynamic Model Selection
Choose any model per request using query parameters:
```bash
POST /predict?model=AdaBoost
POST /predict?model=Random_Forest
POST /predict?model=Gradient_Boosting
```

### 2. Input Validation
All 29 features are validated using Pydantic:
- Type checking (int, float)
- Range validation (e.g., age: 10-100)
- Required field enforcement
- Extra field rejection

### 3. Confidence Scores
Get probability estimates when available:
```json
{
  "prediction": 1,
  "prediction_label": "Positive",
  "confidence": 0.85
}
```

### 4. Model Caching
Models are loaded once and cached for fast predictions:
- First request: ~2-3 seconds (loads model)
- Subsequent requests: ~50-100ms (cached)

---

## 📈 Pipeline Workflow

```
1. EDA (01_eda.py)
   ↓ Load data, analyze distributions, correlations, outliers
   
2. Feature Engineering (02_feature_engineering.py)
   ↓ Clean data, handle missing values, scale features, apply SMOTE
   
3. Model Training (03_model_training.py)
   ↓ Train 8 models, track with MLflow, evaluate performance
   
4. Model Registration (04_mlflow_model_registry.py)
   ↓ Register models in MLflow, set stages, add descriptions
   
5. API Deployment (app/)
   ↓ FastAPI service with dynamic model selection
   
6. Docker Containerization
   ↓ Package everything into containers
```

---

## 🧪 Testing

### Run Example Script
```bash
python predict_example.py
```

This will:
1. List all available models
2. Make predictions with default model
3. Compare predictions across 4 different models
4. Show confidence scores

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

## 🚢 Deployment

### Local Deployment
```bash
python run.py
```

### Docker Deployment
```bash
docker-compose up --build
```

### Cloud Deployment
Ready for deployment on:
- AWS ECS / EKS
- Google Cloud Run
- Azure Container Instances
- Kubernetes
- Heroku

---

## 📝 Documentation

- **[README_CERVICAL_CANCER_API.md](README_CERVICAL_CANCER_API.md)** - Detailed API guide
- **[DOCKER_GUIDE.md](DOCKER_GUIDE.md)** - Complete Docker tutorial
- **[predict_example.py](predict_example.py)** - Working examples
- **Swagger UI** - Interactive docs at http://localhost:8000/docs

---

## 🎯 Next Steps

1. ✅ **Use the API** - Go to http://localhost:8000/docs
2. ✅ **Test predictions** - Try different models
3. ✅ **View MLflow** - Check model registry at http://localhost:5000
4. ✅ **Deploy with Docker** - Run `docker-compose up --build`
5. ✅ **Integrate with frontend** - Call API from web/mobile app
6. ✅ **Monitor in production** - Add logging and monitoring

---

## 👨‍💻 Author

**ML Pipeline** - Built with ❤️ for healthcare AI

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- Dataset: Kaggle - Cervical Cancer Risk Factors
- MLflow for model tracking and registry
- FastAPI for the amazing web framework
- Scikit-learn for machine learning tools

---

<div align="center">

**⭐ Star this repo if you find it helpful!**

Made with ❤️ for better healthcare through AI

</div>