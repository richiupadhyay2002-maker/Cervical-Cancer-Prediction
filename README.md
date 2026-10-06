# 🏥 Cervical Cancer Risk Prediction - MLOps Pipeline

![Python](https://img.shields.io/badge/Python-3.13%2B-blue)
![MLflow](https://img.shields.io/badge/MLflow-2.9%2B-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-green)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-red)

An end-to-end MLOps project for cervical cancer risk prediction with 9 supervised classification algorithms, leakage-safe hyperparameter tuning, and a FastAPI API.

> **Research use only:** This project is for educational and research purposes. It is not a medical device and must not be used to diagnose, screen, or guide clinical decisions.

---

## 📁 Project Structure

```
Cervical-Cancer-Prediction/
│
├── 📓 notebooks/                      # Jupyter notebooks
│   ├── cervical_eda.ipynb             # Exploratory Data Analysis
│   ├── cervical_02_feature_engineering_advanced.ipynb  # Feature Engineering (70/15/15 split)
│   ├── cervical_03_model_training_advanced.ipynb  # 9 models, GridSearchCV + MLflow tracking
│   ├── cervical_04_mlflow_model_registry.ipynb  # MLflow Model Registry (versioning + stages)
│   ├── cervical_04_model_evaluation.ipynb  # Comprehensive evaluation + comparison
│   └── [API-related notebooks: api_01 through api_08]
│
├── 📊 data/                           # All data files
│   ├── raw/                           # Original dataset
│   │   └── kag_risk_factors_cervical_cancer.csv
│
├── 📂 processed_data/                 # Generated locally by the pipeline (not committed)
│
├── 🤖 models/                         # Trained model files
│   ├── *_tuned.pkl                    # 9 tuned model files
│   └── model_comparison_advanced.csv  # Model comparison report
│
├── 📦 notebooks/mlflow.db, mlruns/    # Generated locally; ignored by Git
│
├── 🚀 api/                            # FastAPI application
│   ├── app/                           # API source code
│   │   ├── main.py                    # FastAPI application entry point
│   │   ├── config.py                  # Configuration
│   │   ├── model_loader.py            # Dynamic model loading (MLflow, caching, registry listing)
│   │   ├── models.py                  # Pydantic schemas (35-field PredictionInput)
│   │   ├── preprocessor.py            # Input preprocessing (np.nan defaults, field normalization)
│   │   ├── report_generator.py        # Clinical report generation
│   │   ├── templates/                 # Report templates
│   │   │   └── risk_report_template.md
│   │   └── routers/                   # API endpoints
│   │       ├── health.py              # Health check endpoint
│   │       ├── predict.py             # Prediction endpoint
│   │       └── report.py              # Clinical report endpoint
│   ├── run.py                         # Start the API server
│   ├── requirements.txt               # Python dependencies
│   └── predict_example.py             # Example prediction script
│
├── 📝 docs/                           # Documentation
│   ├── cervical_README.md             # This file (root README.md also exists)
│   ├── cervical_PROJECT_SUMMARY.md    # Project overview
│   ├── README_CERVICAL_CANCER_API.md  # API documentation
│   ├── cervical_DOCKER_GUIDE.md       # Docker deployment guide
│   └── cervical_04_model_evaluation_report.md  # Detailed evaluation report
│
├── 🐳 docker/                         # Docker files
│   ├── cervical_Dockerfile            # Container recipe
│   ├── cervical_docker-compose.yml    # Multi-container setup
│   └── cervical_.dockerignore         # Build exclusions
│
├── 📈 outputs/                        # Visualizations and reports
│   └── evaluation_outputs/            # Generated evaluation outputs
│
└── 📋 config/                         # Configuration files
```

---

## 🔄 Pipeline Workflow

```
cervical_eda.ipynb ──→ cervical_02_feature_engineering_advanced.ipynb ──→ cervical_03_model_training_advanced.ipynb ──→ cervical_04_mlflow_model_registry.ipynb ──→ cervical_04_model_evaluation.ipynb
     │                        │                                         │                          │                                        │
     ▼                        ▼                                         ▼                          ▼                                        ▼
  EDA &            70/15/15 Stratified Split                         9 Models + GridSearchCV       MLflow Registry                    Final Evaluation &
Visualizations        StandardScaler, SMOTE (CV-fold)                  MLflow Run Logging          Versioning + Stage Promotion    Best Model Selection
```

### Data Split
| Dataset | Size | Percentage | Usage |
|---------|------|------------|-------|
| **Training** | 600 samples | 70% | Model training, leakage-safe SMOTE (in CV folds) |
| **Validation** | 129 samples | 15% | Hyperparameter tuning + model selection |
| **Test** | 129 samples | 15% | Final evaluation only |

---

## 🚀 Quick Start

### 1. Run the Pipeline

```bash
# From the repository root
cd Cervical-Cancer-Prediction

# Execute notebooks in order
jupyter notebook notebooks/cervical_eda.ipynb
jupyter notebook notebooks/cervical_02_feature_engineering_advanced.ipynb
jupyter notebook notebooks/cervical_03_model_training_advanced.ipynb
jupyter notebook notebooks/cervical_04_mlflow_model_registry.ipynb
jupyter notebook notebooks/cervical_04_model_evaluation.ipynb
```

### 2. Start the API

```bash
# From the repository root
cd api

# Install dependencies
pip install -r requirements.txt

# Start the server
python run.py
```

### 3. View MLflow UI

```bash
# From the API folder, navigate to notebooks (where mlflow.db is generated)
cd ../notebooks

# Start MLflow UI after running the model registry notebook
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

### 4. Access Services

- **FastAPI Docs**: http://localhost:8000/docs
- **MLflow UI**: http://localhost:5000

---

## 🎯 Model Performance

### All 9 Models (Ranked by Validation F1-Score, Overfitting-Gap Check)

| Model | Train F1 | Val F1 | Test F1 | Train–Val Gap | Verdict |
|-------|----------|--------|---------|---------------|---------|
| **Gradient_Boosting** | 0.7640 | 0.7619 | 0.6316 | 0.0021 | ✅ Recommended |
| **Hist_Gradient_Boosting** | 0.7640 | 0.7273 | 0.6316 | 0.0368 | ✅ Recommended |
| **SGD_ElasticNet** | 0.6667 | 0.6957 | 0.6000 | −0.0290 | Within gap (underfit-leaning) |
| **Logistic_ElasticNet** | 0.7579 | 0.6957 | 0.6316 | 0.0622 | ⚠️ Caution |
| **Linear_SVM** | 0.7579 | 0.6957 | 0.6316 | 0.0622 | ⚠️ Caution |
| **Logistic_Regression** | 0.7660 | 0.6957 | 0.6316 | 0.0703 | ⚠️ Caution |
| **LDA_Shrinkage** | 0.7660 | 0.6957 | 0.6316 | 0.0703 | ⚠️ Caution |
| **RBF_SVM** | 0.8000 | 0.6957 | 0.6667 | 0.1043 | ⚠️ Highest test F1, most overfit |
| **Random_Forest** | 0.8140 | 0.6316 | 0.6316 | 0.1824 | ❌ Most overfit |

### Best Model: Gradient_Boosting
- **Test F1-Score**: 0.6316
- **Test ROC-AUC**: 0.8543
- **Validation F1-Score**: 0.7619
- **Overfitting Gap**: 0.0021 (minimal)
- **Test Recall**: 0.75 (6 of 8 true positive cases identified)
- **Status**: Recommended for further research evaluation (selected using validation-based model selection and an overfitting-gap check)

---

## 🐳 Docker Deployment

```bash
# From the repository root
cd docker

# Build and start containers after running the model pipeline
docker-compose -f cervical_docker-compose.yml up --build
```

---

## 📊 Key Features

### ✅ Leakage-Safe Pipeline & Data Split
- **70/15/15 stratified split** (600 train / 129 val / 129 test)
- **SMOTE embedded in cross-validation folds** (not applied to the full dataset)
- **Validation set** for hyperparameter tuning and model selection
- **Test set** for final evaluation only

### ✅ Hyperparameter Tuning
- GridSearchCV with 5×3 repeated stratified cross-validation (15 folds)
- All 9 algorithms tuned independently
- Best parameters selected based on validation F1 and train–validation gap (overfitting check)

### ✅ Overfitting Detection
- Compares Train vs Test F1-Score
- Threshold: >0.05 indicates potential overfitting
- Detailed analysis for each model

### ✅ Comprehensive Evaluation
- Models: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Visualizations: F1 comparison, overfitting analysis, confusion matrices
- Final recommendations for further research evaluation

### ✅ MLflow Model Registry
- **Experiment**: `Cervical_Cancer_Model_Training_Leakage_Safe`
- **Backend**: SQLite (`mlflow.db`) for tracking and model registry
- **9 models registered** via `mlflow.register_model()` with version history
- **Stage management**: None → Staging → Production (via `client.transition_model_version_stage()`)
- **Production models**: Gradient_Boosting (recommended) and RBF_SVM
- **Registry-based loading**: `models:/Gradient_Boosting/Production`
- **Dynamic model listing**: `MlflowClient().search_registered_models()`
- **MLflow UI**: http://localhost:5000

---

## 📁 File Naming Convention

| Prefix | Purpose |
|--------|---------|
| `01_` | Exploratory Data Analysis |
| `02_` | Feature Engineering |
| `03_` | Model Training & Selection |
| `04_` | Model Evaluation & Registry |

---

## 🛠️ Technologies Used

- **Python 3.13**: Core programming language
- **Scikit-learn**: Machine learning models
- **MLflow**: Experiment tracking and model registry
- **FastAPI**: REST API
- **Docker**: Containerization
- **Pandas/NumPy**: Data processing
- **Matplotlib/Seaborn**: Visualizations
- **SMOTE**: Class imbalance handling

---

## 📧 Contact

For questions or issues, refer to:
- `docs/README_CERVICAL_CANCER_API.md` - API documentation
- `docs/cervical_DOCKER_GUIDE.md` - Docker deployment guide
- `docs/cervical_04_model_evaluation_report.md` - Detailed evaluation report