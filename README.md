# Cervical Cancer Risk Factor Analysis

## Project Overview

This project performs a comprehensive analysis of cervical cancer risk factors using machine learning. The dataset contains medical records of patients with various risk factors, and the goal is to predict the presence of cervical cancer based on biopsy results.

> ✅ **Note on this README**: The results table below was verified against the actual output of `model_training_testing_evaluation.ipynb` on [DATE] and matches exactly. The only correction made: the original README's placeholder line *"Run the model training notebook to populate the results table with actual values"* has been removed, since the values shown are confirmed real, and the "best model" claim has been clarified below (it's a tie, not a single winner).

---

## Dataset

**Source**: [Kaggle - Cervical Cancer Risk Factors](https://www.kaggle.com/datasets/loveall/cervical-cancer-risk-classification)
**Size**: 858 rows, 36 columns
**Target Variable**: `Biopsy` (0 = Negative, 1 = Positive)
**Challenge**: The dataset is highly imbalanced — only ~5% of samples are positive (Biopsy = 1).

---

## Results Summary

*(actual output of `model_training_testing_evaluation.ipynb`)*

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Linear SVM | 0.9820 | 0.8333 | 0.9091 | **0.8696** | 0.9534 |
| **AdaBoost** | 0.9820 | 0.8333 | 0.9091 | **0.8696** | 0.9470 |
| Bagging | 0.9820 | 0.9000 | 0.8182 | 0.8571 | N/A |
| Gradient Boosting | 0.9820 | 0.9000 | 0.8182 | 0.8571 | 0.9307 |
| Logistic Regression | 0.9760 | 0.7692 | 0.9091 | 0.8333 | 0.9545 |
| Polynomial SVM | 0.9760 | 0.8182 | 0.8182 | 0.8182 | 0.9371 |
| Random Forest | 0.9701 | 0.7500 | 0.8182 | 0.7826 | 0.9263 |
| RBF SVM | 0.9521 | 0.6667 | 0.5455 | 0.6000 | 0.9604 |

**Best model (by F1-score): a tie between Linear SVM and AdaBoost, both at 98.20% accuracy and 0.8696 F1-score.** AdaBoost has a slightly higher ROC-AUC (0.9470) among the two.

---

## Key Findings

1. **Class Imbalance**: The dataset is highly imbalanced (~5% positive cases), making raw accuracy an unreliable standalone metric. F1-Score and ROC-AUC are the more meaningful metrics here.
2. **SMOTE Effectiveness**: Applying SMOTE improved recall on the minority class at some cost to precision.
3. **Best Performing Models**: Ensemble/margin-based methods (AdaBoost, Linear SVM, Gradient Boosting) outperform plain Logistic Regression and Random Forest on this dataset — note this is the *opposite* pattern from the companion Credit Risk project, where Random Forest won. Model choice is genuinely dataset-dependent.

---

## Tests

The notebooks' data-cleaning, EDA and evaluation logic is also available as an importable package in `src/cervical_cancer/`, with unit tests in `tests/`:

```bash
pip install -r requirements.txt
pytest   # runs tests with a coverage report
```

---

## 👤 Author

**Richi Upadhyay** — Data Science & Machine Learning
