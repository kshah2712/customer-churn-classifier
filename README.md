# 📉 Customer Churn Classifier

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-009688?logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?logo=scikit-learn)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-FF6B6B)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

A production-style binary classification pipeline that predicts customer churn using the Telco Customer Churn dataset. Covers imbalanced data handling, model explainability with SHAP, and a modern FastAPI REST API — all containerized with Docker.

> **Purpose:** Demonstrate real-world ML challenges — imbalanced classes, model explainability, and building a production-grade REST API with FastAPI (the modern replacement for Flask).

---

## 📌 What This Project Covers

| Concept | Implementation |
|---|---|
| Imbalanced Data | SMOTE oversampling + class weights |
| Feature Engineering | Encoding, scaling, derived features |
| Model Training | Logistic Regression, Random Forest, XGBoost |
| Model Explainability | SHAP values — why did model predict churn? |
| Evaluation Metrics | Precision, Recall, F1, ROC-AUC, Confusion Matrix |
| REST API | FastAPI with `/predict` and `/explain` endpoints |
| Containerization | Dockerfile + docker-compose |
| Testing | pytest with FastAPI test client |

---

## 🗂️ Project Structure

```
customer-churn-classifier/
├── data/
│   ├── raw/                  # Original CSV
│   └── processed/            # Cleaned data
├── notebooks/
│   └── 01_churn_eda.ipynb    # EDA and analysis
├── src/
│   ├── download_data.py      # Download Telco churn dataset
│   ├── preprocess.py         # Cleaning + feature engineering
│   └── train.py              # Model training + evaluation
├── models/                   # Saved .pkl model files
├── api/
│   └── app.py                # FastAPI REST API
├── tests/
│   └── test_api.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/kshah2712/customer-churn-classifier.git
cd customer-churn-classifier
```

### 2. Create virtual environment

```bash
conda create -n churn-classifier python=3.11 -y
conda activate churn-classifier
pip install -r requirements.txt
```

### 3. Download dataset

```bash
python src/download_data.py
```

### 4. Train the models

```bash
python src/train.py
```

### 5. Run FastAPI locally

```bash
uvicorn api.app:app --reload --port 8000
```

### 6. Run with Docker

```bash
docker-compose up --build
```

---

## 🎯 Dataset — Telco Customer Churn

Real telecom customer data. Each row = one customer.

| Feature | Description |
|---|---|
| `tenure` | Months customer has stayed |
| `MonthlyCharges` | Monthly bill amount |
| `TotalCharges` | Total amount charged |
| `Contract` | Month-to-month, One year, Two year |
| `InternetService` | DSL, Fiber optic, No |
| `PaymentMethod` | Payment type |
| `Churn` | **TARGET** — Yes/No |

---

## 🚀 API Usage

### Health check
```bash
GET /health
```

### Predict churn
```bash
POST /predict
{
  "tenure": 12,
  "monthly_charges": 65.5,
  "total_charges": 786.0,
  "contract": "Month-to-month",
  "internet_service": "Fiber optic",
  "payment_method": "Electronic check"
}
```

### Explain prediction
```bash
POST /explain
```
Returns SHAP values showing which features caused the churn prediction.

---

## 📊 Model Results

| Model | Accuracy | ROC-AUC | F1 (Churn) |
|---|---|---|---|
| Logistic Regression | ~80% | ~0.84 | ~0.60 |
| Random Forest | ~82% | ~0.86 | ~0.62 |
| XGBoost | ~82% | ~0.87 | ~0.63 |

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 🛠️ Tech Stack

- **Language:** Python 3.11
- **ML:** Scikit-learn, XGBoost, imbalanced-learn
- **Explainability:** SHAP
- **API:** FastAPI + Uvicorn
- **Containerization:** Docker, Docker Compose
- **Testing:** pytest

---

## 📚 Key Learnings

- Handling **imbalanced datasets** with SMOTE and class weights
- **SHAP values** — making black-box models explainable
- **ROC-AUC** — better metric than accuracy for imbalanced data
- **FastAPI** vs Flask — modern, faster, auto-generates docs
- **Precision-Recall tradeoff** — when to optimize which metric

---

## 🗺️ Part of ML Learning Roadmap

This is **Project 3 of 10** in a progressive ML + GenAI portfolio:

| # | Project | Skills |
|---|---|---|
| ✅ 1 | Classic ML Pipeline | EDA, Sklearn, Flask, Docker |
| ✅ 2 | House Price Predictor | Regression, Feature Eng., Streamlit |
| ✅ 3 | Customer Churn Classifier (this project) | SHAP, FastAPI, Imbalanced data |
| 4 | Image Classifier | PyTorch, CNN, MLflow |
| 5 | Sentiment Analyzer | HuggingFace, BERT, NLP |
| ... | ... | ... |

---

## 👤 Author

**Kashyap Shah**
[GitHub](https://github.com/kshah2712) ·

