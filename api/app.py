import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import shap

# ── App setup ─────────────────────────────────────────────
app = FastAPI(
    title="Customer Churn Classifier API",
    description="Predicts customer churn with SHAP explainability",
    version="1.0.0"
)

# ── Load model and features ───────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
model = joblib.load(os.path.join(BASE_DIR, "models/churn_best_model.pkl"))
feature_names = joblib.load(os.path.join(BASE_DIR, "models/feature_names.pkl"))

# ── Request schema ────────────────────────────────────────
class CustomerData(BaseModel):
    tenure: int = Field(..., ge=0, le=72, description="Months with company")
    monthly_charges: float = Field(..., ge=0, description="Monthly bill amount")
    total_charges: float = Field(..., ge=0, description="Total amount charged")
    senior_citizen: int = Field(0, ge=0, le=1, description="1 if senior citizen")
    partner: int = Field(0, ge=0, le=1, description="1 if has partner")
    dependents: int = Field(0, ge=0, le=1, description="1 if has dependents")
    phone_service: int = Field(1, ge=0, le=1, description="1 if has phone service")
    multiple_lines: int = Field(0, ge=0, le=1, description="1 if has multiple lines")
    online_security: int = Field(0, ge=0, le=1, description="1 if has online security")
    online_backup: int = Field(0, ge=0, le=1, description="1 if has online backup")
    device_protection: int = Field(0, ge=0, le=1, description="1 if has device protection")
    tech_support: int = Field(0, ge=0, le=1, description="1 if has tech support")
    streaming_tv: int = Field(0, ge=0, le=1, description="1 if has streaming TV")
    streaming_movies: int = Field(0, ge=0, le=1, description="1 if has streaming movies")
    paperless_billing: int = Field(0, ge=0, le=1, description="1 if paperless billing")
    gender: int = Field(0, ge=0, le=1, description="1 if male")
    internet_fiber: int = Field(0, ge=0, le=1, description="1 if fiber optic internet")
    internet_no: int = Field(0, ge=0, le=1, description="1 if no internet")
    contract_one_year: int = Field(0, ge=0, le=1, description="1 if one year contract")
    contract_two_year: int = Field(0, ge=0, le=1, description="1 if two year contract")
    payment_credit_card: int = Field(0, ge=0, le=1, description="1 if credit card payment")
    payment_electronic: int = Field(0, ge=0, le=1, description="1 if electronic check")
    payment_mailed: int = Field(0, ge=0, le=1, description="1 if mailed check")

    class Config:
        json_schema_extra = {
            "example": {
                "tenure": 2,
                "monthly_charges": 95.0,
                "total_charges": 190.0,
                "senior_citizen": 0,
                "partner": 0,
                "dependents": 0,
                "phone_service": 1,
                "multiple_lines": 0,
                "online_security": 0,
                "online_backup": 0,
                "device_protection": 0,
                "tech_support": 0,
                "streaming_tv": 0,
                "streaming_movies": 0,
                "paperless_billing": 1,
                "gender": 1,
                "internet_fiber": 1,
                "internet_no": 0,
                "contract_one_year": 0,
                "contract_two_year": 0,
                "payment_credit_card": 0,
                "payment_electronic": 1,
                "payment_mailed": 0
            }
        }


def build_input(data: CustomerData) -> pd.DataFrame:
    charges_per_tenure = data.total_charges / (data.tenure + 1)
    is_new_customer = 1 if data.tenure <= 6 else 0

    row = {
        "gender": data.gender,
        "SeniorCitizen": data.senior_citizen,
        "Partner": data.partner,
        "Dependents": data.dependents,
        "tenure": data.tenure,
        "PhoneService": data.phone_service,
        "MultipleLines": data.multiple_lines,
        "OnlineSecurity": data.online_security,
        "OnlineBackup": data.online_backup,
        "DeviceProtection": data.device_protection,
        "TechSupport": data.tech_support,
        "StreamingTV": data.streaming_tv,
        "StreamingMovies": data.streaming_movies,
        "PaperlessBilling": data.paperless_billing,
        "MonthlyCharges": data.monthly_charges,
        "TotalCharges": data.total_charges,
        "InternetService_Fiber optic": data.internet_fiber,
        "InternetService_No": data.internet_no,
        "Contract_One year": data.contract_one_year,
        "Contract_Two year": data.contract_two_year,
        "PaymentMethod_Credit card (automatic)": data.payment_credit_card,
        "PaymentMethod_Electronic check": data.payment_electronic,
        "PaymentMethod_Mailed check": data.payment_mailed,
        "ChargesPerTenure": charges_per_tenure,
        "IsNewCustomer": is_new_customer
    }
    return pd.DataFrame([row], columns=feature_names)


# ── Routes ─────────────────────────────────────────────────
@app.get("/")
def home():
    return {
        "message": "Customer Churn Classifier API",
        "endpoints": {
            "health":   "GET  /health",
            "predict":  "POST /predict",
            "explain":  "POST /explain",
            "docs":     "GET  /docs"
        }
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "features": len(feature_names)
    }


@app.post("/predict")
def predict(data: CustomerData):
    try:
        input_df = build_input(data)
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

        churn_prob = round(float(proba[1]), 4)
        risk_level = (
            "HIGH RISK 🔴" if churn_prob >= 0.7 else
            "MEDIUM RISK 🟡" if churn_prob >= 0.4 else
            "LOW RISK 🟢"
        )

        return {
            "churn_prediction": bool(pred),
            "result": "Will Churn" if pred == 1 else "Will Not Churn",
            "churn_probability": churn_prob,
            "retention_probability": round(float(proba[0]), 4),
            "risk_level": risk_level,
            "input_summary": {
                "tenure_months": data.tenure,
                "monthly_charges": data.monthly_charges,
                "contract_type": (
                    "Two Year" if data.contract_two_year else
                    "One Year" if data.contract_one_year else
                    "Month-to-Month"
                )
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain")
@app.post("/explain")
def explain(data: CustomerData):
    try:
        input_df = build_input(data)

        # get preprocessor and classifier separately
        preprocessor = model.named_steps["preprocessor"]
        classifier = model.named_steps["classifier"]

        # transform input
        input_processed = preprocessor.transform(input_df)

        # get model coefficients directly for Logistic Regression
        coefficients = classifier.coef_[0]
        
        # multiply coefficients by scaled feature values
        # this gives us the contribution of each feature
        contributions = coefficients * input_processed[0]

        # build explanation dict
        shap_dict = {
            feature_names[i]: round(float(contributions[i]), 4)
            for i in range(len(feature_names))
        }

        # top 5 factors pushing toward churn (positive values)
        sorted_desc = sorted(shap_dict.items(), key=lambda x: x[1], reverse=True)
        top_churn_factors = [
            {"feature": k, "contribution": v, "impact": "increases churn risk"}
            for k, v in sorted_desc[:5] if v > 0
        ]

        # top 5 factors pushing against churn (negative values)
        sorted_asc = sorted(shap_dict.items(), key=lambda x: x[1])
        top_retention_factors = [
            {"feature": k, "contribution": v, "impact": "decreases churn risk"}
            for k, v in sorted_asc[:5] if v < 0
        ]

        return {
            "explanation": "Feature contributions show how each feature pushes prediction toward or away from churn",
            "churn_probability": round(float(model.predict_proba(input_df)[0][1]), 4),
            "top_churn_factors": top_churn_factors,
            "top_retention_factors": top_retention_factors,
            "all_contributions": shap_dict
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))