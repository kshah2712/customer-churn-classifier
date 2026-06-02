import joblib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from preprocess import preprocess, get_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_auc_score, confusion_matrix, f1_score
)
import numpy as np
import pandas as pd


def evaluate_model(y_test, preds, proba, model_name):
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    roc = roc_auc_score(y_test, proba)

    print(f"\n--- {model_name} ---")
    print(f"Accuracy:  {acc:.4f} ({acc*100:.1f}%)")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, preds, target_names=['No Churn', 'Churn'])}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, preds)}")

    return {
        "accuracy": round(acc, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc, 4)
    }


def train_and_evaluate():
    print("="*50)
    print("  Training models on Telco Churn dataset")
    print("="*50)

    X_train, X_test, y_train, y_test = preprocess()

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=100,
            random_state=42,
            eval_metric="logloss",
            scale_pos_weight=3
        )
    }

    results = {}
    best_roc = 0
    best_model = None
    best_name = ""

    for name, clf in models.items():
        pipeline = Pipeline([
            ("preprocessor", get_pipeline()),
            ("classifier", clf)
        ])

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        proba = pipeline.predict_proba(X_test)[:, 1]

        metrics = evaluate_model(y_test, preds, proba, name)
        results[name] = metrics

        if metrics["roc_auc"] > best_roc:
            best_roc = metrics["roc_auc"]
            best_model = pipeline
            best_name = name

    # save best model
    os.makedirs("models", exist_ok=True)
    model_path = "models/churn_best_model.pkl"
    joblib.dump(best_model, model_path)

    print(f"\n✅ Best model: {best_name} (ROC-AUC: {best_roc:.4f})")
    print(f"✅ Saved to: {model_path}")

    results["best_model"] = best_name
    results["best_roc_auc"] = round(best_roc, 4)

    with open("models/results.json", "w") as f:
        json.dump(results, f, indent=2)

    return best_model


if __name__ == "__main__":
    train_and_evaluate()