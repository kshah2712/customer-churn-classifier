import pandas as pd
import numpy as np
import os
import sys
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE


def load_data(path="data/raw/churn.csv"):
    df = pd.read_csv(path)
    print("=== LOADING DATASET ===")
    print(f"Shape: {df.shape}")
    print(f"\nChurn distribution:\n{df['Churn'].value_counts()}")
    return df


def clean_data(df):
    df = df.copy()

    # drop customerID — not useful for prediction
    df = df.drop("customerID", axis=1)

    # TotalCharges has spaces instead of NaN — fix it
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # fill missing TotalCharges with median
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # convert target to binary
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    print("\n=== AFTER CLEANING ===")
    print(f"Missing values:\n{df.isnull().sum().sum()} total")
    return df


def encode_features(df):
    df = df.copy()

    # binary columns — Yes/No → 1/0
    binary_cols = [
        "gender", "Partner", "Dependents", "PhoneService",
        "PaperlessBilling", "MultipleLines", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies"
    ]

    for col in binary_cols:
        if col == "gender":
            df[col] = df[col].map({"Male": 1, "Female": 0})
        else:
            df[col] = df[col].map({"Yes": 1, "No": 0, "No phone service": 0, "No internet service": 0})

    # multi-class columns — one hot encoding
    df = pd.get_dummies(df, columns=["InternetService", "Contract", "PaymentMethod"], drop_first=True)

    print("\n=== AFTER ENCODING ===")
    print(f"Total features: {df.shape[1] - 1}")
    print(f"Columns: {list(df.columns)}")
    return df


def create_features(df):
    df = df.copy()

    # avg monthly spend per tenure month
    df["ChargesPerTenure"] = df["TotalCharges"] / (df["tenure"] + 1)

    # is new customer?
    df["IsNewCustomer"] = (df["tenure"] <= 6).astype(int)

    print("\n=== FEATURE ENGINEERING ===")
    print(f"Added 2 new features: ChargesPerTenure, IsNewCustomer")
    return df


def preprocess(path="data/raw/churn.csv", use_smote=True):
    # load
    df = load_data(path)

    # clean
    df = clean_data(df)

    # encode
    df = encode_features(df)

    # feature engineering
    df = create_features(df)

    # separate features and target
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # split BEFORE applying SMOTE
    # important — never apply SMOTE on test data!
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n=== TRAIN/TEST SPLIT ===")
    print(f"Train: {X_train.shape[0]} samples")
    print(f"Test: {X_test.shape[0]} samples")
    print(f"Train churn rate: {y_train.mean()*100:.1f}%")

    # apply SMOTE only on training data
    if use_smote:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"\n=== AFTER SMOTE ===")
        print(f"Train samples after SMOTE: {X_train.shape[0]}")
        print(f"Churn distribution: {pd.Series(y_train).value_counts().to_dict()}")

    # save feature names
    os.makedirs("models", exist_ok=True)
    joblib.dump(list(X.columns), "models/feature_names.pkl")

    return X_train, X_test, y_train, y_test


def get_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])


if __name__ == "__main__":
    preprocess()