import pandas as pd
import os

os.makedirs("data/raw", exist_ok=True)

# Telco Customer Churn dataset from GitHub
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

df = pd.read_csv(url)

print("=== TELCO CUSTOMER CHURN DATASET ===")
print(f"Shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nChurn distribution:\n{df['Churn'].value_counts()}")
print(f"\nChurn percentage:\n{df['Churn'].value_counts(normalize=True)*100}")
print(f"\nMissing values:\n{df.isnull().sum()}")

df.to_csv("data/raw/churn.csv", index=False)
print("\n✅ Dataset saved to data/raw/churn.csv")