from pathlib import Path

import pandas as pd

# Load the dataset
print("=" * 60)
print("DATASET INSPECTION")
print("=" * 60)

project_root = Path(__file__).resolve().parents[1]
csv_filename = "diabetes_prediction_dataset_200_records.csv"
csv_path = project_root / "dataset" / csv_filename
df = pd.read_csv(csv_path)

print("\n1. Dataset Shape")
print(df.shape)

print("\n2. Column Names")
print(df.columns.tolist())

print("\n3. First 5 Records")
print(df.head())

print("\n4. Data Types")
print(df.dtypes)

print("\n5. Missing Values in Each Column")
print(df.isnull().sum())

print("\n6. Total Number of Duplicate Records")
print(df.duplicated().sum())

print("\n7. Diabetes Outcome Distribution")
print(df["Diabetes_Outcome"].value_counts())

print("\nDATASET INSPECTION COMPLETED SUCCESSFULLY")
