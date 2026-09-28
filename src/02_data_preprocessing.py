from pathlib import Path

import pandas as pd

print("=" * 60)
print("DATA PREPROCESSING")
print("=" * 60)

project_root = Path(__file__).resolve().parents[1]
csv_name = "diabetes_prediction_dataset_200_records.csv"
csv_path = project_root / "dataset" / csv_name
df = pd.read_csv(csv_path)

print("\nOriginal dataset shape:", df.shape)
print("\nMissing values before preprocessing:")
print(df.isnull().sum())

# Remove identifier column
if "Patient_ID" in df.columns:
    df = df.drop(columns=["Patient_ID"])
print("\nShape after removing Patient_ID:", df.shape)

# Remove duplicate records
duplicates_removed = df.duplicated().sum()
df = df.drop_duplicates().reset_index(drop=True)
print("\nNumber of duplicate records removed:", duplicates_removed)

# Remove rows with impossible non-positive health measurements
positive_columns = ["Age", "BMI", "Blood_Glucose_Level", "HbA1c_Level"]
valid_values = (df[positive_columns] > 0).all(axis=1)
df = df[valid_values].reset_index(drop=True)

# Separate target and features
target_column = "Diabetes_Outcome"
X = df.drop(columns=[target_column])
y = df[target_column]

# Identify numerical and categorical columns
numerical_columns = X.select_dtypes(include="number").columns.tolist()
categorical_columns = X.select_dtypes(exclude="number").columns.tolist()

# Fill missing values
for column in numerical_columns:
    X[column] = X[column].fillna(X[column].median())

for column in categorical_columns:
    X[column] = X[column].fillna(X[column].mode()[0])

# Encode categorical variables
X = pd.get_dummies(X, columns=categorical_columns, dtype=int)

# Keep processed data together
# Keep Diabetes_Outcome as target without leakage into features
# Save the processed full dataset
processed_path = project_root / "outputs" / "preprocessed_data.csv"
processed_path.parent.mkdir(parents=True, exist_ok=True)
df_processed = pd.concat([X, y], axis=1)
df_processed.to_csv(processed_path, index=False)

print("\nMissing values after preprocessing:")
print(df_processed.isnull().sum())
print("\nNumerical columns:")
print(numerical_columns)
print("\nCategorical columns:")
print(categorical_columns)
print("\nFinal feature names:")
print(X.columns.tolist())
print("\nFinal X shape:", X.shape)
print("Final y shape:", y.shape)
print("\nDATA PREPROCESSING COMPLETED SUCCESSFULLY")
