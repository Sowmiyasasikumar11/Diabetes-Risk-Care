from pathlib import Path

import pandas as pd

print("=" * 60)
print("FEATURE ENGINEERING / FEATURE EXTRACTION")
print("=" * 60)

project_root = Path(__file__).resolve().parents[1]
preprocessed_path = project_root / "outputs" / "preprocessed_data.csv"

df_processed = pd.read_csv(preprocessed_path)

# Keep the target separately
if "Diabetes_Outcome" not in df_processed.columns:
    raise ValueError("Diabetes_Outcome column is missing from preprocessed data.")

y = df_processed["Diabetes_Outcome"]
X = df_processed.drop(columns=["Diabetes_Outcome"]).copy()

print("\nOriginal feature names before feature engineering:")
print(X.columns.tolist())

X_before_engineering = X.copy()
new_feature_names = []

# BMI_Category
X["BMI_Category"] = pd.cut(
    X["BMI"],
    bins=[0, 18.5, 24.9, 29.9, float("inf")],
    labels=["Underweight", "Normal", "Overweight", "Obese"],
    right=False,
)
new_feature_names.append("BMI_Category")

# Age_Group


X["Age_Group"] = pd.cut(
    X["Age"],
    bins=[0, 18, 35, 50, 65, float("inf")],
    labels=["0-17", "18-34", "35-49", "50-64", "65+"],
    right=False,

)
new_feature_names.append("Age_Group")

# Glucose_Risk
X["Glucose_Risk"] = pd.cut(
    X["Blood_Glucose_Level"],
    bins=[0, 99, 125, float("inf")],
    labels=["Normal", "Prediabetes", "High_Risk"],
    right=False,
)
new_feature_names.append("Glucose_Risk")

# HbA1c_Risk
X["HbA1c_Risk"] = pd.cut(
    X["HbA1c_Level"],
    bins=[0, 5.7, 6.4, float("inf")],
    labels=["Normal", "Prediabetes", "High_Risk"],
    right=False,
)
new_feature_names.append("HbA1c_Risk")

# Cardiovascular_Risk
X["Cardiovascular_Risk"] = (
    X["Hypertension"].astype(int) + X["Heart_Disease"].astype(int)
).clip(lower=0, upper=2)
X["Cardiovascular_Risk"] = X["Cardiovascular_Risk"].map(
    {0: "Low", 1: "Moderate", 2: "High"}
)
new_feature_names.append("Cardiovascular_Risk")

print("\nNewly created feature names:")
print(new_feature_names)

engineered_categorical_columns = [
    col
    for col in new_feature_names
    if pd.api.types.is_object_dtype(X[col])
    or isinstance(X[col].dtype, pd.CategoricalDtype)
]

X_engineered = pd.get_dummies(X.copy(), columns=engineered_categorical_columns, dtype=int)

print("\nFinal engineered feature names:")
print(X_engineered.columns.tolist())
print("\nX shape before feature engineering:", X_before_engineering.shape)
print("X shape after feature engineering:", X_engineered.shape)
print("\nFirst 5 rows of engineered features:")
print(X_engineered.head())

engineered_output = X_engineered.copy()
engineered_output["Diabetes_Outcome"] = y.values
engineered_output_path = project_root / "outputs" / "engineered_features.csv"
engineered_output_path.parent.mkdir(parents=True, exist_ok=True)
engineered_output.to_csv(engineered_output_path, index=False)

print("\nFEATURE ENGINEERING COMPLETED SUCCESSFULLY")
