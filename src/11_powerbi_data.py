from pathlib import Path

import pandas as pd

print("=" * 60)
print("POWER BI DASHBOARD DATA PREPARATION")
print("=" * 60)

project_root = Path(__file__).resolve().parents[1]
raw_path = project_root / "dataset" / "diabetes_prediction_dataset_200_records.csv"
evaluation_path = project_root / "outputs" / "model_evaluation_results.csv"
importance_path = project_root / "outputs" / "random_forest_feature_importance.csv"
regression_path = project_root / "outputs" / "linear_regression_analysis.csv"

raw_df = pd.read_csv(raw_path)
evaluation_df = pd.read_csv(evaluation_path)
importance_df = pd.read_csv(importance_path)
regression_df = pd.read_csv(regression_path)

# Apply the same category boundaries used by Stage 3.
patient_df = raw_df.copy()
patient_df["BMI_Category"] = pd.cut(
    patient_df["BMI"],
    bins=[0, 18.5, 24.9, 29.9, float("inf")],
    labels=["Underweight", "Normal", "Overweight", "Obese"],
    right=False,
)
patient_df["Age_Group"] = pd.cut(
    patient_df["Age"],
    bins=[0, 18, 35, 50, 65, float("inf")],
    labels=["0-17", "18-34", "35-49", "50-64", "65+"],
    right=False,
)
patient_df["Glucose_Risk"] = pd.cut(
    patient_df["Blood_Glucose_Level"],
    bins=[0, 99, 125, float("inf")],
    labels=["Normal", "Prediabetes", "High_Risk"],
    right=False,
)
patient_df["HbA1c_Risk"] = pd.cut(
    patient_df["HbA1c_Level"],
    bins=[0, 5.7, 6.4, float("inf")],
    labels=["Normal", "Prediabetes", "High_Risk"],
    right=False,
)
patient_df["Cardiovascular_Risk"] = (
    patient_df["Hypertension"].astype(int) + patient_df["Heart_Disease"].astype(int)
).clip(lower=0, upper=2).map({0: "Low", 1: "Moderate", 2: "High"})

original_columns = [
    "Patient_ID",
    "Age",
    "Gender",
    "BMI",
    "Blood_Glucose_Level",
    "HbA1c_Level",
    "Hypertension",
    "Heart_Disease",
    "Smoking_History",
    "Pregnancies",
    "Diabetes_Outcome",
]
derived_columns = [
    "Age_Group",
    "BMI_Category",
    "Glucose_Risk",
    "HbA1c_Risk",
    "Cardiovascular_Risk",
]
patient_columns = [
    column for column in original_columns + derived_columns if column in patient_df.columns
]
powerbi_patient_data = patient_df[patient_columns]

performance_metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
powerbi_model_performance = evaluation_df.set_index("Metric").loc[
    performance_metrics
].reset_index()

powerbi_feature_importance = importance_df.sort_values(
    "Importance", ascending=False
).reset_index(drop=True)
powerbi_feature_importance["Rank"] = powerbi_feature_importance.index + 1
powerbi_feature_importance = powerbi_feature_importance[
    ["Feature", "Importance", "Rank"]
]

powerbi_linear_regression = regression_df.sort_values(
    "Absolute_Coefficient", ascending=False
).reset_index(drop=True)
powerbi_linear_regression["Rank"] = powerbi_linear_regression.index + 1
powerbi_linear_regression = powerbi_linear_regression[
    ["Feature", "Coefficient", "Absolute_Coefficient", "Rank"]
]

output_paths = {
    "powerbi_patient_data.csv": powerbi_patient_data,
    "powerbi_model_performance.csv": powerbi_model_performance,
    "powerbi_feature_importance.csv": powerbi_feature_importance,
    "powerbi_linear_regression.csv": powerbi_linear_regression,
}
for filename, dataframe in output_paths.items():
    dataframe.to_csv(project_root / "outputs" / filename, index=False)

positive_cases = int((raw_df["Diabetes_Outcome"] == 1).sum())
negative_cases = int((raw_df["Diabetes_Outcome"] == 0).sum())
diabetes_percentage = positive_cases / len(raw_df) * 100

print("\nRecommended Power BI Dashboard Layout")
print("\nPAGE 1 - OVERVIEW")
print("Cards: Total Patients, Diabetes Cases, Non-Diabetes Cases, Diabetes Percentage")
print("Charts: Diabetes Outcome Distribution, Age Group Distribution, Gender Distribution, BMI Category Distribution")
print("\nPAGE 2 - RISK FACTOR ANALYSIS")
print("Charts: Blood Glucose Distribution, HbA1c Distribution, BMI Distribution")
print("Charts: Hypertension vs Diabetes, Heart Disease vs Diabetes, Smoking History vs Diabetes")
print("\nPAGE 3 - MODEL ANALYTICS")
print("Cards: Accuracy, Precision, Recall, F1-Score, ROC-AUC")
print("Charts: Model Performance, Random Forest Feature Importance, Linear Regression Feature Analysis")

print("\nNumber of Power BI CSV files created:", len(output_paths))
print("Created files:")
for filename in output_paths:
    print(project_root / "outputs" / filename)
print("Number of patient records:", len(powerbi_patient_data))
print("Number of diabetes cases:", positive_cases)
print("Number of non-diabetes cases:", negative_cases)
print("Diabetes percentage:", diabetes_percentage)
print("\nPOWER BI DATA PREPARATION COMPLETED SUCCESSFULLY")
