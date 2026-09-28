from pathlib import Path

import pandas as pd

print("=" * 60)
print("SEVEN EXPERIMENTAL RESULT TABLES")
print("=" * 60)

project_root = Path(__file__).resolve().parents[1]
raw_path = project_root / "dataset" / "diabetes_prediction_dataset_200_records.csv"
preprocessed_path = project_root / "outputs" / "preprocessed_data.csv"
engineered_path = project_root / "outputs" / "engineered_features.csv"
regression_path = project_root / "outputs" / "linear_regression_analysis.csv"
evaluation_path = project_root / "outputs" / "model_evaluation_results.csv"
confusion_path = project_root / "outputs" / "confusion_matrix.csv"
importance_path = project_root / "outputs" / "random_forest_feature_importance.csv"

raw_df = pd.read_csv(raw_path)
preprocessed_df = pd.read_csv(preprocessed_path)
engineered_df = pd.read_csv(engineered_path)
regression_df = pd.read_csv(regression_path)
evaluation_df = pd.read_csv(evaluation_path)
confusion_df = pd.read_csv(confusion_path, index_col=0)
importance_df = pd.read_csv(importance_path)

output_dir = project_root / "outputs" / "experimental_tables"
output_dir.mkdir(parents=True, exist_ok=True)

# Table 1: dataset characteristics.
target_column = "Diabetes_Outcome"
positive_cases = int((raw_df[target_column] == 1).sum())
negative_cases = int((raw_df[target_column] == 0).sum())
table_01 = pd.DataFrame(
    {
        "Characteristic": [
            "Total records",
            "Total original features",
            "Target variable",
            "Positive diabetes cases",
            "Negative diabetes cases",
        ],
        "Value": [
            len(raw_df),
            len(raw_df.columns) - 1,
            target_column,
            positive_cases,
            negative_cases,
        ],
    }
)

# Table 2: preprocessing and data quality.
feature_df = raw_df.drop(columns=["Patient_ID", target_column])
encoded_feature_df = preprocessed_df.drop(columns=[target_column])
missing_before = int(raw_df.isna().sum().sum())
missing_after = int(preprocessed_df.isna().sum().sum())
number_numerical_features = int(
    feature_df.select_dtypes(include="number").shape[1]
)
number_categorical_features = int(
    feature_df.select_dtypes(exclude="number").shape[1]
)
table_02 = pd.DataFrame(
    {
        "Metric": [
            "Missing values before preprocessing",
            "Missing values after preprocessing",
            "Duplicate records removed",
            "Number of numerical features",
            "Number of categorical features",
            "Number of final encoded features",
        ],
        "Value": [
            missing_before,
            missing_after,
            int(raw_df.duplicated().sum()),
            number_numerical_features,
            number_categorical_features,
            encoded_feature_df.shape[1],
        ],
    }
)

# Table 3: descriptive statistics for the numerical health variables.
numerical_health_columns = feature_df.select_dtypes(include="number").columns.tolist()
descriptive_stats = feature_df[numerical_health_columns].describe().T
table_03 = descriptive_stats.reset_index().rename(columns={"index": "Feature"})[
    ["Feature", "mean", "std", "min", "max"]
].rename(
    columns={
        "mean": "Mean",
        "std": "Standard Deviation",
        "min": "Minimum",
        "max": "Maximum",
    }
)

# Table 4: engineered feature names and category counts from Stage 3 output.
engineered_feature_definitions = {
    "BMI_Category": "BMI classification category",
    "Age_Group": "Age range category",
    "Glucose_Risk": "Blood glucose risk category",
    "HbA1c_Risk": "HbA1c risk category",
    "Cardiovascular_Risk": "Combined hypertension and heart disease risk category",
}
table_04_rows = []
for feature_name, description in engineered_feature_definitions.items():
    category_count = sum(
        column.startswith(f"{feature_name}_")
        for column in engineered_df.columns
    )
    table_04_rows.append(
        {
            "Feature name": feature_name,
            "Description": description,
            "Number of categories/values": category_count,
        }
    )
table_04 = pd.DataFrame(table_04_rows)

# Table 5: ranked Linear Regression feature analysis.
table_05 = regression_df[
    ["Feature", "Coefficient", "Absolute_Coefficient"]
].copy()
table_05 = table_05.sort_values(
    "Absolute_Coefficient", ascending=False
).reset_index(drop=True)
table_05["Rank"] = table_05.index + 1
table_05 = table_05[["Feature", "Coefficient", "Absolute_Coefficient", "Rank"]]

# Table 6: Random Forest performance from the existing evaluation output.
metric_order = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
table_06 = evaluation_df.set_index("Metric").loc[metric_order].reset_index()

# Table 7: confusion matrix followed by Random Forest feature importance.
confusion_values = {
    "True Negative": int(confusion_df.loc["Actual_0", "Predicted_0"]),
    "False Positive": int(confusion_df.loc["Actual_0", "Predicted_1"]),
    "False Negative": int(confusion_df.loc["Actual_1", "Predicted_0"]),
    "True Positive": int(confusion_df.loc["Actual_1", "Predicted_1"]),
}
table_07_rows = [
    {
        "Section": "Confusion Matrix",
        "Item": item,
        "Value": value,
        "Feature": "",
        "Importance": "",
        "Rank": "",
    }
    for item, value in confusion_values.items()
]
ranked_importance = importance_df.sort_values(
    "Importance", ascending=False
).reset_index(drop=True)
for rank, row in ranked_importance.iterrows():
    table_07_rows.append(
        {
            "Section": "Random Forest Feature Importance",
            "Item": "",
            "Value": "",
            "Feature": row["Feature"],
            "Importance": row["Importance"],
            "Rank": rank + 1,
        }
    )
table_07 = pd.DataFrame(table_07_rows)

tables = [
    table_01,
    table_02,
    table_03,
    table_04,
    table_05,
    table_06,
    table_07,
]
csv_names = [
    "table_01_dataset_characteristics.csv",
    "table_02_data_quality.csv",
    "table_03_descriptive_statistics.csv",
    "table_04_feature_engineering.csv",
    "table_05_linear_regression_analysis.csv",
    "table_06_random_forest_performance.csv",
    "table_07_confusion_matrix_feature_importance.csv",
]

for table, csv_name in zip(tables, csv_names):
    table.to_csv(output_dir / csv_name, index=False)

workbook_path = project_root / "outputs" / "experimental_results_7_tables.xlsx"
sheet_names = [
    "Table 1 - Dataset",
    "Table 2 - Data Quality",
    "Table 3 - Descriptive Stats",
    "Table 4 - Feature Engineering",
    "Table 5 - Linear Regression",
    "Table 6 - RF Performance",
    "Table 7 - Confusion & Importance",
]
with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
    for table, sheet_name in zip(tables, sheet_names):
        table.to_excel(writer, sheet_name=sheet_name, index=False)

print("\nGenerated experimental tables:", len(tables))
for csv_name in csv_names:
    print(output_dir / csv_name)
print("\nExcel workbook:")
print(workbook_path)
print("\nSEVEN EXPERIMENTAL TABLES GENERATED SUCCESSFULLY")
