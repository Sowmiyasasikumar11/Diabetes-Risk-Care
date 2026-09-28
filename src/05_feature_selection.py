from pathlib import Path

import pandas as pd

print("=" * 60)
print("FEATURE SELECTION / FEATURE ANALYSIS")
print("=" * 60)

project_root = Path(__file__).resolve().parents[1]
engineered_path = project_root / "outputs" / "engineered_features.csv"
analysis_path = project_root / "outputs" / "linear_regression_analysis.csv"

engineered_df = pd.read_csv(engineered_path)
y = engineered_df["Diabetes_Outcome"]
X_engineered = engineered_df.drop(columns=["Diabetes_Outcome"]).copy()

linear_regression_feature_analysis = pd.read_csv(analysis_path)
feature_selection_table = linear_regression_feature_analysis.copy()
feature_selection_table["Selected"] = False

threshold = feature_selection_table["Absolute_Coefficient"].quantile(0.60)
selected_mask = feature_selection_table["Absolute_Coefficient"] >= threshold
feature_selection_table.loc[selected_mask, "Selected"] = True

selected_features = feature_selection_table.loc[selected_mask, "Feature"].tolist()
X_selected = X_engineered[selected_features].copy()

print("\nTotal features before selection:")
print(X_engineered.shape[1])

print("\nTotal features after selection:")
print(X_selected.shape[1])

print("\nSelected Features:")
for i, feature in enumerate(selected_features, start=1):
    print(f"{i}. {feature}")

print("\nFeature | Coefficient | Absolute_Coefficient | Selected")
print(feature_selection_table[["Feature", "Coefficient", "Absolute_Coefficient", "Selected"]].to_string(index=False))

print("\nX_selected rows:", X_selected.shape[0])
print("y rows:", y.shape[0])
print("Target leakage check:", all(feature != "Diabetes_Outcome" for feature in selected_features))
print("Rows match check:", X_selected.shape[0] == y.shape[0])

selected_output = X_selected.copy()
selected_output["Diabetes_Outcome"] = y.values
selected_output_path = project_root / "outputs" / "selected_features.csv"
selected_output_path.parent.mkdir(parents=True, exist_ok=True)
selected_output.to_csv(selected_output_path, index=False)

print("\nFEATURE SELECTION COMPLETED SUCCESSFULLY")
