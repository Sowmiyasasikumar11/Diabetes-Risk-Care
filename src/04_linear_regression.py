from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

print("=" * 60)
print("LINEAR REGRESSION FEATURE ANALYSIS")
print("=" * 60)

project_root = Path(__file__).resolve().parents[1]
engineered_path = project_root / "outputs" / "engineered_features.csv"
engineered_df = pd.read_csv(engineered_path)

y = engineered_df["Diabetes_Outcome"]
X_engineered = engineered_df.drop(columns=["Diabetes_Outcome"]).copy()

X_train, X_test, y_train, y_test = train_test_split(
    X_engineered,
    y,
    test_size=0.20,
    random_state=42,
)

linear_regression_model = LinearRegression()
linear_regression_model.fit(X_train, y_train)

y_pred = linear_regression_model.predict(X_test)

linear_regression_coefficients = linear_regression_model.coef_
linear_regression_intercept = linear_regression_model.intercept_
linear_regression_r2 = r2_score(y_test, y_pred)
linear_regression_mse = mean_squared_error(y_test, y_pred)
linear_regression_rmse = (linear_regression_mse) ** 0.5

print("\nLinear Regression Intercept:")
print(linear_regression_intercept)

print("\nLinear Regression Coefficients:")
for feature, coefficient in zip(X_engineered.columns, linear_regression_coefficients):
    print(f"{feature}: {coefficient}")

print("\nR² Score:")
print(linear_regression_r2)

print("\nMean Squared Error (MSE):")
print(linear_regression_mse)

print("\nRoot Mean Squared Error (RMSE):")
print(linear_regression_rmse)

linear_regression_feature_analysis = pd.DataFrame(
    {
        "Feature": X_engineered.columns,
        "Coefficient": linear_regression_coefficients,
        "Absolute_Coefficient": abs(linear_regression_coefficients),
    }
).sort_values(by="Absolute_Coefficient", ascending=False).reset_index(drop=True)

print("\nFeature Analysis Table (sorted by absolute coefficient):")
print(linear_regression_feature_analysis)

linear_regression_output_path = project_root / "outputs" / "linear_regression_analysis.csv"
linear_regression_output_path.parent.mkdir(parents=True, exist_ok=True)
linear_regression_feature_analysis.to_csv(linear_regression_output_path, index=False)

print("\nLINEAR REGRESSION FEATURE ANALYSIS COMPLETED SUCCESSFULLY")
