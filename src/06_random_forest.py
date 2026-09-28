from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

print("=" * 60)
print("RANDOM FOREST CLASSIFIER")
print("=" * 60)

project_root = Path(__file__).resolve().parents[1]
selected_features_path = project_root / "outputs" / "selected_features.csv"

selected_df = pd.read_csv(selected_features_path)
target_column = "Diabetes_Outcome"
feature_columns = [column for column in selected_df.columns if column != target_column]

X = selected_df[feature_columns]
y = selected_df[target_column]

print("\nInput file:")
print(selected_features_path)
print("\nFeature columns:")
print(feature_columns)
print("\nTarget column:")
print(target_column)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)

training_accuracy = rf_model.score(X_train, y_train)
testing_accuracy = rf_model.score(X_test, y_test)

print("\nNumber of training records:", len(X_train))
print("Number of testing records:", len(X_test))
print("Number of features:", len(feature_columns))
print("Feature names:", feature_columns)
print("Training accuracy:", training_accuracy)
print("Testing accuracy:", testing_accuracy)
print("Prediction probability shape:", y_pred_proba.shape)

rf_feature_importance = pd.DataFrame(
    {
        "Feature": feature_columns,
        "Importance": rf_model.feature_importances_,
    }
).sort_values("Importance", ascending=False).reset_index(drop=True)

print("\nRandom Forest Feature Importance Ranking:")
print(rf_feature_importance.to_string(index=False))

feature_importance_path = project_root / "outputs" / "random_forest_feature_importance.csv"
model_path = project_root / "outputs" / "random_forest_model.pkl"
feature_importance_path.parent.mkdir(parents=True, exist_ok=True)
rf_feature_importance.to_csv(feature_importance_path, index=False)
joblib.dump(rf_model, model_path)

print("\nFeature importance saved to:")
print(feature_importance_path)
print("Trained model saved to:")
print(model_path)
print("\nRANDOM FOREST CLASSIFIER COMPLETED SUCCESSFULLY")
