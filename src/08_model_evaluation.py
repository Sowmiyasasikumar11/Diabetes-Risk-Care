from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

print("=" * 40)
print("RANDOM FOREST MODEL EVALUATION")
print("=" * 40)

project_root = Path(__file__).resolve().parents[1]
model_path = project_root / "outputs" / "random_forest_model.pkl"
selected_features_path = project_root / "outputs" / "selected_features.csv"

rf_model = joblib.load(model_path)
selected_df = pd.read_csv(selected_features_path)
target_column = "Diabetes_Outcome"

if target_column not in selected_df.columns:
    raise ValueError("Diabetes_Outcome column is missing from selected_features.csv.")

feature_names = [
    column for column in selected_df.columns if column != target_column
]
model_feature_names = list(rf_model.feature_names_in_)

if feature_names != model_feature_names:
    raise ValueError(
        "The selected feature order does not match the trained Random Forest model."
    )

X = selected_df[feature_names]
y = selected_df[target_column]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Evaluate the existing model on the held-out test set only.
y_pred = rf_model.predict(X_test)
y_probability = rf_model.predict_proba(X_test)
positive_class_index = list(rf_model.classes_).index(1)
y_positive_probability = y_probability[:, positive_class_index]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
roc_auc = roc_auc_score(y_test, y_positive_probability)
confusion = confusion_matrix(y_test, y_pred, labels=[0, 1])
report = classification_report(y_test, y_pred, labels=[0, 1], zero_division=0)

print("\nTest Set Size:", len(X_test))
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
print("ROC-AUC:", roc_auc)

print("\nCONFUSION MATRIX")
print(confusion)

print("\nCLASSIFICATION REPORT")
print(report)

results = pd.DataFrame(
    {
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
        "Value": [accuracy, precision, recall, f1, roc_auc],
    }
)
results_path = project_root / "outputs" / "model_evaluation_results.csv"
results.to_csv(results_path, index=False)

confusion_df = pd.DataFrame(
    confusion,
    index=["Actual_0", "Actual_1"],
    columns=["Predicted_0", "Predicted_1"],
)
confusion_path = project_root / "outputs" / "confusion_matrix.csv"
confusion_df.to_csv(confusion_path, index=True)

print("\nEvaluation results saved to:")
print(results_path)
print("Confusion matrix saved to:")
print(confusion_path)
print("\nMODEL EVALUATION COMPLETED SUCCESSFULLY")
