from pathlib import Path

import joblib
import pandas as pd

print("=" * 60)
print("DIABETES RISK PREDICTION")
print("=" * 60)

project_root = Path(__file__).resolve().parents[1]
model_path = project_root / "outputs" / "random_forest_model.pkl"
selected_features_path = project_root / "outputs" / "selected_features.csv"

rf_model = joblib.load(model_path)
selected_df = pd.read_csv(selected_features_path)

if "Diabetes_Outcome" not in selected_df.columns:
    raise ValueError("Diabetes_Outcome column is missing from selected_features.csv.")

selected_feature_names = [
    column for column in selected_df.columns if column != "Diabetes_Outcome"
]
model_feature_names = list(rf_model.feature_names_in_)

if selected_feature_names != model_feature_names:
    raise ValueError(
        "The selected feature order does not match the trained Random Forest model."
    )


def get_positive_number(prompt):
    while True:
        value = input(prompt).strip()
        try:
            number = float(value)
        except ValueError:
            print("Please enter a numeric value.")
            continue
        if number <= 0:
            print("Please enter a value greater than 0.")
            continue
        return number


def get_binary_value(prompt):
    while True:
        value = input(prompt).strip()
        if value in {"0", "1"}:
            return int(value)
        print("Please enter 0 or 1.")


def get_smoking_history():
    valid_values = {"current", "former", "never"}
    while True:
        value = input("Smoking history (current/former/never): ").strip().lower()
        if value in valid_values:
            return value
        print("Please enter current, former, or never.")


age = get_positive_number("Age: ")
bmi = get_positive_number("BMI: ")
blood_glucose = get_positive_number("Blood glucose level: ")
hba1c = get_positive_number("HbA1c level: ")
hypertension = get_binary_value("Hypertension (0 or 1): ")
heart_disease = get_binary_value("Heart disease (0 or 1): ")
smoking_history = get_smoking_history()

patient_data = pd.DataFrame(
    [
        {
            "Age": age,
            "BMI": bmi,
            "Blood_Glucose_Level": blood_glucose,
            "HbA1c_Level": hba1c,
            "Hypertension": hypertension,
            "Heart_Disease": heart_disease,
            "Smoking_History": smoking_history,
        }
    ]
)

# Apply the same bins and categorical encoding used by Stage 3.
patient_data["BMI_Category"] = pd.cut(
    patient_data["BMI"],
    bins=[0, 18.5, 24.9, 29.9, float("inf")],
    labels=["Underweight", "Normal", "Overweight", "Obese"],
    right=False,
)
patient_data["Age_Group"] = pd.cut(
    patient_data["Age"],
    bins=[0, 18, 35, 50, 65, float("inf")],
    labels=["0-17", "18-34", "35-49", "50-64", "65+"],
    right=False,
)
patient_data["Glucose_Risk"] = pd.cut(
    patient_data["Blood_Glucose_Level"],
    bins=[0, 99, 125, float("inf")],
    labels=["Normal", "Prediabetes", "High_Risk"],
    right=False,
)
patient_data["HbA1c_Risk"] = pd.cut(
    patient_data["HbA1c_Level"],
    bins=[0, 5.7, 6.4, float("inf")],
    labels=["Normal", "Prediabetes", "High_Risk"],
    right=False,
)
patient_data["Cardiovascular_Risk"] = (
    patient_data["Hypertension"] + patient_data["Heart_Disease"]
).clip(lower=0, upper=2).map({0: "Low", 1: "Moderate", 2: "High"})

engineered_categorical_columns = [
    "BMI_Category",
    "Age_Group",
    "Glucose_Risk",
    "HbA1c_Risk",
    "Cardiovascular_Risk",
    "Smoking_History",
]
engineered_input = pd.get_dummies(
    patient_data,
    columns=engineered_categorical_columns,
    dtype=int,
)
input_data = engineered_input.reindex(columns=model_feature_names, fill_value=0)

prediction = int(rf_model.predict(input_data)[0])
prediction_probabilities = rf_model.predict_proba(input_data)[0]
class_index = list(rf_model.classes_).index(prediction)
risk_probability = float(prediction_probabilities[class_index])
risk_label = "HIGH RISK" if prediction == 1 else "LOW RISK"

print("\n----------------------------------------")
print("DIABETES RISK PREDICTION")
print("----------------------------------------")
print(f"Prediction: {risk_label}")
print(f"Risk Probability: {risk_probability:.2%}")
print("----------------------------------------")
print(
    "Diabetes Risk Prediction: HIGH RISK"
    if prediction == 1
    else "Diabetes Risk Prediction: LOW RISK"
)
print(
    "This result is a machine-learning risk prediction and is not a medical diagnosis. "
    "Please consult a qualified healthcare professional for medical advice."
)
print("\nDIABETES RISK PREDICTION COMPLETED SUCCESSFULLY")
