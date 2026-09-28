from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

print("=" * 60)
print("DATA VISUALIZATION / GRAPHS")
print("=" * 60)

project_root = Path(__file__).resolve().parents[1]
figures_dir = project_root / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

raw_path = project_root / "dataset" / "diabetes_prediction_dataset_200_records.csv"
regression_path = project_root / "outputs" / "linear_regression_analysis.csv"
importance_path = project_root / "outputs" / "random_forest_feature_importance.csv"
confusion_path = project_root / "outputs" / "confusion_matrix.csv"
evaluation_path = project_root / "outputs" / "model_evaluation_results.csv"

raw_df = pd.read_csv(raw_path)
regression_df = pd.read_csv(regression_path)
importance_df = pd.read_csv(importance_path)
confusion_df = pd.read_csv(confusion_path, index_col=0)
evaluation_df = pd.read_csv(evaluation_path)

sns.set_theme(style="whitegrid", context="notebook")


def save_figure(fig, stem):
    fig.tight_layout()
    fig.savefig(figures_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


generated_files = []

outcome_labels = {0: "No Diabetes", 1: "Diabetes"}
outcome_counts = raw_df["Diabetes_Outcome"].value_counts().sort_index()
outcome_counts.index = [outcome_labels.get(value, str(value)) for value in outcome_counts.index]
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(
    x=outcome_counts.index,
    y=outcome_counts.values,
    hue=outcome_counts.index,
    palette=["#4C78A8", "#E45756"],
    legend=False,
    ax=ax,
)
ax.set_title("Diabetes Outcome Distribution")
ax.set_xlabel("Diabetes Outcome")
ax.set_ylabel("Number of Records")
save_figure(fig, "01_diabetes_outcome_distribution")
generated_files.extend(
    [
        "01_diabetes_outcome_distribution.png",
        "01_diabetes_outcome_distribution.pdf",
    ]
)

plot_df = raw_df.copy()
plot_df["Outcome"] = plot_df["Diabetes_Outcome"].map(outcome_labels)
plot_specs = [
    ("Age", "Age Distribution by Diabetes Outcome", "Age", "02_age_distribution"),
    ("BMI", "BMI Distribution by Diabetes Outcome", "BMI", "03_bmi_distribution"),
    (
        "Blood_Glucose_Level",
        "Blood Glucose Level by Diabetes Outcome",
        "Blood Glucose Level",
        "04_glucose_distribution",
    ),
    (
        "HbA1c_Level",
        "HbA1c Level by Diabetes Outcome",
        "HbA1c Level",
        "05_hba1c_distribution",
    ),
]
for column, title, x_label, stem in plot_specs:
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.histplot(
        data=plot_df,
        x=column,
        hue="Outcome",
        bins=15,
        stat="density",
        common_norm=False,
        element="step",
        fill=True,
        alpha=0.35,
        palette={"No Diabetes": "#4C78A8", "Diabetes": "#E45756"},
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Density")
    save_figure(fig, stem)
    generated_files.extend([f"{stem}.png", f"{stem}.pdf"])

correlation_columns = [
    "Age",
    "BMI",
    "Blood_Glucose_Level",
    "HbA1c_Level",
    "Hypertension",
    "Heart_Disease",
    "Pregnancies",
    "Diabetes_Outcome",
]
correlation_columns = [
    column for column in correlation_columns if column in raw_df.columns
]
correlation = raw_df[correlation_columns].corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="vlag",
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"label": "Correlation"},
    ax=ax,
)
ax.set_title("Correlation Heatmap")
save_figure(fig, "06_correlation_heatmap")
generated_files.extend(["06_correlation_heatmap.png", "06_correlation_heatmap.pdf"])

regression_sorted = regression_df.sort_values(
    "Absolute_Coefficient", ascending=True
)
fig, ax = plt.subplots(figsize=(10, 12))
ax.barh(
    regression_sorted["Feature"],
    regression_sorted["Absolute_Coefficient"],
    color="#4C78A8",
)
ax.set_title("Linear Regression Feature Coefficients")
ax.set_xlabel("Absolute Coefficient")
ax.set_ylabel("Feature")
save_figure(fig, "07_linear_regression_coefficients")
generated_files.extend(
    ["07_linear_regression_coefficients.png", "07_linear_regression_coefficients.pdf"]
)

importance_sorted = importance_df.sort_values("Importance", ascending=True)
fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(
    importance_sorted["Feature"],
    importance_sorted["Importance"],
    color="#F58518",
)
ax.set_title("Random Forest Feature Importance")
ax.set_xlabel("Importance")
ax.set_ylabel("Feature")
save_figure(fig, "08_random_forest_feature_importance")
generated_files.extend(
    ["08_random_forest_feature_importance.png", "08_random_forest_feature_importance.pdf"]
)

confusion_labels = ["No Diabetes", "Diabetes"]
confusion_values = confusion_df.loc[
    ["Actual_0", "Actual_1"], ["Predicted_0", "Predicted_1"]
].to_numpy()
confusion_annotations = pd.DataFrame(
    [
        [f"TN\n{confusion_values[0, 0]}", f"FP\n{confusion_values[0, 1]}"],
        [f"FN\n{confusion_values[1, 0]}", f"TP\n{confusion_values[1, 1]}"],
    ],
    index=confusion_labels,
    columns=confusion_labels,
)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    confusion_values,
    annot=confusion_annotations,
    fmt="",
    cmap="Blues",
    cbar=False,
    xticklabels=confusion_labels,
    yticklabels=confusion_labels,
    linewidths=1,
    linecolor="white",
    ax=ax,
)
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted Outcome")
ax.set_ylabel("Actual Outcome")
save_figure(fig, "09_confusion_matrix")
generated_files.extend(["09_confusion_matrix.png", "09_confusion_matrix.pdf"])

metric_order = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
metrics = evaluation_df.set_index("Metric").loc[metric_order, "Value"]
fig, ax = plt.subplots(figsize=(9, 6))
sns.barplot(
    x=metrics.index,
    y=metrics.values,
    hue=metrics.index,
    palette="Blues_d",
    legend=False,
    ax=ax,
)
ax.set_title("Random Forest Model Performance Metrics")
ax.set_xlabel("Metric")
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
for index, value in enumerate(metrics.values):
    ax.text(index, value, f"{value:.3f}", ha="center", va="bottom")
save_figure(fig, "10_model_performance_metrics")
generated_files.extend(
    ["10_model_performance_metrics.png", "10_model_performance_metrics.pdf"]
)

print("\nNumber of figures generated:", len(generated_files) // 2)
print("Generated files:")
for filename in generated_files:
    print(figures_dir / filename)
print("\nDATA VISUALIZATION COMPLETED SUCCESSFULLY")
