# ============================================================
# ShopSense - Evaluate all saved models
# Run train_models.py first, then run:  python evaluate_models.py
# Results are saved in the "reports" folder.
# ============================================================

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# Use the SAME cleaning steps as training
from train_models import load_and_clean_data, make_inputs

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

df = load_and_clean_data()
X = make_inputs(df)
y_reg = df["total_spending"]
y_class = df["high_value"]


# ------------------------------------------------------------
# PART 1: Regression model (Linear Regression)
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

regression_model = joblib.load(MODELS_DIR / "regression_model.pkl")
predictions = regression_model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("=" * 45)
print("Linear Regression (predicts total spending)")
print("=" * 45)
print("MAE :", round(mae, 2), " -> average mistake in spending")
print("RMSE:", round(rmse, 2))
print("R2  :", round(r2, 4), " -> closer to 1 is better")

pd.DataFrame([{
    "Model": "Linear Regression",
    "MAE": round(mae, 2),
    "RMSE": round(rmse, 2),
    "R2 Score": round(r2, 4),
}]).to_csv(REPORTS_DIR / "regression_performance.csv", index=False)


# ------------------------------------------------------------
# PART 2: Classification models (High-Value / Regular)
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_class, test_size=0.2, random_state=42, stratify=y_class
)

# (name, file prefix, uses scaler?)
models = [
    ("Logistic Regression", "classification", True),
    ("KNN", "knn", True),
    ("ANN", "ann", True),
    ("SVM", "svm", True),
    ("Decision Tree", "decision_tree", False),
    ("Naive Bayes", "naive_bayes", False),
    ("Random Forest", "random_forest", False),
]

results = []

for name, prefix, uses_scaler in models:
    model = joblib.load(MODELS_DIR / f"{prefix}_model.pkl")
    columns = joblib.load(MODELS_DIR / f"{prefix}_columns.pkl")

    # Put the columns in the same order used in training
    X_test_model = X_test[columns]

    # Scale only the models that were trained on scaled data
    if uses_scaler:
        scaler = joblib.load(MODELS_DIR / f"{prefix}_scaler.pkl")
        X_test_model = scaler.transform(X_test_model)

    predictions = model.predict(X_test_model)

    metrics = {
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, predictions) * 100, 2),
        "Precision": round(precision_score(y_test, predictions) * 100, 2),
        "Recall": round(recall_score(y_test, predictions) * 100, 2),
        "F1-Score": round(f1_score(y_test, predictions) * 100, 2),
    }
    results.append(metrics)

    print("\n" + "=" * 45)
    print(name)
    print("=" * 45)
    print("Accuracy :", metrics["Accuracy"], "%")
    print("Precision:", metrics["Precision"], "%")
    print("Recall   :", metrics["Recall"], "%")
    print("F1-Score :", metrics["F1-Score"], "%")
    print("\nConfusion Matrix  [[TN FP] [FN TP]]:")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification Report:")
    print(classification_report(
        y_test, predictions, target_names=["Regular", "High-Value"]
    ))

# Sort from best to worst accuracy and save
results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
results_df.to_csv(REPORTS_DIR / "model_performance.csv", index=False)
results_df.to_csv(BASE_DIR / "model_performance.csv", index=False)

print("\n\nFINAL MODEL COMPARISON")
print(results_df.to_string(index=False))
print("\nResults saved to reports/model_performance.csv")
