# ============================================================
# ShopSense - FastAPI backend
# Start it with:  uvicorn main:app --reload
# Then open:      http://127.0.0.1:8000
# API docs:       http://127.0.0.1:8000/docs
# ============================================================

from typing import Literal
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pathlib import Path
import pandas as pd
import joblib

app = FastAPI(title="ShopSense - Customer Spending Predictor")

# Allow the frontend to call the API (even if index.html is opened directly)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"


# ------------------------------------------------------------
# Load all saved models when the server starts
# ------------------------------------------------------------

# Linear Regression (predicts total spending)
regression_model = joblib.load(MODELS_DIR / "regression_model.pkl")
regression_columns = joblib.load(MODELS_DIR / "regression_columns.pkl")

# Logistic Regression (needs scaler)
classification_model = joblib.load(MODELS_DIR / "classification_model.pkl")
classification_scaler = joblib.load(MODELS_DIR / "classification_scaler.pkl")
classification_columns = joblib.load(MODELS_DIR / "classification_columns.pkl")

# KNN (needs scaler)
knn_model = joblib.load(MODELS_DIR / "knn_model.pkl")
knn_scaler = joblib.load(MODELS_DIR / "knn_scaler.pkl")
knn_columns = joblib.load(MODELS_DIR / "knn_columns.pkl")

# ANN (needs scaler)
ann_model = joblib.load(MODELS_DIR / "ann_model.pkl")
ann_scaler = joblib.load(MODELS_DIR / "ann_scaler.pkl")
ann_columns = joblib.load(MODELS_DIR / "ann_columns.pkl")

# SVM (needs scaler)
svm_model = joblib.load(MODELS_DIR / "svm_model.pkl")
svm_scaler = joblib.load(MODELS_DIR / "svm_scaler.pkl")
svm_columns = joblib.load(MODELS_DIR / "svm_columns.pkl")

# Decision Tree
decision_tree_model = joblib.load(MODELS_DIR / "decision_tree_model.pkl")
decision_tree_columns = joblib.load(MODELS_DIR / "decision_tree_columns.pkl")

# Naive Bayes
naive_bayes_model = joblib.load(MODELS_DIR / "naive_bayes_model.pkl")
naive_bayes_columns = joblib.load(MODELS_DIR / "naive_bayes_columns.pkl")

# Random Forest
random_forest_model = joblib.load(MODELS_DIR / "random_forest_model.pkl")
random_forest_columns = joblib.load(MODELS_DIR / "random_forest_columns.pkl")


# ------------------------------------------------------------
# The data the frontend sends (with allowed values)
# ------------------------------------------------------------
class Customer(BaseModel):
    age: int = Field(ge=18, le=90)
    education: Literal["Basic", "2n Cycle", "Graduation", "Master", "PhD"]
    marital_status: Literal["Single", "Married", "Together", "Divorced", "Widow"]
    income: float = Field(ge=0, le=200000)
    kids: int = Field(ge=0, le=5)
    teens: int = Field(ge=0, le=5)
    recency: int = Field(ge=0, le=365)
    web_visits: int = Field(ge=0, le=30)


def prepare_data(customer, columns):
    """Turn the customer into a one-row table the model understands."""
    data = pd.DataFrame([customer.model_dump()])
    data = pd.get_dummies(data, dtype=int)                # text -> 0/1 columns
    data = data.reindex(columns=columns, fill_value=0)    # same columns as training
    return data


def predict_classification_model(customer, model, columns, scaler=None):
    """Run one classification model and return the result."""
    data = prepare_data(customer, columns)

    if scaler is not None:
        data = scaler.transform(data)

    prediction = int(model.predict(data)[0])
    result = "High-Value" if prediction == 1 else "Regular"

    response = {"prediction": prediction, "result": result}

    # Some models can also give a probability (confidence)
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(data)[0][1])
        response["probability"] = round(probability * 100, 1)

    return response


# ------------------------------------------------------------
# Routes (endpoints)
# ------------------------------------------------------------

# Homepage -> the website
@app.get("/")
def homepage():
    return FileResponse(BASE_DIR / "index.html")


# Linear Regression -> expected spending
@app.post("/predict/regression")
def predict_regression(customer: Customer):
    data = prepare_data(customer, regression_columns)
    prediction = float(regression_model.predict(data)[0])
    # A straight line can go below zero, but spending cannot
    return {"prediction": round(max(0, prediction), 2)}


# Logistic Regression
@app.post("/predict/classification")
def predict_classification(customer: Customer):
    return predict_classification_model(
        customer, classification_model, classification_columns, classification_scaler
    )


# KNN
@app.post("/predict/knn")
def predict_knn(customer: Customer):
    return predict_classification_model(customer, knn_model, knn_columns, knn_scaler)


# ANN
@app.post("/predict/ann")
def predict_ann(customer: Customer):
    return predict_classification_model(customer, ann_model, ann_columns, ann_scaler)


# SVM
@app.post("/predict/svm")
def predict_svm(customer: Customer):
    return predict_classification_model(customer, svm_model, svm_columns, svm_scaler)


# Decision Tree
@app.post("/predict/decision-tree")
def predict_decision_tree(customer: Customer):
    return predict_classification_model(customer, decision_tree_model, decision_tree_columns)


# Naive Bayes
@app.post("/predict/naive-bayes")
def predict_naive_bayes(customer: Customer):
    return predict_classification_model(customer, naive_bayes_model, naive_bayes_columns)


# Random Forest
@app.post("/predict/random-forest")
def predict_random_forest(customer: Customer):
    return predict_classification_model(customer, random_forest_model, random_forest_columns)


# Model performance table (made by evaluate_models.py)
@app.get("/model-performance")
def get_model_performance():
    path = REPORTS_DIR / "model_performance.csv"
    if not path.exists():
        return {"error": "Run evaluate_models.py first"}
    return pd.read_csv(path).to_dict(orient="records")


# Regression performance (made by evaluate_models.py)
@app.get("/regression-performance")
def get_regression_performance():
    path = REPORTS_DIR / "regression_performance.csv"
    if not path.exists():
        return {"error": "Run evaluate_models.py first"}
    return pd.read_csv(path).to_dict(orient="records")[0]
