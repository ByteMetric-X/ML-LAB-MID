# ============================================================
# ShopSense - Train all machine learning models
# This file cleans the data, trains 8 models and saves them
# in the "models" folder.
# Run it with:  python train_models.py
# ============================================================

import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# The 8 models used in this project
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# A customer who spent 400 or more is called a "High-Value Customer"
HIGH_VALUE_LIMIT = 400


def load_and_clean_data():
    """Load the CSV file and clean it. Used by every script."""
    df = pd.read_csv(BASE_DIR / "marketing_campaign.csv")

    # 1. Remove repeated customers (same data, only the ID is different)
    df = df.drop("ID", axis=1).drop_duplicates()

    # 2. Create the regression target: total money spent in 2 years
    spending_columns = ["MntWines", "MntFruits", "MntMeatProducts",
                        "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
    df["total_spending"] = df[spending_columns].sum(axis=1)

    # 3. Change year of birth into age (the data was collected in 2014)
    df["age"] = 2014 - df["Year_Birth"]

    # 4. Keep only the columns we need and give them simple names
    df = df[["age", "Education", "Marital_Status", "Income", "Kidhome",
             "Teenhome", "Recency", "NumWebVisitsMonth", "total_spending"]]
    df.columns = ["age", "education", "marital_status", "income", "kids",
                  "teens", "recency", "web_visits", "total_spending"]

    # 5. Remove rows where income is missing
    df = df.dropna()

    # 6. Remove impossible values (age above 90, income above 200,000)
    df = df[(df["age"] <= 90) & (df["income"] <= 200000)]

    # 7. Fix strange marital status values
    df["marital_status"] = df["marital_status"].replace(
        {"Alone": "Single", "Absurd": "Single", "YOLO": "Single"}
    )

    # 8. Create the classification target
    df["high_value"] = (df["total_spending"] >= HIGH_VALUE_LIMIT).astype(int)

    return df


def make_inputs(df):
    """Turn text columns into 0/1 columns (one-hot encoding)."""
    X = df.drop(["total_spending", "high_value"], axis=1)
    X = pd.get_dummies(X, drop_first=True, dtype=int)
    return X


if __name__ == "__main__":

    # --------------------------------------------------------
    # STEP 1: Load and clean the data
    # --------------------------------------------------------
    df = load_and_clean_data()
    print("Clean data shape:", df.shape)

    X = make_inputs(df)
    y_reg = df["total_spending"]   # number -> regression
    y_class = df["high_value"]     # 0 or 1 -> classification

    # Save the column names so the website sends data in the same order
    columns = X.columns.tolist()

    with open(BASE_DIR / "threshold.txt", "w") as file:
        file.write(str(HIGH_VALUE_LIMIT))

    # --------------------------------------------------------
    # STEP 2: Linear Regression (predicts total spending)
    # --------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )

    regression_model = LinearRegression()
    regression_model.fit(X_train, y_train)

    predictions = regression_model.predict(X_test)
    print("\nLinear Regression R2 :", round(r2_score(y_test, predictions), 4))
    print("Linear Regression MAE:", round(mean_absolute_error(y_test, predictions), 2))

    joblib.dump(regression_model, MODELS_DIR / "regression_model.pkl")
    joblib.dump(columns, MODELS_DIR / "regression_columns.pkl")

    # --------------------------------------------------------
    # STEP 3: Classification models (High-Value or Regular)
    # --------------------------------------------------------
    # stratify keeps the same 0/1 balance in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_class, test_size=0.2, random_state=42, stratify=y_class
    )

    # Income is in thousands but kids is 0-2, so some models need
    # scaling to put every column on a similar range.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # (name, file prefix, model, needs scaling?)
    models = [
        ("Logistic Regression", "classification", LogisticRegression(max_iter=1000), True),
        # n_neighbors=15 was chosen with 5-fold cross-validation (see KNN notebook)
        ("KNN", "knn", KNeighborsClassifier(n_neighbors=15), True),
        ("ANN", "ann", MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=2000, random_state=42), True),
        ("SVM", "svm", SVC(kernel="rbf", random_state=42), True),
        # max_depth=4: depths 4 and 5 tied in 5-fold cross-validation, so we keep
        # the smaller, simpler tree (see Decision_Tree notebook)
        ("Decision Tree", "decision_tree", DecisionTreeClassifier(max_depth=4, random_state=42), False),
        ("Naive Bayes", "naive_bayes", GaussianNB(), False),
        ("Random Forest", "random_forest", RandomForestClassifier(n_estimators=100, random_state=42), False),
    ]

    for name, prefix, model, needs_scaling in models:
        if needs_scaling:
            model.fit(X_train_scaled, y_train)
            accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
            # Save the scaler too, because new data must be scaled the same way
            joblib.dump(scaler, MODELS_DIR / f"{prefix}_scaler.pkl")
        else:
            model.fit(X_train, y_train)
            accuracy = accuracy_score(y_test, model.predict(X_test))

        joblib.dump(model, MODELS_DIR / f"{prefix}_model.pkl")
        joblib.dump(columns, MODELS_DIR / f"{prefix}_columns.pkl")

        print(f"{name:20s} accuracy: {accuracy * 100:.2f} %")

    print("\nAll 8 models trained and saved in the 'models' folder!")
