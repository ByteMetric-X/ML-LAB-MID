# ============================================================
# Simple Logistic Regression script (standalone practice file)
# Classifies a customer as High-Value (1) or Regular (0)
# Run it with:  python classification_model.py
# ============================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# 1. Load the dataset and remove repeated customers
data = pd.read_csv("marketing_campaign.csv")
data = data.drop("ID", axis=1).drop_duplicates()

# 2. Total spending and age
data["Total_Spending"] = (
    data["MntWines"] + data["MntFruits"] + data["MntMeatProducts"]
    + data["MntFishProducts"] + data["MntSweetProducts"] + data["MntGoldProds"]
)
data["Age"] = 2014 - data["Year_Birth"]

# 3. Clean the data
data = data.dropna(subset=["Income"])
data = data[(data["Age"] <= 90) & (data["Income"] <= 200000)]

# 4. Create the class: 1 = spent 400 or more (High-Value)
data["High_Value"] = (data["Total_Spending"] >= 400).astype(int)

# 5. Inputs (X) and target (y)
X = data[["Age", "Income", "Kidhome", "Teenhome", "Recency", "NumWebVisitsMonth"]]
y = data["High_Value"]

# 6. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7. Scale the data (income is very large compared to other columns)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 8. Create and train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 9. Test the model
predictions = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")

# 10. Save the model and scaler
joblib.dump(model, "classification_model.joblib")
joblib.dump(scaler, "classification_scaler.joblib")
print("Classification model saved!")
