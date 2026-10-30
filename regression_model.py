# ============================================================
# Simple Linear Regression script (standalone practice file)
# Predicts how much a customer will spend in 2 years
# Run it with:  python regression_model.py
# ============================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

# 1. Load the dataset and remove repeated customers
data = pd.read_csv("marketing_campaign.csv")
data = data.drop("ID", axis=1).drop_duplicates()

# 2. Create the target: total spending on all products
data["Total_Spending"] = (
    data["MntWines"] + data["MntFruits"] + data["MntMeatProducts"]
    + data["MntFishProducts"] + data["MntSweetProducts"] + data["MntGoldProds"]
)

# 3. Create age from year of birth
data["Age"] = 2014 - data["Year_Birth"]

# 4. Clean the data
data = data.dropna(subset=["Income"])                       # missing income
data = data[(data["Age"] <= 90) & (data["Income"] <= 200000)]  # wrong values

# 5. Select number columns as inputs (X) and the target (y)
X = data[["Age", "Income", "Kidhome", "Teenhome", "Recency", "NumWebVisitsMonth"]]
y = data["Total_Spending"]

# 6. Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 8. Check model performance
predictions = model.predict(X_test)
print("R2 Score:", round(r2_score(y_test, predictions), 4))
print("MAE     :", round(mean_absolute_error(y_test, predictions), 2))

# 9. Show what the model learned (one weight per column)
print("\nEffect of each column on spending:")
for column, weight in zip(X.columns, model.coef_):
    print(f"  {column:20s} {weight:+.4f}")

# 10. Save the trained model
joblib.dump(model, "regression_model.joblib")
print("\nRegression model saved!")
