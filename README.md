# ShopSense — Customer Spending & Value Predictor

Machine Learning Lab Midterm Project · COMSATS University Islamabad

## Business Problem
A food delivery and retail company wants to spend its marketing budget wisely.
Sending a loyalty offer to every customer is expensive, so the company needs to know:

1. **How much will this customer spend in two years?** → Regression
2. **Is this customer High-Value or Regular?** → Classification

A **High-Value Customer** is one who spends **400 or more** in two years (close to the median spending of 397, so the two groups are almost equal in size).

## Dataset
**Customer Personality Analysis** (iFood marketing data) — `marketing_campaign.csv`, **2,240 customers**, 29 columns.

- **Source (Kaggle):** https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis
- **Download copy used (original iFood file on GitHub):** https://raw.githubusercontent.com/nailson/ifood-data-business-analyst-test/master/ml_project1_data.csv

| Column used | Meaning |
|---|---|
| age (from Year_Birth) | Customer age (data collected in 2014) |
| education | Basic, 2n Cycle, Graduation, Master, PhD |
| marital_status | Single, Married, Together, Divorced, Widow |
| income | Yearly household income |
| kids / teens | Small children / teenagers at home |
| recency | Days since last purchase |
| web_visits | Website visits last month |
| **total_spending** | **Target** = sum of money spent on wine, fruit, meat, fish, sweets and gold |

## Preprocessing
| Step | Records |
|---|---|
| Raw data | 2,240 |
| Removed repeated customers (same data, new ID) | −182 |
| Removed rows with missing income | −24 |
| Removed impossible values (age 114–121, income 666,666) | −4 |
| **Clean data** | **2,030** |

- Odd marital values ("Alone", "Absurd", "YOLO") → "Single"
- One-hot encoding for education and marital status (`get_dummies`, `drop_first=True`)
- 80% / 20% train-test split (`random_state=42`, stratified for classification)
- `StandardScaler` for Logistic Regression, KNN, ANN and SVM (income is much larger than the other columns)
- Decision Tree depth and KNN's K checked with 5-fold cross-validation on the training data only

## Models and Results

| Model | Task | Test result |
|---|---|---|
| Linear Regression | Regression | **R² = 0.7189**, MAE = 238.56, RMSE = 303.52 |
| Logistic Regression | Classification | **90.64%** accuracy (best) |
| Decision Tree (depth 4) | Classification | 89.66% |
| ANN (16, 8 neurons) | Classification | 89.41% |
| Random Forest (100 trees) | Classification | 89.16% |
| SVM (RBF kernel) | Classification | 89.16% |
| Gaussian Naive Bayes | Classification | 87.19% |
| KNN (K = 15) | Classification | 85.71% |

Full precision / recall / F1 table: `reports/model_performance.csv`.
5-fold cross-validation puts the top five models within about 1.5% of each other (std about 2%), so they perform about equally well.
**Income** is the strongest factor, followed by **small children at home** and **website visits** (both lower spending).

## Project Structure
```
ML-LAB-MID/
├── marketing_campaign.csv      # dataset
├── train_models.py             # cleans data, trains all 8 models -> models/
├── evaluate_models.py          # tests all models -> reports/
├── regression_model.py         # simple standalone Linear Regression
├── classification_model.py     # simple standalone Logistic Regression
├── main.py                     # FastAPI backend
├── index.html                  # frontend (HTML + CSS + JavaScript)
├── threshold.txt               # High-Value limit (400)
├── model_performance.csv       # copy of the comparison table
├── requirements.txt
├── SHOPSENSE.docx              # report (editable)
├── models/                     # saved .pkl models, scalers and column lists
├── notebooks/                  # EDA + one notebook per model + comparison
└── reports/                    # FINAL REPORT.pdf + performance CSV files
```

## How to Run
1. Open a terminal inside the project folder.
2. (Optional) create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # Mac / Linux
   ```
3. Install the libraries:
   ```
   pip install -r requirements.txt
   ```
4. Train the models:
   ```
   python train_models.py
   ```
5. Evaluate the models:
   ```
   python evaluate_models.py
   ```
6. Start the backend:
   ```
   uvicorn main:app --reload
   ```
7. Open **http://127.0.0.1:8000** in the browser. API documentation: **http://127.0.0.1:8000/docs**

> Run step 4 on your own computer first. Saved `.pkl` files depend on the scikit-learn version, so retraining avoids version warnings.

To open the notebooks: `jupyter notebook` → open the `notebooks` folder.

## API Endpoints
| Method | Endpoint | Returns |
|---|---|---|
| GET | `/` | The website |
| POST | `/predict/regression` | Expected spending |
| POST | `/predict/classification` | Logistic Regression result |
| POST | `/predict/knn` | KNN result |
| POST | `/predict/ann` | ANN result |
| POST | `/predict/svm` | SVM result |
| POST | `/predict/decision-tree` | Decision Tree result |
| POST | `/predict/naive-bayes` | Naive Bayes result |
| POST | `/predict/random-forest` | Random Forest result |
| GET | `/model-performance` | Classifier comparison table |
| GET | `/regression-performance` | R², MAE, RMSE |

Example request body:
```json
{
  "age": 45,
  "education": "Graduation",
  "marital_status": "Married",
  "income": 52000,
  "kids": 0,
  "teens": 1,
  "recency": 49,
  "web_visits": 5
}
```
