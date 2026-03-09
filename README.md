# Checkpoint-2

## Overview
This project builds an **end-to-end machine learning pipeline** to predict whether a loan applicant is likely to default on a loan. The system includes data preprocessing, feature engineering, model training, evaluation, optimization, and deployment through a **FastAPI prediction API**.

The goal is to help financial institutions **identify high-risk borrowers before approving loans**.

---

# Problem Statement

Financial institutions face significant losses when borrowers fail to repay loans. Accurately predicting loan defaults allows lenders to:

- Reduce financial risk
- Improve credit approval decisions
- Detect high-risk borrowers early

This project builds a machine learning model that predicts **loan default probability based on borrower information**.

---

# Dataset

The dataset contains **255,347 loan records** with borrower financial and demographic attributes.

### Features

| Feature | Description |
|------|------|
| Age | Borrower's age |
| Income | Annual income |
| LoanAmount | Amount of loan requested |
| CreditScore | Borrower's credit score |
| MonthsEmployed | Employment duration |
| NumCreditLines | Number of credit lines |
| InterestRate | Loan interest rate |
| LoanTerm | Loan duration |
| DTIRatio | Debt-to-income ratio |
| Education | Education level |
| EmploymentType | Employment category |
| MaritalStatus | Marital status |
| HasMortgage | Whether borrower has mortgage |
| HasDependents | Whether borrower has dependents |
| LoanPurpose | Purpose of the loan |
| HasCoSigner | Whether loan has co-signer |

### Target Variable

| Column | Meaning |
|------|------|
| Default | 1 = borrower defaulted, 0 = loan repaid |

---

# Project Workflow

## 1. Data Inspection

The dataset was loaded using pandas and inspected to understand:

- Feature structure
- Data types
- Target variable distribution

---

## 2. Class Distribution Analysis

Non-default ≈ 88%
Default ≈ 12%


Because of this imbalance, **accuracy alone is not a reliable metric**, so additional metrics such as recall, F1-score, and ROC-AUC were used.

---

## 3. Feature Engineering

New financial indicators were created to better represent borrower risk.

### Engineered Features

| Feature | Description |
|------|------|
| LoanIncomeRatio | Loan amount relative to borrower income |
| LoanTermIncomeRatio | Loan term relative to income |
| EmploymentIncomeRatio | Employment duration relative to age |
| CreditScoreBand | Credit score categorized into risk bands |

These features help the model capture **financial stress and repayment ability**.

---

## 4. Train-Test Split

The dataset was split into:

Training set: 80%
Test set: 20%


Stratified sampling was used to maintain the same default ratio in both datasets.

---

## 5. Data Preprocessing

A preprocessing pipeline was built using **ColumnTransformer**.

### Numerical Features
Scaled using:
StandardScaler


### Categorical Features
Encoded using:
OneHotEncoder


This ensures that all model inputs are numerical.

---

# Models Trained

Multiple classification algorithms were evaluated.

| Model | Purpose |
|------|------|
| Logistic Regression | Baseline model |
| Decision Tree | Captures nonlinear relationships |
| Random Forest | Ensemble learning approach |
| XGBoost | Gradient boosting model |

---

# Handling Class Imbalance

Because default cases are rare, the XGBoost model used:
scale_pos_weight


This increases the penalty for misclassifying default cases.

---

# Hyperparameter Tuning

The XGBoost model was optimized using:
GridSearchCV

with **3-fold cross-validation**.

### Parameters tuned

- n_estimators
- max_depth
- learning_rate
- subsample
- colsample_bytree

The configuration with the **highest ROC-AUC score** was selected.

---

# Model Evaluation

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

### Final Model Performance

| Metric | Value |
|------|------|
| Accuracy | 0.82 |
| Precision | 0.32 |
| Recall | 0.45 |
| F1 Score | 0.37 |
| ROC-AUC | 0.76 |

The **XGBoost model achieved the best overall performance**.

---

# Threshold Optimization

Instead of using the default classification threshold (0.5), multiple thresholds were evaluated.

The optimal value was found to be:
Threshold = 0.65


This produced the best balance between precision and recall and improved the F1-score.

---

# Feature Importance

The trained XGBoost model identified the most influential features affecting loan default prediction.

Top features included:

- Age
- InterestRate
- Income
- LoanAmount
- MonthsEmployed

These features strongly influence borrower repayment risk.

---

# Model Deployment

The trained model was deployed using **FastAPI** to allow real-time predictions.

### Saved Components

Objects were saved using `joblib`.

| File | Description |
|------|------|
| loan_default_model.pkl | Trained XGBoost model |
| preprocessor.pkl | Data preprocessing pipeline |
| threshold.pkl | Optimized classification threshold |

---

# API Deployment

A FastAPI application was created to serve predictions.

The API performs the following steps:

1. Receives borrower information
2. Recreates engineered features
3. Applies preprocessing pipeline
4. Runs the trained model
5. Applies threshold rule
6. Returns prediction and default probability

---

# Running the API

Start the server:
python -m uvicorn app:app --reload


Open the API interface:
http://127.0.0.1:8000/docs


This provides an interactive interface to test predictions.

---

# Example API Request
  POST /predict
  
Example input:


{
"Age":45,
"Income":60000,
"LoanAmount":120000,
"CreditScore":650,
"MonthsEmployed":24,
"NumCreditLines":3,
"InterestRate":9.5,
"LoanTerm":36,
"DTIRatio":0.4,
"Education":"Bachelor's",
"EmploymentType":"Full-time",
"MaritalStatus":"Married",
"HasMortgage":"Yes",
"HasDependents":"No",
"LoanPurpose":"Home",
"HasCoSigner":"No"
}

{
"default_probability": 0.32,
"prediction": 0
}

Where: 
default_probability = predicted probability that the borrower will default
prediction = final classification based on optimized threshold

# Technologies Used

Python

Pandas

NumPy

Scikit-Learn

XGBoost

Matplotlib

FastAPI

Joblib

# Conclusion

This project demonstrates a complete machine learning pipeline for credit risk prediction, from data preprocessing to deployment as an API service.

The system can be integrated into financial platforms to assist in automated loan approval decisions and reduce lending risk.
