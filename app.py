from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("loan_default_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")
threshold = joblib.load("threshold.pkl")


@app.get("/")
def home():
    return {"message": "Loan Default Prediction API is running"}


@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    # recreate engineered features
    df["LoanIncomeRatio"] = df["LoanAmount"] / df["Income"]
    df["LoanTermIncomeRatio"] = df["LoanTerm"] / (df["Income"] + 1)
    df["EmploymentIncomeRatio"] = df["MonthsEmployed"] / (df["Age"] + 1)

    df["CreditScoreBand"] = pd.cut(
        df["CreditScore"],
        bins=[300,580,670,740,800,850],
        labels=["Poor","Fair","Good","VeryGood","Excellent"]
    )

    # preprocess
    processed = preprocessor.transform(df)

    # prediction
    prob = model.predict_proba(processed)[:,1][0]

    prediction = int(prob >= threshold)

    return {
        "default_probability": float(prob),
        "prediction": prediction
    }