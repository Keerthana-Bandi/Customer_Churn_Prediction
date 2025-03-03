from fastapi import FastAPI
import joblib  
import numpy as np
import pandas as pd
from pydantic import BaseModel
import pickle
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from fastapi.responses import FileResponse

app = FastAPI()

model = joblib.load("DecisionTree.pkl")

class CustomerData(BaseModel):
    # Assuming features like age, tenure, subscription_type, etc.
    Tenure: float
    CityTier: int
    WarehouseToHome: float
    HourSpendOnApp: float
    NumberOfDeviceRegistered: int
    SatisfactionScore: int
    NumberOfAddress: int
    Complain: int
    OrderAmountHikeFromlastYear: float
    CouponUsed: float
    OrderCount: float
    DaySinceLastOrder: float
    CashbackAmount: float
    PreferredLoginDevice_MobilePhone: int
    PreferredPaymentMode_CreditCard: int
    PreferredPaymentMode_DebitCard: int
    PreferredPaymentMode_Ewallet: int
    PreferredPaymentMode_UPI: int
    Gender_Male: int
    PreferedOrderCat_Grocery: int
    PreferedOrderCat_Laptop_Accessory : int
    PreferedOrderCat_Mobile : int
    PreferedOrderCat_Others : int
    MaritalStatus_Married : int
    MaritalStatus_Single : int

        # Load reference data
reference_data = pd.read_csv('reference_data.csv')

# Initialize current data with the same columns as reference data
current_data = pd.DataFrame(columns=reference_data.columns)

@app.post("/predict/")
def predict(data: CustomerData):
    x = data.dict()
    global current_data
    input_data = pd.DataFrame([x])
    
    # Ensure input data has the same columns as reference data
    input_data = input_data[reference_data.columns]
    
    prediction_pro = model.predict_proba(input_data)
    prediction = model.predict(input_data)
    
    # Append input data to current data
    current_data = pd.concat([current_data, input_data], ignore_index=True)
    # Drop any unnamed columns
    current_data.to_csv("current_data.csv")
    current_data = pd.read_csv("current_data.csv")
    current_data = current_data.loc[:, ~current_data.columns.str.contains('^Unnamed')]
    
    
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data)
    html_file_path = "data_drift_report.html"
    report.save_html(html_file_path)
    return {
        "prediction": int(prediction)
    }

@app.get("/download_drift_report")
def download_drift_report():
    return FileResponse("data_drift_report.html", media_type="text/html", filename="data_drift_report.html")

