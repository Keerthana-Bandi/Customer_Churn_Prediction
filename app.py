from fastapi import FastAPI
import joblib  # For loading your trained model
import numpy as np
import pandas as pd
from pydantic import BaseModel
import pickle
# Load the model into memory (ensure your model file is in the same directory as app.py or adjust path)
#model = joblib.load('DecisionTree.pkl')
app = FastAPI()
pickle_in = open("DecisionTree.pkl","rb")
model=pickle.load(pickle_in)

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
@app.post("/predict/")
def predict(data: CustomerData):
    # Preprocess the input data (if necessary)
    input_data = pd.DataFrame([data])
    
    # Use the model to make predictions
    prediction = model.predict(input_data)
    
    # Return prediction result (0 = No churn, 1 = Churn)
    return {"prediction": int(prediction[0])}

# Run the FastAPI app with Uvicorn if this file is executed directly
# if __name__ == '__main__':
#     import uvicorn
#     uvicorn.run(app, host='127.0.0.1', port=8000)
