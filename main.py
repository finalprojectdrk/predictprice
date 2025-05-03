from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import requests
from datetime import datetime, timedelta

# FastAPI app setup
app = FastAPI()

# CORS configuration - allow your frontend origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://farmerssmarket.com"],  # Use actual domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model to receive crop name
class CropRequest(BaseModel):
    crop: str

# Fetch crop prices from the government API
def fetch_crop_prices():
    url = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070?api-key=579b464db66ec23bdd0000017704f08e67e4414747189afb9ef2d662&format=json&offset=0&limit=4000"
    response = requests.get(url)
    data = response.json()
    return data['records']

# Preprocess the data to be in time series format
def preprocess_data(records, selected_crop):
    df = pd.DataFrame(records)

    # Check for valid date column
    if 'arrival_date' in df.columns:
        df['date'] = pd.to_datetime(df['arrival_date'], errors='coerce')
    elif 'reported_date' in df.columns:
        df['date'] = pd.to_datetime(df['reported_date'], errors='coerce')
    else:
        raise HTTPException(status_code=400, detail="Missing required column: date")

    df['price'] = pd.to_numeric(df['modal_price'], errors='coerce')
    df = df.dropna(subset=['price', 'date'])

    df = df[df['commodity'].str.lower() == selected_crop.lower()]

    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data found for crop: {selected_crop}")

    df = df.sort_values('date')
    return df[['date', 'price']]

# Train an ARIMA model and predict the next 7 days
def train_arima_model(df):
    model = ARIMA(df['price'], order=(5, 1, 0))  # ARIMA(p,d,q)
    model_fit = model.fit()
    return model_fit

# Forecast prices
def predict_price(model_fit, steps=7):
    forecast = model_fit.forecast(steps)
    return forecast

# FastAPI prediction endpoint
@app.post("/predict")
def predict(request: CropRequest):
    selected_crop = request.crop

    records = fetch_crop_prices()

    df = preprocess_data(records, selected_crop)

    model_fit = train_arima_model(df)

    predicted_prices = predict_price(model_fit, steps=7)

    prediction_result = {
        "crop": selected_crop,
        "prices": predicted_prices.tolist(),
        "dates": [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(7)]
    }

    return prediction_result
