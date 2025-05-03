from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import requests
from datetime import datetime, timedelta

# FastAPI app setup
app = FastAPI()

# Middleware to handle cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://farmerssmarket.com/"],  # Allow all origins for testing purposes, change to ["https://farmerssmarket.com"] for production
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
    
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to fetch data from API")
    
    data = response.json()
    return data.get('records', [])

# Preprocess the data to be in time series format
def preprocess_data(records, selected_crop):
    # Convert the records into a DataFrame
    df = pd.DataFrame(records)
    
    # Check if required columns exist
    required_columns = ['date', 'modal_price', 'commodity']
    for col in required_columns:
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Missing required column: {col}")
    
    # Convert 'date' column to datetime and 'price' to numeric
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['price'] = pd.to_numeric(df['modal_price'], errors='coerce')
    df = df.dropna(subset=['price', 'date'])  # Drop rows where price or date is missing
    
    # Filter for selected crop (case-insensitive)
    df = df[df['commodity'].str.lower() == selected_crop.lower()]
    
    if df.empty:
        raise HTTPException(status_code=404, detail="No data found for the selected crop")
    
    return df

# Train an ARIMA model and predict the next 7 days
def train_arima_model(df):
    model = ARIMA(df['price'], order=(5, 1, 0))  # ARIMA(p,d,q) values
    model_fit = model.fit()
    return model_fit

def predict_price(model_fit, steps=7):
    forecast = model_fit.forecast(steps)
    return forecast

# FastAPI endpoint for prediction
@app.post("/predict")
def predict(request: CropRequest):
    selected_crop = request.crop

    # Fetch crop prices from the government API
    records = fetch_crop_prices()

    if not records:
        raise HTTPException(status_code=404, detail="No crop price records found")

    # Preprocess the data
    df = preprocess_data(records, selected_crop)

    # Train the ARIMA model
    model_fit = train_arima_model(df)

    # Predict the next 7 days of prices
    predicted_prices = predict_price(model_fit, steps=7)

    # Prepare the response with predicted prices and dates
    prediction_result = {
        "crop": selected_crop,
        "prices": predicted_prices.tolist(),
        "dates": [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(7)]
    }

    return prediction_result
