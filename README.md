# Crop Price Prediction API

A FastAPI app that provides 7-day crop price predictions using simulated ARIMA logic.

## Endpoint

GET `/predict?crop=Wheat`

## Deploy

Use on Render.com with start command:

```
uvicorn main:app --host 0.0.0.0 --port 10000
```