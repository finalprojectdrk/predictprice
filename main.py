from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from arima_model import get_prediction

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/predict")
def predict(crop: str = Query(...)):
    return get_prediction(crop)