from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from arima_model import get_prediction

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class CropRequest(BaseModel):
    crop: str

@app.post("/predict")
def predict(request: CropRequest):
    return get_prediction(request.crop)
