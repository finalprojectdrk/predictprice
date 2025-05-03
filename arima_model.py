from datetime import datetime, timedelta
import numpy as np

def get_prediction(crop):
    today = datetime.today()
    dates = [(today + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]
    prices = [round(100 + i * 3 + np.random.rand(), 2) for i in range(7)]
    return {"crop": crop, "dates": dates, "prices": prices}