from fastapi import FastAPI
from scalar_fastapi import get_scalar_api_reference
import joblib
from pydantic import BaseModel, ConfigDict
 
app = FastAPI()
 
 
class PredictionRequest(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float
 
 
model = joblib.load("models/best_housing_regressor.joblib")
 
@app.post("/predict")
def predict(features: PredictionRequest):
    rows = [
        [
            features.MedInc,
            features.HouseAge,
            features.AveRooms,
            features.AveBedrms,
            features.Population,
            features.AveOccup,
            features.Latitude,
            features.Longitude,
        ]
    ]
    prediction = model.predict(rows)[0]
 
    return {
        "predicted_median_house_value": prediction,
        "predicted_dollar_value": prediction * 100_000,
    }
 
 
@app.post("/cal_house_val")
def get_scalar_docs():
    return get_scalar_api_reference()
    

 
 