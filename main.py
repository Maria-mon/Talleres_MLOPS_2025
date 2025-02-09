from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Cargar el modelo
model = joblib.load('logistic_regression_model.pkl')

#Aplicación fast api
app = FastAPI()

class PredictionInput(BaseModel):
    culmen_length_mm: float
    culmen_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float

# Crar dirección
@app.post("/predict")
def predict(input_data: PredictionInput): 
    data = np.array([[input_data.culmen_length_mm, input_data.culmen_depth_mm,
                      input_data.flipper_length_mm, input_data.body_mass_g]]) 
    prediction = model.predict(data)[0]
    label = "MALE" if prediction == 1 else "FEMALE"
    
    return {"prediction": label}