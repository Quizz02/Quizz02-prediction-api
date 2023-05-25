from fastapi import APIRouter
import joblib
from sklearn.calibration import LabelEncoder
from schemas.prediction import Prediction
import pandas as pd
import numpy as np

prediction = APIRouter()

posts = []

@prediction.get("/predictions")
def get_prediction():
    return posts

@prediction.get("/prediction/{id_}")
def get_prediction_id(id_ : int):
    for post in posts:
        if post["Id"] == id_:
            return post

    return "Data not found"

@prediction.post("/prediction")
def create_data(data_prediction: Prediction):
    modeloArbolReg = joblib.load('routes/DecisionTreeRegressor.pkl')

    le = LabelEncoder()

    tp_animal = pd.Series(np.array([data_prediction.animalType]))
    le.fit(tp_animal)
    data_prediction.p_animal = le.transform(tp_animal)

    le = LabelEncoder() #Coloca 1 como perro y el 0 como estado aceptable, 0 como gato y 1 como grave 

    tp_Estado = pd.Series(np.array([data_prediction.status]))
    le.fit(tp_Estado)
    data_prediction.p_Estado = le.transform(tp_Estado)

    datos_a_predecir = {
        'Tipo_animal': data_prediction.p_animal,
        'Peso': data_prediction.weight,
        'Estado': data_prediction.p_Estado,
        'Día': data_prediction.day,
        'Mes': data_prediction.month,
        'Hora': data_prediction.hour
    }

    datos_a_predecir = pd.DataFrame(datos_a_predecir)

    result = modeloArbolReg.predict(datos_a_predecir)

    final_data = {
        'Id' : data_prediction.id,
        'Tipo_animal': data_prediction.animalType,
        'Peso': data_prediction.weight,
        'Estado': data_prediction.status,
        'Día': data_prediction.day,
        'Mes': data_prediction.month,
        'Hora': data_prediction.hour,
        'Latitud': result[0][0],
        'Longitud': result[0][1]
    }

    posts.append(final_data)
    return final_data