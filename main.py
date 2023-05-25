from fastapi import FastAPI
from routes.prediction_model import prediction

app = FastAPI(
    title="API - Backend - PRY20231063",
    description= " Integrantes: Klaus Matthew Mollan Neyra y Juan de Dios Quiroz Rodriguez"
    )
app.include_router(prediction)