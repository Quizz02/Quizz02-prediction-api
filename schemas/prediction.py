from typing import Optional
from pydantic import BaseModel

class Prediction(BaseModel):
    id : int
    animalType : str
    weight : float
    status : str
    day : int
    month : int
    hour : int
    p_Estado : Optional[int]
    p_animal : Optional[int]