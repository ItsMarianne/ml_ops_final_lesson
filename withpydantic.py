from fastapi import FastAPI
from pydantic import BaseModel
from scalar_fastapi import get_scalar_api_reference

app = FastAPI()

@app.get("/seiya")
def scalar_docs():
    return get_scalar_api_reference()

class Numbers(BaseModel):
    number_1: float
    number_2: float

@app.post("/add_two_numbers")
def add_numbers(payload: Numbers):
    result = payload.number_1 + payload.number_2
    return {"result": result}



