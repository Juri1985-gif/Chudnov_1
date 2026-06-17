from pathlib import Path

import joblib
import uvicorn

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel


app = FastAPI()

with open("rf_fitted.pkl", 'rb') as file:
    model = joblib.load(file)

class ModelRequestData(BaseModel):
    total_square: float
    rooms: float
    floor: float
    city: str
    source: str


class Result(BaseModel):
    result: int

@app.get("/health")
def health():
    return JSONResponse(content={"message": "It's alive!"}, status_code=200)

@app.get("/predict_get", response_model=Result)
def predict_get( total_square: float, rooms: float, floor: float, city: str, source: str):
    input_data = pd.DataFrame([{
        "total_square": float(total_square),
        "rooms": int(rooms),
        "floor": int(floor),
        "city": city,
        "source": source
    }])
    result = model.predict(input_data)[0]
    return Result(result=int(result))


@app.post("/predict_post", response_model=Result)
def preprocess_data(data: ModelRequestData):
    input_data = pd.DataFrame([data.model_dump()])
    result = model.predict(input_data)[0]
    return Result(result=int(result))



if __name__ == "__main__":

    uvicorn.run(app, host="127.0.0.1", port=8000)
