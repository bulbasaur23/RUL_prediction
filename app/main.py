from fastapi import FastAPI
from pydantic import BaseModel
from app.predict import features,predict_rul
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse 

app=FastAPI(title="CMAPSS RUL Predictor")

# SensorInput is a Pydantic Model
SensorInput=type(
    "SensorInput",
    (BaseModel,),
    {"__annotations__":{f:float for f in features}}
)

@app.get("/health")
def health_check():
    return {"status":"ok"}

@app.post("/predict")
def predict(data:SensorInput):
    sensor_dict=data.dict() #pydantic obejct converted to dictionary since our predict_rul expects a dictionary 
    rul=predict_rul(sensor_dict)
    return {"predicted_rul":rul}

@app.get("/features")
def get_features():
    return {"features": features }

@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")

app.mount("/static",StaticFiles(directory="static"),name="static")


