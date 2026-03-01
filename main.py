from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import io
import os
from core.signal_processing import SignalProcessor
from core.biomechanics import BiomechanicsAnalyzer
from services.ml_engine import MLEngine
from db.database import create_tables
from api.auth_routes import router as auth_router
from api.session_routes import router as sessions_router
from api.strava_routes import router as strava_router

app = FastAPI(
    title="+Statistics API",
    description="Running biomechanics analysis backend",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup():
    create_tables()

app.include_router(auth_router)
app.include_router(sessions_router)
app.include_router(strava_router)

processor = SignalProcessor()
analyzer = BiomechanicsAnalyzer()
ml_engine = MLEngine()

@app.get("/")
def root():
    return {"status": "ok", "app": "+Statistics API", "version": "2.0.0"}

@app.get("/devices")
def get_devices():
    return {
        "positions": ["Espalda / Canguro", "Muñeca", "Tobillo"],
        "supported_sensors": ["Accelerometer", "Gyroscope", "Location"]
    }

@app.post("/analyze")
async def analyze(
    accelerometer: UploadFile = File(...),
    location: UploadFile = File(None)
):
    try:
        acc_content = await accelerometer.read()
        acc_df = pd.read_csv(io.StringIO(acc_content.decode("utf-8")))
        loc_df = None
        if location:
            loc_content = await location.read()
            loc_df = pd.read_csv(io.StringIO(loc_content.decode("utf-8")))
        acc_clean = processor.process_accelerometer(acc_df)
        metrics = analyzer.analyze(acc_clean, loc_df)
        return {"status": "success", "metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

@app.post("/analyze/demo")
def analyze_demo():
    try:
        t = np.linspace(0, 600, 6000)
        acc_df = pd.DataFrame({
            "time": t,
            "x": np.sin(2 * np.pi * 3 * t) * 0.8 + np.random.normal(0, 0.05, len(t)),
            "y": np.cos(2 * np.pi * 3 * t) * 1.2 + np.random.normal(0, 0.05, len(t)),
            "z": np.sin(2 * np.pi * 3 * t + 0.5) * 0.6 + np.random.normal(0, 0.05, len(t))
        })
        loc_df = pd.DataFrame({
            "time": np.linspace(0, 600, 100),
            "velocity": np.random.normal(3.0, 0.3, 100)
        })
        acc_clean = processor.process_accelerometer(acc_df)
        metrics = analyzer.analyze(acc_clean, loc_df)
        return {"status": "demo", "metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/ml")
async def analyze_with_ml(
    accelerometer: UploadFile = File(...),
    location: UploadFile = File(None)
):
    try:
        acc_content = await accelerometer.read()
        acc_df = pd.read_csv(io.StringIO(acc_content.decode("utf-8")))
        loc_df = None
        if location:
            loc_content = await location.read()
            loc_df = pd.read_csv(io.StringIO(loc_content.decode("utf-8")))
        acc_clean = processor.process_accelerometer(acc_df)
        metrics = analyzer.analyze(acc_clean, loc_df)
        ml_results = ml_engine.predict(metrics)
        return {"status": "success", "metrics": metrics, "ml": ml_results}
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

@app.post("/ml/predict")
def predict_only(metrics: dict):
    try:
        ml_results = ml_engine.predict(metrics)
        return {"status": "success", "predictions": ml_results}
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))