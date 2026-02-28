"""
backend/main.py
FastAPI — servidor principal de +Statistics
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import pandas as pd
import numpy as np
import io
import json
from datetime import datetime
from typing import Optional

from core.signal_processing import preprocess_accelerometer, detect_steps
from core.biomechanics import analyze_session, SessionMetrics

app = FastAPI(
    title="+Statistics API",
    description="Running biomechanics analysis backend",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción: lista de dominios permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE DISPOSITIVOS
# ─────────────────────────────────────────────
DEVICE_POSITIONS = {
    "Pecho / Arnés":        {"gss_good": (0, 3),  "gss_warn": (3, 6)},
    "Brazo / Muñeca":       {"gss_good": (1, 4),  "gss_warn": (4, 8)},
    "Bolsillo / Cintura":   {"gss_good": (2, 6),  "gss_warn": (6, 10)},
    "Espalda / Canguro":    {"gss_good": (4, 9),  "gss_warn": (9, 13)},
    "Mano (sostenido)":     {"gss_good": (3, 8),  "gss_warn": (8, 14)},
}

# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "app": "+Statistics API", "version": "2.0.0"}


@app.get("/devices")
def get_devices():
    """Lista de posiciones de dispositivo disponibles."""
    return {"devices": list(DEVICE_POSITIONS.keys())}


@app.post("/analyze")
async def analyze(
    accel_file: UploadFile = File(...),
    gps_file: Optional[UploadFile] = File(None),
    device: str = "Espalda / Canguro",
):
    """
    Analiza una sesión de carrera.
    Recibe CSV de acelerómetro (obligatorio) y GPS (opcional).
    """
    try:
        # Leer acelerómetro
        accel_bytes = await accel_file.read()
        accel_df = pd.read_csv(io.StringIO(accel_bytes.decode("utf-8", errors="replace")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error leyendo acelerómetro: {str(e)}")

    # Leer GPS si viene
    gps_df = None
    if gps_file:
        try:
            gps_bytes = await gps_file.read()
            gps_df = pd.read_csv(io.StringIO(gps_bytes.decode("utf-8", errors="replace")))
            gps_df.columns = [c.strip().lower() for c in gps_df.columns]
        except Exception:
            gps_df = None

    # Configuración del dispositivo
    device_config = DEVICE_POSITIONS.get(device, DEVICE_POSITIONS["Espalda / Canguro"])
    device_config = {**device_config, "name": device}

    try:
        # Pipeline de análisis
        accel_clean = preprocess_accelerometer(accel_df)
        peaks, peak_times, peak_values = detect_steps(accel_clean)

        if len(peaks) < 10:
            raise HTTPException(status_code=422, detail="No se detectaron suficientes pisadas. Verifica el archivo CSV.")

        metrics = analyze_session(
            accel=accel_clean,
            peak_times=peak_times,
            peak_values=peak_values,
            gps=gps_df,
            device_config=device_config,
        )

        return _metrics_to_response(metrics)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en análisis: {str(e)}")


@app.post("/analyze/demo")
async def analyze_demo(device: str = "Espalda / Canguro"):
    """Analiza con datos de demostración."""
    from core.signal_processing import preprocess_accelerometer, detect_steps
    import pandas as pd
    import numpy as np

    # Generar datos demo
    duration = 600
    fs = 100
    t = np.linspace(0, duration, duration * fs)
    np.random.seed(42)
    fatigue = np.linspace(1.0, 1.4, len(t))
    cadence_hz = 2.83

    z = np.sin(2 * np.pi * cadence_hz * t) * 0.8 * fatigue + np.random.normal(0, 0.15, len(t)) + 9.81
    x = np.sin(2 * np.pi * cadence_hz * t + np.pi/4) * 0.3 * fatigue + np.random.normal(0, 0.1, len(t))
    y = np.sin(2 * np.pi * cadence_hz * t + np.pi/2) * 0.15 + np.random.normal(0, 0.08, len(t))

    accel_df = pd.DataFrame({"time": t, "x": x, "y": y, "z": z})
    gps_t = np.arange(0, duration, 1)
    gps_df = pd.DataFrame({"time": gps_t, "speed": 3.0 + 0.5 * np.sin(2 * np.pi * gps_t / 120)})

    device_config = {**DEVICE_POSITIONS.get(device, DEVICE_POSITIONS["Espalda / Canguro"]), "name": device}

    accel_clean = preprocess_accelerometer(accel_df)
    peaks, peak_times, peak_values = detect_steps(accel_clean)
    metrics = analyze_session(accel_clean, peak_times, peak_values, gps_df, device_config)

    return _metrics_to_response(metrics)


def _metrics_to_response(m: SessionMetrics) -> dict:
    """Serializa SessionMetrics a dict JSON."""
    return {
        "date": datetime.now().isoformat(),
        "duration": round(m.duration, 1),
        "steps": m.steps,
        "device": m.device,

        # Métricas base
        "rei": m.rei,
        "gss": m.gss,
        "cadence": m.cadence,
        "asymmetry": m.asymmetry,
        "speed": round(m.speed, 2),
        "pace_min_km": m.pace_min_km,
        "gps_available": m.gps_available,

        # Fatigue
        "fi_times": m.fi_times,
        "fi_values": m.fi_values,
        "fatigue_slope": round(m.fatigue_slope, 4),

        # Knee Load Index
        "kli": m.kli,
        "kli_status": m.kli_status,
        "cumulative_load": m.cumulative_load,
        "load_per_step": m.load_per_step,
        "load_rate": m.load_rate,

        # Rangos del dispositivo
        "gss_good": list(m.gss_good),
        "gss_warn": list(m.gss_warn),
    }


@app.post("/analyze/ml")
async def analyze_with_ml(
    accel_file: UploadFile = File(...),
    gps_file: Optional[UploadFile] = File(None),
    device: str = "Espalda / Canguro",
    history_json: str = "[]",
    target_distance_km: float = 10.0,
):
    """
    Analiza sesión + corre los 3 modelos de ML.
    history_json: historial de sesiones anteriores del usuario (JSON string).
    """
    from services.ml_engine import ml_engine

    # Análisis biomecánico base
    accel_bytes = await accel_file.read()
    accel_df = pd.read_csv(io.StringIO(accel_bytes.decode("utf-8", errors="replace")))

    gps_df = None
    if gps_file:
        try:
            gps_bytes = await gps_file.read()
            gps_df = pd.read_csv(io.StringIO(gps_bytes.decode("utf-8", errors="replace")))
            gps_df.columns = [c.strip().lower() for c in gps_df.columns]
        except Exception:
            gps_df = None

    device_config = {**DEVICE_POSITIONS.get(device, DEVICE_POSITIONS["Espalda / Canguro"]), "name": device}

    accel_clean = preprocess_accelerometer(accel_df)
    peaks, peak_times, peak_values = detect_steps(accel_clean)

    if len(peaks) < 10:
        raise HTTPException(status_code=422, detail="No se detectaron suficientes pisadas.")

    metrics = analyze_session(accel_clean, peak_times, peak_values, gps_df, device_config)
    session_data = _metrics_to_response(metrics)

    # Parsear historial
    try:
        history = json.loads(history_json)
    except Exception:
        history = []

    # Correr ML
    ml_results = ml_engine.analyze(
        session=session_data,
        history=history,
        target_distance_km=target_distance_km,
    )

    return {**session_data, "ml": ml_results}


@app.post("/ml/predict")
async def predict_only(
    session_json: str,
    history_json: str = "[]",
    target_distance_km: float = 10.0,
):
    """
    Corre solo los modelos ML sobre datos ya calculados.
    Útil para recalcular predicciones sin re-analizar el CSV.
    """
    from services.ml_engine import ml_engine
    try:
        session = json.loads(session_json)
        history = json.loads(history_json)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"JSON inválido: {e}")

    return ml_engine.analyze(session=session, history=history,
                             target_distance_km=target_distance_km)

