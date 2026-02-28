"""
core/biomechanics.py
Cálculo de métricas biomecánicas de carrera.
Incluye el nuevo índice de carga en rodilla (KLI).
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SessionMetrics:
    """Todas las métricas de una sesión de carrera."""
    # Básicas
    rei: float = 0.0           # Running Economy Index (0-100)
    gss: float = 0.0           # Ground Shock Score (m/s²)
    cadence: float = 0.0       # Cadencia media (ppm)
    asymmetry: float = 0.0     # Asimetría izq/der (%)
    speed: float = 0.0         # Velocidad media (m/s)
    duration: float = 0.0      # Duración (min)
    steps: int = 0             # Total pisadas

    # Fatigue
    fi_times: list = field(default_factory=list)
    fi_values: list = field(default_factory=list)
    fatigue_slope: float = 0.0

    # Carga en rodilla (nuevo)
    kli: float = 0.0           # Knee Load Index (0-100, mayor = más riesgo)
    kli_status: str = "OK"     # "OK", "WARNING", "STOP"
    cumulative_load: float = 0.0  # Carga acumulada absoluta
    load_per_step: float = 0.0    # Carga promedio por pisada
    load_rate: float = 0.0        # Tasa de carga (m/s² por segundo)

    # GPS
    gps_available: bool = False
    pace_min_km: float = 0.0

    # Configuración
    device: str = "Espalda / Canguro"
    gss_good: tuple = (4, 9)
    gss_warn: tuple = (9, 13)


def running_economy_index(accel: pd.DataFrame, peak_values: np.ndarray) -> float:
    """REI: combina varianza vertical + consistencia de impacto. Rango 0-100."""
    z = accel["z_filt"].values - 9.81 if "z_filt" in accel.columns else accel["z"].values - 9.81

    signal_range = np.percentile(np.abs(z), 95) or 1.0
    vertical_var = np.var(z) / (signal_range ** 2)
    score_var = max(0.0, 1.0 - vertical_var * 3)

    if len(peak_values) > 4:
        cv = np.std(peak_values) / (np.mean(np.abs(peak_values)) + 1e-6)
        score_cv = max(0.0, 1.0 - cv * 2)
    else:
        score_cv = 0.5

    return round((score_var * 0.6 + score_cv * 0.4) * 100, 1)


def ground_shock_score(peak_values: np.ndarray) -> tuple:
    """GSS: impacto medio (m/s²). Devuelve (gss, mean, max)."""
    mean_i = float(np.mean(np.abs(peak_values)))
    max_i = float(np.max(np.abs(peak_values)))
    return round(mean_i, 2), mean_i, max_i


def cadence_and_asymmetry(peak_times: np.ndarray) -> tuple:
    """Cadencia (ppm) y asimetría (%) filtradas fisiológicamente."""
    if len(peak_times) < 4:
        return 0.0, 0.0

    intervals = np.diff(peak_times)
    valid = (intervals >= 0.25) & (intervals <= 1.0)
    clean = intervals[valid] if valid.sum() >= 2 else intervals

    median_iv = np.median(clean)
    if median_iv <= 0:
        return 0.0, 0.0

    cadence = round(60.0 / median_iv, 1)

    left, right = clean[0::2], clean[1::2]
    n = min(len(left), len(right))
    asymmetry = 0.0
    if n > 0:
        asymmetry = (abs(np.mean(left[:n]) - np.mean(right[:n])) / median_iv) * 100

    return cadence, round(asymmetry, 2)


def fatigue_index(accel: pd.DataFrame, peak_times: np.ndarray,
                  peak_values: np.ndarray, window_minutes: int = 2) -> tuple:
    """Índice de fatiga por ventana de tiempo."""
    total_time = accel["time"].max()
    window = window_minutes * 60
    fi_times, fi_values = [], []

    for w_start in np.arange(0, min(total_time, 3600), window):
        mask = (peak_times >= w_start) & (peak_times < w_start + window)
        wp = peak_values[mask]
        if len(wp) < 4:
            continue
        fi = float(np.mean(np.abs(wp)) * 0.6 + np.std(wp) * 0.4)
        fi_times.append(float(w_start / 60))
        fi_values.append(round(fi, 3))

    return fi_times, fi_values


def knee_load_index(
    peak_values: np.ndarray,
    peak_times: np.ndarray,
    cadence: float,
    asymmetry: float,
    gss: float,
    gss_warn: tuple,
    duration_min: float,
    user_profile: Optional[dict] = None
) -> dict:
    """
    Knee Load Index (KLI) — índice de carga en rodilla.

    Modelo basado en factores de riesgo biomecánicos conocidos:
    - Impacto por pisada (peak_values)
    - Tasa de carga (load rate) — cuán rápido aumenta la fuerza
    - Cadencia baja → mayor tiempo de contacto → más carga
    - Asimetría → sobrecarga de un lado
    - Carga acumulada total de la sesión
    - Historial personal (si disponible)

    Retorna dict con KLI (0-100), status, y desglose.
    """
    if len(peak_values) < 4 or duration_min <= 0:
        return {"kli": 0.0, "status": "OK", "cumulative_load": 0.0,
                "load_per_step": 0.0, "load_rate": 0.0, "factors": {}}

    impacts = np.abs(peak_values)
    mean_impact = float(np.mean(impacts))
    total_steps = len(peak_values)

    # ── Factor 1: Carga acumulada (impacto × pasos) ──
    cumulative_load = float(mean_impact * total_steps)
    # Normalizar: ~10,000 pasos × 5 m/s² = 50,000 unidades base (score 50)
    load_score = min(100.0, (cumulative_load / 50_000) * 50)

    # ── Factor 2: Load Rate (tasa de carga) ──
    # Estimamos la velocidad de impacto = variabilidad de los peaks
    if len(peak_times) > 1:
        impact_diff = np.abs(np.diff(impacts))
        time_diff = np.abs(np.diff(peak_times))
        rates = impact_diff / (time_diff + 1e-6)
        load_rate = float(np.mean(rates))
    else:
        load_rate = 0.0
    rate_score = min(30.0, load_rate * 10)

    # ── Factor 3: Penalización por cadencia baja ──
    # Cadencia <160 → mayor tiempo en suelo → más carga por pisada
    cadence_penalty = 0.0
    if cadence > 0:
        if cadence < 155:
            cadence_penalty = 15.0
        elif cadence < 165:
            cadence_penalty = 8.0
        elif cadence < 170:
            cadence_penalty = 3.0

    # ── Factor 4: Penalización por asimetría ──
    # Asimetría >10% indica sobrecarga unilateral
    asym_penalty = 0.0
    if asymmetry > 10:
        asym_penalty = 12.0
    elif asymmetry > 5:
        asym_penalty = 5.0

    # ── Factor 5: GSS relativo al dispositivo ──
    gss_threshold = gss_warn[1]
    gss_penalty = 0.0
    if gss > gss_threshold:
        gss_penalty = min(15.0, ((gss - gss_threshold) / gss_threshold) * 15)

    # ── KLI compuesto ──
    kli = load_score + rate_score + cadence_penalty + asym_penalty + gss_penalty
    kli = round(min(100.0, kli), 1)

    # ── Status ──
    if kli >= 70:
        status = "STOP"      # Alto riesgo — parar o reducir
    elif kli >= 45:
        status = "WARNING"   # Precaución — monitorear
    else:
        status = "OK"

    # ── Umbral personal (si hay historial) ──
    personal_threshold = None
    if user_profile and "kli_history" in user_profile:
        hist = user_profile["kli_history"]
        if len(hist) >= 3:
            personal_threshold = float(np.mean(hist[-5:]) + np.std(hist[-5:]) * 1.5)
            if kli > personal_threshold:
                status = max(status, "WARNING")  # Supera su propio umbral

    return {
        "kli": kli,
        "status": status,
        "cumulative_load": round(cumulative_load, 1),
        "load_per_step": round(mean_impact, 3),
        "load_rate": round(load_rate, 4),
        "personal_threshold": personal_threshold,
        "factors": {
            "load_score": round(load_score, 1),
            "rate_score": round(rate_score, 1),
            "cadence_penalty": cadence_penalty,
            "asym_penalty": asym_penalty,
            "gss_penalty": round(gss_penalty, 1),
        }
    }


def analyze_session(
    accel: pd.DataFrame,
    peak_times: np.ndarray,
    peak_values: np.ndarray,
    gps: Optional[pd.DataFrame],
    device_config: dict,
    user_profile: Optional[dict] = None
) -> SessionMetrics:
    """
    Función principal: calcula todas las métricas de una sesión.
    Recibe datos ya preprocesados.
    """
    m = SessionMetrics()
    m.device = device_config.get("name", "Desconocido")
    m.gss_good = device_config.get("gss_good", (4, 9))
    m.gss_warn = device_config.get("gss_warn", (9, 13))
    m.steps = len(peak_times)
    m.duration = float(accel["time"].max() / 60)

    # Métricas base
    m.rei = running_economy_index(accel, peak_values)
    m.gss, _, _ = ground_shock_score(peak_values)
    m.cadence, m.asymmetry = cadence_and_asymmetry(peak_times)
    m.fi_times, m.fi_values = fatigue_index(accel, peak_times, peak_values)

    if len(m.fi_values) >= 2:
        m.fatigue_slope = float(np.polyfit(m.fi_times, m.fi_values, 1)[0])

    # Velocidad
    if gps is not None and "speed" in gps.columns:
        m.speed = float(np.mean(gps["speed"].values))
        m.gps_available = True
        if m.speed > 0:
            m.pace_min_km = round((1000 / m.speed) / 60, 2)
    elif m.steps > 0 and m.duration > 0:
        m.speed = round(m.steps / 2 / m.duration / 60, 2)

    # Knee Load Index
    kli_result = knee_load_index(
        peak_values=peak_values,
        peak_times=peak_times,
        cadence=m.cadence,
        asymmetry=m.asymmetry,
        gss=m.gss,
        gss_warn=m.gss_warn,
        duration_min=m.duration,
        user_profile=user_profile
    )
    m.kli = kli_result["kli"]
    m.kli_status = kli_result["status"]
    m.cumulative_load = kli_result["cumulative_load"]
    m.load_per_step = kli_result["load_per_step"]
    m.load_rate = kli_result["load_rate"]

    return m
