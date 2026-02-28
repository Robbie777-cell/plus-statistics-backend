"""
services/ml_engine.py

Motor de Machine Learning de +Statistics.
3 modelos adaptativos que aprenden del historial personal del usuario.

Arquitectura diseñada para escalar:
- Ahora:  modelos clásicos (scikit-learn) que funcionan desde sesión 1
- Futuro: reemplazar por LSTM/Transformer sin cambiar la interfaz

Autor: +Statistics
"""

import numpy as np
import json
import os
import pickle
from dataclasses import dataclass, field, asdict
from typing import Optional
from datetime import datetime, timedelta


# ══════════════════════════════════════════════════════════
# ESTRUCTURAS DE DATOS
# ══════════════════════════════════════════════════════════

@dataclass
class InjuryRiskResult:
    probability: float        # 0-100 — probabilidad de lesión en 48h
    level: str                # "LOW" | "MODERATE" | "HIGH" | "CRITICAL"
    confidence: float         # 0-1 — confianza del modelo
    main_factor: str          # Factor más determinante
    recommendation: str       # Qué hacer
    contributing_factors: dict = field(default_factory=dict)


@dataclass
class PaceRecommendation:
    pace_min_km: float        # Pace recomendado (min/km)
    pace_range: tuple         # (mínimo, máximo) aceptable
    intensity: str            # "EASY" | "MODERATE" | "TEMPO" | "REST"
    reason: str               # Por qué este pace
    confidence: float         # 0-1


@dataclass
class RecoveryRecommendation:
    days_rest: int            # Días de descanso recomendados
    ready_date: str           # Fecha en que puede correr fuerte otra vez
    recovery_score: float     # 0-100 (100 = completamente recuperado)
    load_status: str          # "UNDERLOAD" | "OPTIMAL" | "OVERLOAD" | "DANGER"
    weekly_load: float        # Carga acumulada esta semana
    chronic_load: float       # Carga crónica (4 semanas)
    acute_chronic_ratio: float  # ATL/CTL — el ratio más usado en ciencia deportiva
    suggestion: str


# ══════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════

def extract_features(session: dict, history: list) -> dict:
    """
    Extrae features de una sesión + historial para los modelos.
    Funciona desde sesión 1 (imputa valores faltantes con defaults seguros).
    """
    now = datetime.now()

    # ── Features de la sesión actual ──
    f = {
        "kli":              float(session.get("kli", 0)),
        "gss":              float(session.get("gss", 0)),
        "asymmetry":        float(session.get("asymmetry", 0)),
        "cadence":          float(session.get("cadence", 170)),
        "fatigue_slope":    float(session.get("fatigue_slope", 0)),
        "rei":              float(session.get("rei", 50)),
        "duration_min":     float(session.get("duration", 0)),
        "speed":            float(session.get("speed", 0)),
        "steps":            float(session.get("steps", 0)),
        "load_per_step":    float(session.get("load_per_step", 0)),
        "load_rate":        float(session.get("load_rate", 0)),
        "cumulative_load":  float(session.get("cumulative_load", 0)),
    }

    # ── Features del historial ──
    if len(history) == 0:
        f.update({
            "sessions_last_7d": 0,
            "sessions_last_14d": 0,
            "days_since_last": 99,
            "avg_kli_7d": f["kli"],
            "avg_duration_7d": f["duration_min"],
            "total_load_7d": f["cumulative_load"],
            "total_load_28d": f["cumulative_load"],
            "max_kli_7d": f["kli"],
            "trend_rei": 0.0,
            "trend_kli": 0.0,
            "trend_asymmetry": 0.0,
            "pace_7d_avg": 1000 / max(f["speed"], 0.1) / 60,
            "consecutive_hard_days": 0,
        })
        return f

    # Parsear fechas del historial
    def parse_date(s):
        for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M"):
            try:
                return datetime.strptime(s[:19], fmt[:len(s[:19])])
            except Exception:
                pass
        return now - timedelta(days=999)

    dated = [(parse_date(s.get("date", "")), s) for s in history]
    dated.sort(key=lambda x: x[0], reverse=True)

    last_7d  = [(d, s) for d, s in dated if (now - d).days <= 7]
    last_14d = [(d, s) for d, s in dated if (now - d).days <= 14]
    last_28d = [(d, s) for d, s in dated if (now - d).days <= 28]

    days_since = (now - dated[0][0]).days if dated else 99

    f["sessions_last_7d"]  = len(last_7d)
    f["sessions_last_14d"] = len(last_14d)
    f["days_since_last"]   = days_since

    def avg(lst, key, default=0):
        vals = [s.get(key, default) for _, s in lst if s.get(key) is not None]
        return float(np.mean(vals)) if vals else default

    f["avg_kli_7d"]      = avg(last_7d,  "kli", f["kli"])
    f["avg_duration_7d"] = avg(last_7d,  "duration", f["duration_min"])
    f["max_kli_7d"]      = max((s.get("kli", 0) for _, s in last_7d), default=f["kli"])
    f["total_load_7d"]   = sum(s.get("cumulative_load", 0) for _, s in last_7d)
    f["total_load_28d"]  = sum(s.get("cumulative_load", 0) for _, s in last_28d)
    f["pace_7d_avg"]     = avg(last_7d, "pace_min_km", 1000 / max(f["speed"], 0.1) / 60)

    # Tendencias (si hay suficiente historia)
    if len(last_7d) >= 3:
        kli_vals = [s.get("kli", 0)       for _, s in last_7d[:5]]
        rei_vals = [s.get("rei", 50)       for _, s in last_7d[:5]]
        asy_vals = [s.get("asymmetry", 0)  for _, s in last_7d[:5]]
        x = np.arange(len(kli_vals))
        f["trend_kli"]       = float(np.polyfit(x, kli_vals[::-1], 1)[0])
        f["trend_rei"]       = float(np.polyfit(x, rei_vals[::-1], 1)[0])
        f["trend_asymmetry"] = float(np.polyfit(x, asy_vals[::-1], 1)[0])
    else:
        f["trend_kli"] = f["trend_rei"] = f["trend_asymmetry"] = 0.0

    # Días consecutivos con KLI alto (>50)
    consecutive = 0
    for d, s in dated:
        if s.get("kli", 0) > 50:
            consecutive += 1
        else:
            break
    f["consecutive_hard_days"] = consecutive

    return f


# ══════════════════════════════════════════════════════════
# MODELO 1: RIESGO DE LESIÓN
# ══════════════════════════════════════════════════════════

class InjuryRiskModel:
    """
    Predice probabilidad de lesión en las próximas 48h.

    Fase 1 — Modelo basado en reglas biomecánicas validadas en literatura:
      Gabbett (2016) — Acute:Chronic Workload Ratio
      Dye (1996) — Envelope of Function (rodilla)
      Nielsen (2014) — Cadencia y carga

    Fase 2 — Con suficiente historial (>30 sesiones) se reemplaza
    automáticamente por Gradient Boosting entrenado en datos personales.
    """

    MIN_SESSIONS_FOR_ML = 30

    def predict(self, features: dict, history: list) -> InjuryRiskResult:
        if len(history) >= self.MIN_SESSIONS_FOR_ML:
            return self._predict_ml(features, history)
        return self._predict_rules(features, history)

    def _predict_rules(self, f: dict, history: list) -> InjuryRiskResult:
        """Modelo de reglas biomecánicas — funciona desde sesión 1."""
        risk = 0.0
        factors = {}

        # Factor 1: KLI actual (peso 30%)
        kli_risk = f["kli"] / 100 * 30
        risk += kli_risk
        factors["kli"] = round(kli_risk, 1)

        # Factor 2: Tendencia KLI creciente (peso 20%)
        if f["trend_kli"] > 2:
            trend_risk = min(20, f["trend_kli"] * 5)
            risk += trend_risk
            factors["kli_trend"] = round(trend_risk, 1)

        # Factor 3: Días consecutivos con alta carga (peso 20%)
        consec_risk = min(20, f["consecutive_hard_days"] * 6)
        risk += consec_risk
        if consec_risk > 0:
            factors["consecutive_load"] = round(consec_risk, 1)

        # Factor 4: Asimetría (peso 15%)
        if f["asymmetry"] > 10:
            asym_risk = min(15, (f["asymmetry"] - 5) * 1.5)
            risk += asym_risk
            factors["asymmetry"] = round(asym_risk, 1)

        # Factor 5: Carga semanal muy alta vs crónica (ACR) (peso 15%)
        if f["total_load_28d"] > 0:
            acr = f["total_load_7d"] / (f["total_load_28d"] / 4 + 1e-6)
            if acr > 1.5:
                acr_risk = min(15, (acr - 1.5) * 15)
                risk += acr_risk
                factors["acute_chronic_ratio"] = round(acr_risk, 1)

        risk = min(100.0, round(risk, 1))

        # Nivel
        if risk >= 70:
            level = "CRITICAL"
        elif risk >= 50:
            level = "HIGH"
        elif risk >= 30:
            level = "MODERATE"
        else:
            level = "LOW"

        # Factor principal
        main_factor = max(factors, key=factors.get) if factors else "none"
        factor_labels = {
            "kli": "carga en rodilla",
            "kli_trend": "aumento progresivo de carga",
            "consecutive_load": "días consecutivos de alta carga",
            "asymmetry": "asimetría entre piernas",
            "acute_chronic_ratio": "aumento brusco de volumen semanal",
        }

        recommendations = {
            "LOW":      "Puedes continuar con tu plan normal.",
            "MODERATE": "Reduce el volumen un 20% en la próxima sesión. Considera un día de recuperación activa.",
            "HIGH":     "Descansa mañana. Si corres, hazlo a ritmo suave máximo 30 min. Revisa técnica de pisada.",
            "CRITICAL": "Para ahora. Tu cuerpo está en zona de riesgo real. Descansa 2-3 días antes de retomar.",
        }

        # Confianza aumenta con el historial
        confidence = min(0.85, 0.4 + len(history) * 0.015)

        return InjuryRiskResult(
            probability=risk,
            level=level,
            confidence=round(confidence, 2),
            main_factor=factor_labels.get(main_factor, main_factor),
            recommendation=recommendations[level],
            contributing_factors=factors,
        )

    def _predict_ml(self, features: dict, history: list) -> InjuryRiskResult:
        """Gradient Boosting entrenado en historial personal — activa con 30+ sesiones."""
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            from sklearn.preprocessing import StandardScaler

            feature_keys = [
                "kli", "gss", "asymmetry", "cadence", "fatigue_slope",
                "rei", "duration_min", "cumulative_load", "load_rate",
                "sessions_last_7d", "days_since_last", "avg_kli_7d",
                "total_load_7d", "total_load_28d", "max_kli_7d",
                "trend_kli", "trend_asymmetry", "consecutive_hard_days",
            ]

            # Construir dataset: X = features de sesión N, y = KLI de sesión N+1
            # (proxy de lesión futura)
            X, y = [], []
            history_sorted = sorted(history, key=lambda s: s.get("date", ""))

            for i in range(len(history_sorted) - 1):
                prev_history = history_sorted[:i]
                f_i = extract_features(history_sorted[i], prev_history)
                X.append([f_i.get(k, 0) for k in feature_keys])
                next_kli = history_sorted[i + 1].get("kli", 0)
                y.append(next_kli)

            if len(X) < 5:
                return self._predict_rules(features, history)

            X, y = np.array(X), np.array(y)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
            model.fit(X_scaled, y)

            x_new = np.array([[features.get(k, 0) for k in feature_keys]])
            predicted_next_kli = float(model.predict(scaler.transform(x_new))[0])
            risk = min(100.0, round(predicted_next_kli, 1))

            # Importancia de features
            importances = dict(zip(feature_keys, model.feature_importances_))
            main_factor = max(importances, key=importances.get)

            if risk >= 70:
                level, rec = "CRITICAL", "Para ahora. Riesgo real basado en TU historial personal."
            elif risk >= 50:
                level, rec = "HIGH", "Reduce carga mañana. Tu patrón histórico indica riesgo."
            elif risk >= 30:
                level, rec = "MODERATE", "Precaución. Modera el ritmo en la próxima sesión."
            else:
                level, rec = "LOW", "Tu historial indica que estás en zona segura."

            return InjuryRiskResult(
                probability=risk,
                level=level,
                confidence=0.85,
                main_factor=main_factor.replace("_", " "),
                recommendation=rec,
                contributing_factors={k: round(v, 3) for k, v in
                                       sorted(importances.items(), key=lambda x: -x[1])[:5]},
            )

        except ImportError:
            return self._predict_rules(features, history)


# ══════════════════════════════════════════════════════════
# MODELO 2: PACE ÓPTIMO
# ══════════════════════════════════════════════════════════

class PaceRecommendationModel:
    """
    Recomienda el pace óptimo para la próxima sesión
    basado en carga acumulada, recuperación y objetivos.
    """

    def predict(self, features: dict, history: list,
                target_distance_km: float = 10.0) -> PaceRecommendation:

        recovery = self._estimate_recovery(features, history)
        base_pace = self._get_base_pace(features, history)

        # Ajustar pace según nivel de recuperación
        if recovery >= 80:
            multiplier, intensity = 1.0,  "TEMPO"
            reason = "Estás bien recuperado. Puedes correr a tu ritmo objetivo."
        elif recovery >= 60:
            multiplier, intensity = 1.07, "MODERATE"
            reason = "Recuperación parcial. Corre 7% más lento que tu pace base."
        elif recovery >= 40:
            multiplier, intensity = 1.15, "EASY"
            reason = "Fatiga acumulada. Sesión suave de regeneración."
        else:
            multiplier, intensity = 1.0,  "REST"
            reason = "Tu cuerpo necesita descanso. Considera no correr hoy."

        pace = round(base_pace * multiplier, 2)
        pace_range = (round(pace * 0.97, 2), round(pace * 1.08, 2))

        # Ajuste por distancia objetivo
        if target_distance_km > 15 and recovery < 70:
            pace = round(pace * 1.05, 2)
            reason += " Distancia larga → pace más conservador."

        return PaceRecommendation(
            pace_min_km=pace,
            pace_range=pace_range,
            intensity=intensity,
            reason=reason,
            confidence=round(min(0.9, 0.4 + len(history) * 0.02), 2),
        )

    def _estimate_recovery(self, f: dict, history: list) -> float:
        """Estima % de recuperación (0-100)."""
        recovery = 100.0

        # Penalizar por días sin descanso
        days_since = f.get("days_since_last", 99)
        if days_since == 0:
            recovery -= 20  # corrió hoy
        elif days_since == 1:
            recovery -= 10  # corrió ayer

        # Penalizar por carga acumulada semana
        load_7d = f.get("total_load_7d", 0)
        load_28d = f.get("total_load_28d", 0)
        if load_28d > 0:
            acr = load_7d / (load_28d / 4 + 1e-6)
            if acr > 1.3:
                recovery -= min(40, (acr - 1.0) * 30)

        # Penalizar por KLI alto reciente
        avg_kli = f.get("avg_kli_7d", 0)
        if avg_kli > 60:
            recovery -= 20
        elif avg_kli > 40:
            recovery -= 10

        return max(0.0, min(100.0, round(recovery, 1)))

    def _get_base_pace(self, f: dict, history: list) -> float:
        """Pace base del corredor en min/km."""
        if f.get("pace_7d_avg", 0) > 0:
            return f["pace_7d_avg"]
        speed = f.get("speed", 0)
        if speed > 0:
            return round((1000 / speed) / 60, 2)
        return 6.0  # default conservador


# ══════════════════════════════════════════════════════════
# MODELO 3: RECUPERACIÓN (ATL/CTL)
# ══════════════════════════════════════════════════════════

class RecoveryModel:
    """
    Modelo de carga-recuperación basado en el marco ATL/CTL
    (Acute Training Load / Chronic Training Load).

    ATL = carga aguda (últimos 7 días) — "fatiga"
    CTL = carga crónica (últimos 28 días) — "fitness"
    ACR = ATL/CTL — el ratio de oro en ciencia del deporte

    ACR óptimo: 0.8-1.3
    ACR >1.5 → zona de riesgo de lesión (Gabbett, 2016)
    """

    def predict(self, features: dict, history: list) -> RecoveryRecommendation:
        total_7d  = features.get("total_load_7d", features.get("cumulative_load", 0))
        total_28d = features.get("total_load_28d", total_7d * 4)

        atl = total_7d / 7     # carga diaria promedio últimos 7 días
        ctl = total_28d / 28   # carga diaria promedio últimos 28 días

        acr = atl / (ctl + 1e-6)

        # Recovery score
        if 0.8 <= acr <= 1.3:
            recovery_score = 85.0
            load_status = "OPTIMAL"
            days_rest = 1
            suggestion = "Carga óptima. Continúa con tu plan. Un día de descanso cada 3-4 sesiones."
        elif acr < 0.8:
            recovery_score = 95.0
            load_status = "UNDERLOAD"
            days_rest = 0
            suggestion = "Puedes aumentar el volumen un 10-15% esta semana."
        elif acr <= 1.5:
            recovery_score = 60.0
            load_status = "OVERLOAD"
            days_rest = 2
            suggestion = "Semana cargada. Descansa 2 días o haz sesiones muy suaves."
        else:
            recovery_score = 30.0
            load_status = "DANGER"
            days_rest = 3
            suggestion = "Carga peligrosamente alta. Descansa 3 días mínimo antes de volver a correr."

        # Ajuste fino por KLI
        kli = features.get("kli", 0)
        if kli > 70 and days_rest < 3:
            days_rest = 3
            suggestion += " Tu índice de carga en rodilla también indica descanso urgente."
        elif kli > 50 and days_rest < 2:
            days_rest = 2

        ready_date = (datetime.now() + timedelta(days=days_rest)).strftime("%A %d de %B")

        return RecoveryRecommendation(
            days_rest=days_rest,
            ready_date=ready_date,
            recovery_score=round(recovery_score, 1),
            load_status=load_status,
            weekly_load=round(total_7d, 1),
            chronic_load=round(total_28d / 4, 1),
            acute_chronic_ratio=round(acr, 2),
            suggestion=suggestion,
        )


# ══════════════════════════════════════════════════════════
# INTERFAZ UNIFICADA
# ══════════════════════════════════════════════════════════

class MLEngine:
    """
    Punto de entrada único para todos los modelos de ML.
    Uso:
        engine = MLEngine()
        result = engine.analyze(session_data, user_history)
    """

    def __init__(self):
        self.injury_model   = InjuryRiskModel()
        self.pace_model     = PaceRecommendationModel()
        self.recovery_model = RecoveryModel()

    def analyze(self, session: dict, history: list,
                target_distance_km: float = 10.0) -> dict:
        """
        Corre los 3 modelos y devuelve resultados unificados.
        """
        features = extract_features(session, history)

        injury     = self.injury_model.predict(features, history)
        pace       = self.pace_model.predict(features, history, target_distance_km)
        recovery   = self.recovery_model.predict(features, history)

        # Alerta máxima (el más urgente gana)
        alert_priority = {"LOW": 0, "OPTIMAL": 0, "UNDERLOAD": 0,
                          "MODERATE": 1, "OVERLOAD": 2,
                          "HIGH": 3, "DANGER": 3, "CRITICAL": 4}
        top_alert = max(
            injury.level,
            recovery.load_status,
            key=lambda x: alert_priority.get(x, 0)
        )

        return {
            "ml_version": "1.0-rules" if len(history) < 30 else "1.0-gbm",
            "sessions_analyzed": len(history),
            "sessions_until_ml": max(0, 30 - len(history)),

            "injury_risk": {
                "probability":   injury.probability,
                "level":         injury.level,
                "confidence":    injury.confidence,
                "main_factor":   injury.main_factor,
                "recommendation": injury.recommendation,
                "factors":       injury.contributing_factors,
            },

            "pace_recommendation": {
                "pace_min_km":  pace.pace_min_km,
                "pace_range":   pace.pace_range,
                "intensity":    pace.intensity,
                "reason":       pace.reason,
                "confidence":   pace.confidence,
            },

            "recovery": {
                "days_rest":           recovery.days_rest,
                "ready_date":          recovery.ready_date,
                "recovery_score":      recovery.recovery_score,
                "load_status":         recovery.load_status,
                "weekly_load":         recovery.weekly_load,
                "chronic_load":        recovery.chronic_load,
                "acute_chronic_ratio": recovery.acute_chronic_ratio,
                "suggestion":          recovery.suggestion,
            },

            "top_alert": top_alert,
        }


# ══════════════════════════════════════════════════════════
# INSTANCIA GLOBAL
# ══════════════════════════════════════════════════════════

ml_engine = MLEngine()
