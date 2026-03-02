from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from db.database import get_db
from db.models import User, SessionRecord
from services.auth import get_current_user
from services.ml_engine import ml_engine
from pydantic import BaseModel
from typing import Optional
import json
from datetime import datetime

router = APIRouter(prefix="/sessions", tags=["sessions"])


class SessionSave(BaseModel):
    session_data: dict
    device: Optional[str] = "Espalda / Canguro"


@router.post("/save")
def save_session(
    body: SessionSave,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    d = body.session_data
    record = SessionRecord(
        user_id=current_user.id,
        date=d.get("date", datetime.now().isoformat()),
        duration=d.get("duration", 0),
        steps=d.get("steps", 0),
        device=body.device,
        rei=d.get("rei", 0),
        gss=d.get("gss", 0),
        cadence=d.get("cadence", 0),
        asymmetry=d.get("asymmetry", 0),
        speed=d.get("speed", 0),
        kli=d.get("kli", 0),
        kli_status=d.get("kli_status", "OK"),
        cumulative_load=d.get("cumulative_load", 0),
        fatigue_slope=d.get("fatigue_slope", 0),
        fi_times=json.dumps(d.get("fi_times", [])),
        fi_values=json.dumps(d.get("fi_values", [])),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return {"status": "saved", "session_id": record.id}


@router.get("/history")
def get_history(
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    records = (
        db.query(SessionRecord)
        .filter(SessionRecord.user_id == current_user.id)
        .order_by(SessionRecord.id.desc())
        .limit(limit)
        .all()
    )
    return {
        "sessions": [_record_to_dict(r) for r in records],
        "total": len(records)
    }


@router.get("/stats")
def get_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    records = db.query(SessionRecord).filter(
        SessionRecord.user_id == current_user.id
    ).all()
    if not records:
        return {"total_sessions": 0}

    return {
        "total_sessions": len(records),
        "avg_rei": round(sum((r.rei or 0) for r in records) / len(records), 1),
        "avg_cadence": round(sum((r.cadence or 0) for r in records) / len(records), 1),
        "avg_kli": round(sum((r.kli or 0) for r in records) / len(records), 1),
        "total_steps": sum((r.steps or 0) for r in records),
        "total_distance_km": round(sum((r.distance_km or 0) for r in records), 2),
        "avg_pace": _calc_avg_pace(records),
        "injury_risk": _get_latest_risk(records),
        "recovery_score": _get_latest_recovery(records),
        "optimal_pace": _get_latest_pace(records),
    }


@router.post("/ml/analyze")
def ml_analyze(
    body: SessionSave,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    records = db.query(SessionRecord).filter(
        SessionRecord.user_id == current_user.id
    ).order_by(SessionRecord.id.desc()).limit(50).all()
    history = [_record_to_dict(r) for r in records]
    result = ml_engine.analyze(
        session=body.session_data,
        history=history
    )
    return result


@router.post("/ml/analyze-history")
def ml_analyze_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Analiza el historial completo y actualiza predicciones ML en todas las sesiones."""
    records = db.query(SessionRecord).filter(
        SessionRecord.user_id == current_user.id
    ).order_by(SessionRecord.date.asc()).all()

    if not records:
        return {"status": "no_data", "message": "No hay sesiones para analizar"}

    history_dicts = [_record_to_dict(r) for r in records]
    updated = 0

    for i, record in enumerate(records):
        prev_history = history_dicts[:i]
        current_session = history_dicts[i]

        result = ml_engine.analyze(
            session=current_session,
            history=prev_history
        )

        record.injury_risk = result["injury_risk"]["probability"]
        record.injury_risk_level = result["injury_risk"]["level"]
        record.optimal_pace = result["pace_recommendation"]["pace_min_km"]
        record.recovery_days = result["recovery"]["days_rest"]
        updated += 1

    db.commit()

    # Resultado final basado en la sesión más reciente
    last = history_dicts[-1]
    final_result = ml_engine.analyze(session=last, history=history_dicts[:-1])

    return {
        "status": "ok",
        "sessions_analyzed": updated,
        "ml_version": final_result["ml_version"],
        "sessions_until_full_ml": final_result["sessions_until_ml"],
        "latest": {
            "injury_risk": final_result["injury_risk"],
            "pace_recommendation": final_result["pace_recommendation"],
            "recovery": final_result["recovery"],
        }
    }


def _record_to_dict(r: SessionRecord) -> dict:
    duration_sec = r.duration or 0
    distance_km = r.distance_km or 0
    speed = r.speed or 0

    # Calcular pace en min/km
    pace_min_km = 0.0
    if speed > 0:
        pace_min_km = round((1000 / speed) / 60, 2)
    elif duration_sec > 0 and distance_km > 0:
        pace_min_km = round((duration_sec / 60) / distance_km, 2)

    return {
        "id": r.id,
        "date": r.date,
        "source": r.source or "manual",
        "duration": duration_sec,
        "duration_minutes": r.duration_minutes or round(duration_sec / 60, 2),
        "distance_km": distance_km,
        "steps": r.steps or 0,
        "device": r.device or "",
        "rei": r.rei or 0,
        "gss": r.gss or 0,
        "cadence": r.cadence or 0,
        "cadence_avg": r.cadence_avg or 0,
        "asymmetry": r.asymmetry or 0,
        "speed": speed,
        "kli": r.kli or 0,
        "kli_status": r.kli_status or "OK",
        "cumulative_load": r.cumulative_load or 0,
        "fatigue_slope": r.fatigue_slope or 0,
        "heart_rate": r.heart_rate or 0,
        "pace_min_km": pace_min_km,
        "activity_name": r.activity_name or "",
        "activity_type": r.activity_type or "",
        "strava_activity_id": r.strava_activity_id or "",
        "injury_risk": r.injury_risk or 0,
        "injury_risk_level": r.injury_risk_level or "LOW",
        "optimal_pace": r.optimal_pace or 0,
        "recovery_days": r.recovery_days or 0,
        "fi_times": json.loads(r.fi_times) if r.fi_times else [],
        "fi_values": json.loads(r.fi_values) if r.fi_values else [],
    }


def _calc_avg_pace(records) -> float:
    paces = []
    for r in records:
        if (r.speed or 0) > 0:
            paces.append((1000 / r.speed) / 60)
        elif (r.duration or 0) > 0 and (r.distance_km or 0) > 0:
            paces.append((r.duration / 60) / r.distance_km)
    return round(sum(paces) / len(paces), 2) if paces else 0.0


def _get_latest_risk(records) -> dict:
    for r in reversed(records):
        if r.injury_risk is not None:
            return {"probability": r.injury_risk, "level": r.injury_risk_level or "LOW"}
    return {"probability": 0, "level": "LOW"}


def _get_latest_recovery(records) -> float:
    for r in reversed(records):
        if r.recovery_days is not None:
            return max(0, 100 - (r.recovery_days * 20))
    return 95.0


def _get_latest_pace(records) -> float:
    for r in reversed(records):
        if r.optimal_pace and r.optimal_pace > 0:
            return r.optimal_pace
    return 0.0