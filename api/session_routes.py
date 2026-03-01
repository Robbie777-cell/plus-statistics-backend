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
    limit: int = 20,
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
        "avg_rei": round(sum(r.rei for r in records) / len(records), 1),
        "avg_cadence": round(sum(r.cadence for r in records) / len(records), 1),
        "avg_kli": round(sum(r.kli for r in records) / len(records), 1),
        "total_steps": sum(r.steps for r in records),
    }


@router.post("/ml/analyze")
def ml_analyze(
    body: SessionSave,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Corre los 3 modelos ML sobre la sesión + historial del usuario."""
    records = db.query(SessionRecord).filter(
        SessionRecord.user_id == current_user.id
    ).order_by(SessionRecord.id.desc()).limit(30).all()

    history = [_record_to_dict(r) for r in records]
    result = ml_engine.analyze(
        session=body.session_data,
        history=history
    )
    return result


def _record_to_dict(r: SessionRecord) -> dict:
    return {
        "id": r.id,
        "date": r.date,
        "duration": r.duration,
        "steps": r.steps,
        "device": r.device,
        "rei": r.rei,
        "gss": r.gss,
        "cadence": r.cadence,
        "asymmetry": r.asymmetry,
        "speed": r.speed,
        "kli": r.kli,
        "kli_status": r.kli_status,
        "cumulative_load": r.cumulative_load,
        "fatigue_slope": r.fatigue_slope,
        "fi_times": json.loads(r.fi_times) if r.fi_times else [],
        "fi_values": json.loads(r.fi_values) if r.fi_values else [],
    }
