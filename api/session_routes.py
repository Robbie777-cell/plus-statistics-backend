from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session as DBSession
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel
from ..db.database import get_db
from ..db.models import Session as SessionModel, User
from ..services.auth import get_current_user

router = APIRouter(prefix="/sessions", tags=["sessions"])

class SessionCreate(BaseModel):
    session_date: Optional[str] = None
    device_position: Optional[str] = "espalda"
    source: Optional[str] = "manual"
    duration_minutes: Optional[float] = None
    distance_km: Optional[float] = None
    steps: Optional[int] = None
    cadence_avg: Optional[float] = None
    velocity_avg: Optional[float] = None
    ground_shock_avg: Optional[float] = None
    asymmetry: Optional[float] = None
    fatigue_index: Optional[float] = None
    running_economy: Optional[float] = None
    kli: Optional[float] = None
    kli_status: Optional[str] = None
    cumulative_load: Optional[float] = None
    injury_risk: Optional[float] = None
    injury_risk_level: Optional[str] = None
    optimal_pace: Optional[float] = None
    recovery_days: Optional[int] = None
    raw_metrics: Optional[dict] = None

class SessionResponse(BaseModel):
    id: int
    created_at: str
    session_date: Optional[str]
    device_position: Optional[str]
    source: Optional[str]
    duration_minutes: Optional[float]
    distance_km: Optional[float]
    steps: Optional[int]
    cadence_avg: Optional[float]
    velocity_avg: Optional[float]
    ground_shock_avg: Optional[float]
    asymmetry: Optional[float]
    fatigue_index: Optional[float]
    running_economy: Optional[float]
    kli: Optional[float]
    kli_status: Optional[str]
    cumulative_load: Optional[float]
    injury_risk: Optional[float]
    injury_risk_level: Optional[str]
    optimal_pace: Optional[float]
    recovery_days: Optional[int]

    class Config:
        from_attributes = True

@router.post("/save", response_model=SessionResponse)
def save_session(
    data: SessionCreate,
    db: DBSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    session = SessionModel(
        user_id=current_user.id,
        session_date=datetime.fromisoformat(data.session_date) if data.session_date else datetime.utcnow(),
        **{k: v for k, v in data.dict().items() if k != "session_date"}
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return SessionResponse(
        **{c: str(getattr(session, c)) if isinstance(getattr(session, c), datetime) else getattr(session, c)
           for c in ["id", "created_at", "session_date", "device_position", "source",
                     "duration_minutes", "distance_km", "steps", "cadence_avg", "velocity_avg",
                     "ground_shock_avg", "asymmetry", "fatigue_index", "running_economy",
                     "kli", "kli_status", "cumulative_load", "injury_risk", "injury_risk_level",
                     "optimal_pace", "recovery_days"]}
    )

@router.get("/history", response_model=List[SessionResponse])
def get_history(
    limit: int = 50,
    db: DBSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    sessions = db.query(SessionModel)\
        .filter(SessionModel.user_id == current_user.id)\
        .order_by(SessionModel.created_at.desc())\
        .limit(limit).all()

    result = []
    for s in sessions:
        result.append(SessionResponse(
            **{c: str(getattr(s, c)) if isinstance(getattr(s, c), datetime) else getattr(s, c)
               for c in ["id", "created_at", "session_date", "device_position", "source",
                         "duration_minutes", "distance_km", "steps", "cadence_avg", "velocity_avg",
                         "ground_shock_avg", "asymmetry", "fatigue_index", "running_economy",
                         "kli", "kli_status", "cumulative_load", "injury_risk", "injury_risk_level",
                         "optimal_pace", "recovery_days"]}
        ))
    return result

@router.get("/stats")
def get_stats(
    db: DBSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    sessions = db.query(SessionModel)\
        .filter(SessionModel.user_id == current_user.id).all()

    if not sessions:
        return {"total_sessions": 0, "message": "Sin sesiones aún"}

    total = len(sessions)
    avg_cadence = sum(s.cadence_avg for s in sessions if s.cadence_avg) / max(1, sum(1 for s in sessions if s.cadence_avg))
    avg_kli = sum(s.kli for s in sessions if s.kli) / max(1, sum(1 for s in sessions if s.kli))
    high_risk = sum(1 for s in sessions if s.injury_risk_level in ["HIGH", "CRITICAL"])

    return {
        "total_sessions": total,
        "avg_cadence": round(avg_cadence, 1),
        "avg_kli": round(avg_kli, 2),
        "high_risk_sessions": high_risk,
        "ml_ready": total >= 30
    }

@router.delete("/{session_id}")
def delete_session(
    session_id: int,
    db: DBSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    session = db.query(SessionModel)\
        .filter(SessionModel.id == session_id, SessionModel.user_id == current_user.id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")
    db.delete(session)
    db.commit()
    return {"message": "Sesión eliminada"}
