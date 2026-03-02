from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import RedirectResponse
import httpx
import os
from db.database import get_db
from db.models import SessionRecord, StravaToken
from services.auth import get_current_user
from services.strava import (
    save_strava_token,
    get_strava_token,
    refresh_access_token,
    fetch_strava_activities,
    strava_activity_to_session,
)
from sqlalchemy.orm import Session
import json

router = APIRouter(prefix="/strava", tags=["strava"])

CLIENT_ID = os.getenv("STRAVA_CLIENT_ID")
CLIENT_SECRET = os.getenv("STRAVA_CLIENT_SECRET")
REDIRECT_URI = os.getenv("STRAVA_REDIRECT_URI")


@router.get("/connect")
def connect_strava():
    """Inicia el flujo OAuth con Strava."""
    url = (
        f"https://www.strava.com/oauth/authorize"
        f"?client_id={CLIENT_ID}"
        f"&redirect_uri={REDIRECT_URI}"
        f"&response_type=code"
        f"&scope=activity:read_all"
    )
    return RedirectResponse(url)


@router.get("/callback")
async def strava_callback(
    code: str,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user)
):
    """Recibe el código OAuth de Strava y guarda el token."""
    async with httpx.AsyncClient() as client:
        response = await client.post("https://www.strava.com/oauth/token", data={
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "code": code,
            "grant_type": "authorization_code"
        })
    data = response.json()
    if "access_token" not in data:
        raise HTTPException(status_code=400, detail="Error conectando Strava")

    save_strava_token(db, current_user.id, data)

    return {
        "status": "conectado",
        "athlete": data.get("athlete", {}).get("firstname"),
        "message": "Strava conectado exitosamente. Ya puedes importar tus actividades."
    }


@router.get("/status")
def strava_status(
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Verifica si el usuario tiene Strava conectado."""
    token = get_strava_token(db, current_user.id)
    if not token:
        return {"connected": False}
    return {
        "connected": True,
        "athlete_name": token.athlete_name,
        "athlete_id": token.athlete_id,
    }


@router.post("/import")
async def import_activities(
    per_page: int = Query(default=30, le=100),
    page: int = Query(default=1, ge=1),
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Importa actividades de Strava y las guarda como sesiones."""
    token = get_strava_token(db, current_user.id)
    if not token:
        raise HTTPException(status_code=400, detail="Strava no está conectado. Ve a /strava/connect primero.")

    access_token = await refresh_access_token(db, token)
    activities = await fetch_strava_activities(access_token, per_page=per_page, page=page)

    if not activities:
        return {"imported": 0, "message": "No hay actividades nuevas para importar."}

    imported = 0
    skipped = 0

    for activity in activities:
        # Solo importar actividades de running
        if activity.get("type") not in ["Run", "TrailRun", "VirtualRun"]:
            skipped += 1
            continue

        strava_id = str(activity.get("id", ""))

        # Verificar si ya existe
        existing = db.query(SessionRecord).filter(
            SessionRecord.user_id == current_user.id,
            SessionRecord.strava_activity_id == strava_id
        ).first()

        if existing:
            skipped += 1
            continue

        session_data = strava_activity_to_session(activity, current_user.id)
        record = SessionRecord(**session_data)
        db.add(record)
        imported += 1

    db.commit()

    return {
        "imported": imported,
        "skipped": skipped,
        "total_processed": len(activities),
        "message": f"Se importaron {imported} actividades de running."
    }


@router.get("/activities")
async def get_strava_activities(
    per_page: int = Query(default=10, le=50),
    page: int = Query(default=1, ge=1),
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Obtiene actividades directamente desde Strava (sin guardar)."""
    token = get_strava_token(db, current_user.id)
    if not token:
        raise HTTPException(status_code=400, detail="Strava no está conectado.")

    access_token = await refresh_access_token(db, token)
    activities = await fetch_strava_activities(access_token, per_page=per_page, page=page)

    return {
        "activities": [
            {
                "id": a.get("id"),
                "name": a.get("name"),
                "type": a.get("type"),
                "date": a.get("start_date_local"),
                "distance_km": round(a.get("distance", 0) / 1000, 2),
                "duration_min": round(a.get("moving_time", 0) / 60, 1),
                "speed_ms": round(a.get("average_speed", 0), 2),
                "cadence": a.get("average_cadence", 0),
                "heart_rate": a.get("average_heartrate", 0),
            }
            for a in activities
        ],
        "page": page,
        "per_page": per_page
    }


@router.delete("/disconnect")
def disconnect_strava(
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Desconecta Strava del usuario."""
    token = db.query(StravaToken).filter(StravaToken.user_id == current_user.id).first()
    if token:
        db.delete(token)
        db.commit()
    return {"status": "desconectado", "message": "Strava ha sido desconectado."}
