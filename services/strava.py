import httpx
import os
from sqlalchemy.orm import Session
from db.models import StravaToken, SessionRecord
from datetime import datetime
import json

STRAVA_TOKEN_URL = "https://www.strava.com/oauth/token"
STRAVA_ACTIVITIES_URL = "https://www.strava.com/api/v3/athlete/activities"

CLIENT_ID = os.getenv("STRAVA_CLIENT_ID")
CLIENT_SECRET = os.getenv("STRAVA_CLIENT_SECRET")


def save_strava_token(db: Session, user_id: int, token_data: dict) -> None:
    """Guarda o actualiza el token de Strava para un usuario."""
    existing = db.query(StravaToken).filter(StravaToken.user_id == user_id).first()
    if existing:
        existing.access_token = token_data["access_token"]
        existing.refresh_token = token_data.get("refresh_token", "")
        existing.expires_at = token_data.get("expires_at", 0)
        existing.athlete_id = str(token_data.get("athlete", {}).get("id", ""))
        existing.athlete_name = token_data.get("athlete", {}).get("firstname", "")
        existing.updated_at = datetime.utcnow()
    else:
        token = StravaToken(
            user_id=user_id,
            access_token=token_data["access_token"],
            refresh_token=token_data.get("refresh_token", ""),
            expires_at=token_data.get("expires_at", 0),
            athlete_id=str(token_data.get("athlete", {}).get("id", "")),
            athlete_name=token_data.get("athlete", {}).get("firstname", ""),
        )
        db.add(token)
    db.commit()


def get_strava_token(db: Session, user_id: int) -> StravaToken | None:
    """Obtiene el token de Strava de un usuario."""
    return db.query(StravaToken).filter(StravaToken.user_id == user_id).first()


async def refresh_access_token(db: Session, token: StravaToken) -> str:
    """Refresca el access token si expiró."""
    import time
    if token.expires_at > int(time.time()) + 300:
        return token.access_token

    async with httpx.AsyncClient() as client:
        response = await client.post(STRAVA_TOKEN_URL, data={
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "grant_type": "refresh_token",
            "refresh_token": token.refresh_token,
        })
    data = response.json()
    token.access_token = data["access_token"]
    token.refresh_token = data.get("refresh_token", token.refresh_token)
    token.expires_at = data.get("expires_at", 0)
    db.commit()
    return token.access_token


async def fetch_strava_activities(access_token: str, per_page: int = 30, page: int = 1) -> list:
    """Obtiene actividades de Strava."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            STRAVA_ACTIVITIES_URL,
            headers={"Authorization": f"Bearer {access_token}"},
            params={"per_page": per_page, "page": page}
        )
    if response.status_code != 200:
        return []
    return response.json()


def strava_activity_to_session(activity: dict, user_id: int) -> dict:
    """Convierte una actividad de Strava al formato SessionRecord."""
    distance = activity.get("distance", 0)  # metros
    duration = activity.get("moving_time", 0)  # segundos
    speed = (distance / duration) if duration > 0 else 0  # m/s

    # Cadencia promedio (Strava la da en pasos/min para running)
    cadence = activity.get("average_cadence", 0) * 2 if activity.get("average_cadence") else 0

    return {
        "user_id": user_id,
        "date": activity.get("start_date_local", datetime.utcnow().isoformat()),
        "duration": round(duration / 60, 2),  # convertir a minutos
        "steps": int(cadence * (duration / 60)) if cadence > 0 else 0,
        "device": f"Strava - {activity.get('device_name', 'GPS')}",
        "speed": round(speed, 2),
        "cadence": round(cadence, 1),
        "rei": 0.0,
        "gss": round(activity.get("average_speed", 0), 2),
        "asymmetry": 0.0,
        "kli": 0.0,
        "kli_status": "OK",
        "cumulative_load": round(distance / 1000, 2),  # km
        "fatigue_slope": 0.0,
        "fi_times": "[]",
        "fi_values": "[]",
        "strava_activity_id": str(activity.get("id", "")),
        "activity_name": activity.get("name", ""),
        "activity_type": activity.get("type", "Run"),
    }
