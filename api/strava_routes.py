from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import RedirectResponse
import httpx
import os
from services.auth import get_current_user
from database import get_db
from sqlalchemy.orm import Session

router = APIRouter(prefix="/strava", tags=["strava"])

CLIENT_ID = os.getenv("STRAVA_CLIENT_ID")
CLIENT_SECRET = os.getenv("STRAVA_CLIENT_SECRET")
REDIRECT_URI = os.getenv("STRAVA_REDIRECT_URI")

@router.get("/connect")
def connect_strava():
    url = (
        f"https://www.strava.com/oauth/authorize"
        f"?client_id={CLIENT_ID}"
        f"&redirect_uri={REDIRECT_URI}"
        f"&response_type=code"
        f"&scope=activity:read_all"
    )
    return RedirectResponse(url)

@router.get("/callback")
async def strava_callback(code: str, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
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
    return {"status": "conectado", "athlete": data.get("athlete", {}).get("firstname")} 
