from fastapi import APIRouter
from db.database import engine
from sqlalchemy import text

router = APIRouter(prefix="/admin", tags=["admin"])

@router.post("/migrate-strava")
def migrate_strava():
    """Agrega columnas necesarias a las tablas si no existen."""
    migrations = [
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS strava_activity_id VARCHAR",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS activity_name VARCHAR",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS activity_type VARCHAR",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT NOW()",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS source VARCHAR DEFAULT 'manual'",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS duration FLOAT",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS distance_km FLOAT",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS heart_rate FLOAT",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS cadence FLOAT",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS device VARCHAR",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS fatigue_slope FLOAT",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS rei FLOAT",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS gss FLOAT",
    ]
    results = []
    with engine.connect() as conn:
        for sql in migrations:
            try:
                conn.execute(text(sql))
                conn.commit()
                results.append({"sql": sql[:50], "status": "ok"})
            except Exception as e:
                results.append({"sql": sql[:50], "status": "skip", "reason": str(e)[:100]})
    return {"status": "done", "migrations": results}