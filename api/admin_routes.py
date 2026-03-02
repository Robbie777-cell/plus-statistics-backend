from fastapi import APIRouter
from db.database import engine
from sqlalchemy import text

router = APIRouter(prefix="/admin", tags=["admin"])

@router.post("/migrate-strava")
def migrate_strava():
    """Agrega todas las columnas necesarias a las tablas si no existen."""
    migrations = [
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS source VARCHAR DEFAULT 'manual'",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS duration FLOAT",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS duration_minutes FLOAT",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS distance_km FLOAT",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS heart_rate FLOAT",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS cadence FLOAT",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS cadence_avg FLOAT",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS velocity_avg FLOAT",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS speed FLOAT",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS ground_shock_avg FLOAT",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS gss FLOAT",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS asymmetry FLOAT",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS fatigue_index FLOAT",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS fatigue_slope FLOAT",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS running_economy FLOAT",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS rei FLOAT",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS kli FLOAT",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS kli_status VARCHAR",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS cumulative_load FLOAT",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS injury_risk FLOAT",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS injury_risk_level VARCHAR",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS optimal_pace FLOAT",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS recovery_days INTEGER",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS strava_activity_id VARCHAR",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS activity_name VARCHAR",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS activity_type VARCHAR",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS device VARCHAR",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS fi_times VARCHAR",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS fi_values VARCHAR",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS raw_metrics JSONB",
    ]
    results = []
    with engine.connect() as conn:
        for sql in migrations:
            try:
                conn.execute(text(sql))
                conn.commit()
                results.append({"sql": sql[:60], "status": "ok"})
            except Exception as e:
                results.append({"sql": sql[:60], "status": "skip", "reason": str(e)[:100]})
    return {"status": "done", "migrations": results}