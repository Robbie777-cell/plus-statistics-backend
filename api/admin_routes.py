from fastapi import APIRouter
from db.database import engine
from sqlalchemy import text

router = APIRouter(prefix="/admin", tags=["admin"])

@router.post("/migrate-strava")
def migrate_strava():
    """Agrega columnas de Strava a la tabla sessions si no existen."""
    migrations = [
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS strava_activity_id VARCHAR",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS activity_name VARCHAR",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS activity_type VARCHAR",
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT NOW()",
        "ALTER TABLE strava_tokens ADD COLUMN IF NOT EXISTS id SERIAL PRIMARY KEY" if False else "SELECT 1",
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
