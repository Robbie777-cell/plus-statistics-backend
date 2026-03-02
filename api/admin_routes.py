@router.post("/clear-sessions")
def clear_sessions():
    """Borra todas las sesiones para reimportar desde cero."""
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM sessions"))
        conn.commit()
    return {"status": "ok", "message": "Todas las sesiones borradas"}