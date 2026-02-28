from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    strava_token = Column(String, nullable=True)
    strava_refresh_token = Column(String, nullable=True)
    strava_athlete_id = Column(String, nullable=True)

    sessions = relationship("Session", back_populates="owner")


class Session(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    session_date = Column(DateTime, nullable=True)
    device_position = Column(String, default="espalda")
    source = Column(String, default="manual")  # manual, strava, demo

    # Métricas principales
    duration_minutes = Column(Float, nullable=True)
    distance_km = Column(Float, nullable=True)
    steps = Column(Integer, nullable=True)
    cadence_avg = Column(Float, nullable=True)
    velocity_avg = Column(Float, nullable=True)
    ground_shock_avg = Column(Float, nullable=True)
    asymmetry = Column(Float, nullable=True)
    fatigue_index = Column(Float, nullable=True)
    running_economy = Column(Float, nullable=True)

    # KLI - Knee Load Index
    kli = Column(Float, nullable=True)
    kli_status = Column(String, nullable=True)
    cumulative_load = Column(Float, nullable=True)

    # ML predictions
    injury_risk = Column(Float, nullable=True)
    injury_risk_level = Column(String, nullable=True)
    optimal_pace = Column(Float, nullable=True)
    recovery_days = Column(Integer, nullable=True)

    # Raw data JSON para análisis futuro
    raw_metrics = Column(JSON, nullable=True)

    owner = relationship("User", back_populates="sessions")
