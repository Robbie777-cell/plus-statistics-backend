from sqlalchemy import Column, Integer, Float, String, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship
from db.database import Base
from datetime import datetime


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    sessions = relationship("SessionRecord", back_populates="owner")
    strava_token = relationship("StravaToken", back_populates="owner", uselist=False)


class SessionRecord(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    date = Column(DateTime, default=datetime.utcnow)
    source = Column(String, default="manual")

    duration_minutes = Column(Float, nullable=True)
    duration = Column(Float, nullable=True)
    distance_km = Column(Float, nullable=True)
    steps = Column(Integer, nullable=True)
    cadence_avg = Column(Float, nullable=True)
    cadence = Column(Float, nullable=True)
    velocity_avg = Column(Float, nullable=True)
    speed = Column(Float, nullable=True)
    ground_shock_avg = Column(Float, nullable=True)
    gss = Column(Float, nullable=True)
    asymmetry = Column(Float, nullable=True)
    fatigue_index = Column(Float, nullable=True)
    fatigue_slope = Column(Float, nullable=True)
    running_economy = Column(Float, nullable=True)
    rei = Column(Float, nullable=True)
    kli = Column(Float, nullable=True)
    kli_status = Column(String, nullable=True)
    cumulative_load = Column(Float, nullable=True)
    injury_risk = Column(Float, nullable=True)
    injury_risk_level = Column(String, nullable=True)
    optimal_pace = Column(Float, nullable=True)
    recovery_days = Column(Integer, nullable=True)
    heart_rate = Column(Float, nullable=True)
    strava_activity_id = Column(String, nullable=True)
    activity_name = Column(String, nullable=True)
    activity_type = Column(String, nullable=True)
    device = Column(String, nullable=True)
    fi_times = Column(String, nullable=True)
    fi_values = Column(String, nullable=True)
    raw_metrics = Column(JSON, nullable=True)

    owner = relationship("User", back_populates="sessions")


class StravaToken(Base):
    __tablename__ = "strava_tokens"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True)
    access_token = Column(String)
    refresh_token = Column(String)
    expires_at = Column(Integer)
    athlete_id = Column(String, nullable=True)
    athlete_name = Column(String, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow)

    owner = relationship("User", back_populates="strava_token")