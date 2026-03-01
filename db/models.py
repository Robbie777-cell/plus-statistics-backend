from sqlalchemy import Column, Integer, String, Float, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String, default="")
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    sessions = relationship("SessionRecord", back_populates="user")


class SessionRecord(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    date = Column(String, default="")
    duration = Column(Float, default=0.0)
    steps = Column(Integer, default=0)
    device = Column(String, default="")

    # Métricas biomecánicas
    rei = Column(Float, default=0.0)
    gss = Column(Float, default=0.0)
    cadence = Column(Float, default=0.0)
    asymmetry = Column(Float, default=0.0)
    speed = Column(Float, default=0.0)

    # KLI
    kli = Column(Float, default=0.0)
    kli_status = Column(String, default="OK")
    cumulative_load = Column(Float, default=0.0)

    # Fatiga
    fatigue_slope = Column(Float, default=0.0)
    fi_times = Column(Text, default="[]")
    fi_values = Column(Text, default="[]")

    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="sessions")
