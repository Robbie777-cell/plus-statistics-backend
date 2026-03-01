import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db.models import Base

# Limpiar el DATABASE_URL de espacios y saltos de línea invisibles
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./plusstats.db").strip()

# Railway a veces da postgres:// en lugar de postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Si el URL termina en /railway (nombre incorrecto), corregir a /postgres
if DATABASE_URL.endswith("/railway"):
    DATABASE_URL = DATABASE_URL[:-8] + "/postgres"

# Crear engine con manejo de errores
try:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=300)
    Base.metadata.create_all(bind=engine)
    print("✅ Base de datos conectada correctamente")
except Exception as e:
    print(f"⚠️ Error de base de datos: {e}")
    # Fallback a SQLite para que el servidor arranque igual
    DATABASE_URL = "sqlite:///./plusstats.db"
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    print("⚠️ Usando SQLite como fallback")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    Base.metadata.create_all(bind=engine)
