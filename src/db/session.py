"""Database session management."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.config import DATABASE_URL
from src.db.models import Base


def _sqlite_url(url: str) -> str:
    if url.startswith("sqlite:///") and not url.startswith("sqlite:////"):
        return url
    return url


engine = create_engine(
    _sqlite_url(DATABASE_URL),
    echo=False,
    connect_args={"check_same_thread": False, "timeout": 60} if "sqlite" in DATABASE_URL else {},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    Base.metadata.create_all(engine)
    if "sqlite" in DATABASE_URL:
        with engine.connect() as conn:
            conn.exec_driver_sql("PRAGMA journal_mode=WAL")
            conn.commit()


@contextmanager
def get_session() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
