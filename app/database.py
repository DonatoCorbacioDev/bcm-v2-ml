# Copyright (c) 2025 Donato Corbacio. All rights reserved.
# Licensed under the terms of the LICENSE file at the repository root.

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from .config import settings


class Base(DeclarativeBase):
    pass


def connect_args(ssl_ca: str) -> dict:
    """TLS to MySQL with the server's certificate verified against the database's
    CA, when one is given. MySQL generated that CA and its certificate names no
    host, so the chain is checked and the host name is not (sslMode=VERIFY_CA on
    the backend's side)."""
    if not ssl_ca:
        return {}
    return {"ssl": {"ca": ssl_ca, "check_hostname": False}}


engine = create_engine(settings.DB_URL, pool_pre_ping=True, connect_args=connect_args(settings.DB_SSL_CA))
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
