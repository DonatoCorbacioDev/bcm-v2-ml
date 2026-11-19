# Copyright (c) 2025 Donato Corbacio. All rights reserved.
# Licensed under the terms of the LICENSE file at the repository root.

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from .config import settings


class Base(DeclarativeBase):
    pass


def connect_args(ssl_ca: str) -> dict:
    """TLS to MySQL with the server's certificate verified, when its CA is given:
    the chain against that CA and the name against the host in DB_URL (mysql),
    like sslMode=VERIFY_IDENTITY on the backend's side (ops/mysql/issue-tls-cert.sh
    issues a certificate naming it)."""
    if not ssl_ca:
        return {}
    return {"ssl": {"ca": ssl_ca, "check_hostname": True}}


engine = create_engine(settings.DB_URL, pool_pre_ping=True, connect_args=connect_args(settings.DB_SSL_CA))
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
