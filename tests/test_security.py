import pytest
from fastapi import HTTPException

from app.config import settings
from app.security import verify_internal_api_key


def test_check_disabled_when_key_unset(monkeypatch):
    monkeypatch.setattr(settings, "INTERNAL_API_KEY", "")
    verify_internal_api_key(x_internal_api_key=None)  # should not raise


def test_rejects_missing_header_when_key_set(monkeypatch):
    monkeypatch.setattr(settings, "INTERNAL_API_KEY", "secret")
    with pytest.raises(HTTPException) as exc:
        verify_internal_api_key(x_internal_api_key=None)
    assert exc.value.status_code == 401


def test_rejects_wrong_key(monkeypatch):
    monkeypatch.setattr(settings, "INTERNAL_API_KEY", "secret")
    with pytest.raises(HTTPException) as exc:
        verify_internal_api_key(x_internal_api_key="wrong")
    assert exc.value.status_code == 401


def test_accepts_correct_key(monkeypatch):
    monkeypatch.setattr(settings, "INTERNAL_API_KEY", "secret")
    verify_internal_api_key(x_internal_api_key="secret")  # should not raise
