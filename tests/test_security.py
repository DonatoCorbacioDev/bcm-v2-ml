import pytest
from fastapi import HTTPException

from app.config import settings
from app.security import verify_internal_api_key


@pytest.fixture(autouse=True)
def _reset_key():
    original = settings.INTERNAL_API_KEY
    yield
    settings.INTERNAL_API_KEY = original


def test_check_disabled_when_key_unset():
    settings.INTERNAL_API_KEY = ""
    verify_internal_api_key(x_internal_api_key=None)  # should not raise


def test_rejects_missing_header_when_key_set():
    settings.INTERNAL_API_KEY = "secret"
    with pytest.raises(HTTPException) as exc:
        verify_internal_api_key(x_internal_api_key=None)
    assert exc.value.status_code == 401


def test_rejects_wrong_key():
    settings.INTERNAL_API_KEY = "secret"
    with pytest.raises(HTTPException) as exc:
        verify_internal_api_key(x_internal_api_key="wrong")
    assert exc.value.status_code == 401


def test_accepts_correct_key():
    settings.INTERNAL_API_KEY = "secret"
    verify_internal_api_key(x_internal_api_key="secret")  # should not raise
