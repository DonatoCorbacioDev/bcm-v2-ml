from tests.ids import uid
import base64
import time

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException

from app.config import settings
from app.security import verify_internal_api_key, verify_internal_claims


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


# ── verify_internal_claims ────────────────────────────────────────────────

@pytest.fixture(scope="module")
def keypair():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_der = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return private_key, base64.b64encode(public_der).decode()


def _sign(private_key, org_id=None, manager_id=None, exp_delta=30, **overrides):
    now = int(time.time())
    payload = {"orgId": org_id, "managerId": manager_id, "exp": now + exp_delta, "iat": now,
               "iss": "bcm-backend", "aud": "bcm-ml"}
    payload.update(overrides)
    return jwt.encode({k: v for k, v in payload.items() if v is not _DROP}, private_key, algorithm="RS256")


_DROP = object()


@pytest.mark.parametrize("overrides", [
    {"iss": "someone-else"}, {"iss": _DROP},
    {"aud": "bcm-api"}, {"aud": _DROP},
    {"exp": _DROP}, {"iat": _DROP},
], ids=["other issuer", "no issuer", "other audience", "no audience", "no expiry", "no issue date"])
def test_claims_reject_a_token_not_issued_for_this_service(monkeypatch, keypair, overrides):
    private_key, public_key_b64 = keypair
    monkeypatch.setattr(settings, "INTERNAL_CLAIMS_PUBLIC_KEY", public_key_b64)
    token = _sign(private_key, org_id=uid(7), **overrides)

    with pytest.raises(HTTPException) as exc:
        verify_internal_claims(org_id=None, manager_id=None, x_internal_claims=token)
    assert exc.value.status_code == 401


def test_claims_fall_back_to_query_params_when_no_key_configured(monkeypatch):
    monkeypatch.setattr(settings, "INTERNAL_CLAIMS_PUBLIC_KEY", "")

    claims = verify_internal_claims(org_id=uid(7), manager_id=uid(42), x_internal_claims=None)

    assert claims.org_id == uid(7)
    assert claims.manager_id == uid(42)


def test_claims_rejects_missing_header_when_key_configured(monkeypatch, keypair):
    _, public_key_b64 = keypair
    monkeypatch.setattr(settings, "INTERNAL_CLAIMS_PUBLIC_KEY", public_key_b64)

    org_id = uid(7)
    with pytest.raises(HTTPException) as exc:
        verify_internal_claims(org_id=org_id, manager_id=None, x_internal_claims=None)
    assert exc.value.status_code == 401


def test_claims_rejects_malformed_token(monkeypatch, keypair):
    _, public_key_b64 = keypair
    monkeypatch.setattr(settings, "INTERNAL_CLAIMS_PUBLIC_KEY", public_key_b64)

    with pytest.raises(HTTPException) as exc:
        verify_internal_claims(org_id=None, manager_id=None, x_internal_claims="not-a-jwt")
    assert exc.value.status_code == 401


def test_claims_rejects_expired_token(monkeypatch, keypair):
    private_key, public_key_b64 = keypair
    monkeypatch.setattr(settings, "INTERNAL_CLAIMS_PUBLIC_KEY", public_key_b64)
    token = _sign(private_key, org_id=uid(7), exp_delta=-10)

    with pytest.raises(HTTPException) as exc:
        verify_internal_claims(org_id=None, manager_id=None, x_internal_claims=token)
    assert exc.value.status_code == 401


def test_claims_rejects_token_signed_by_a_different_key(monkeypatch, keypair):
    _, public_key_b64 = keypair
    monkeypatch.setattr(settings, "INTERNAL_CLAIMS_PUBLIC_KEY", public_key_b64)
    other_private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    forged_token = _sign(other_private_key, org_id=uid(7))

    with pytest.raises(HTTPException) as exc:
        verify_internal_claims(org_id=None, manager_id=None, x_internal_claims=forged_token)
    assert exc.value.status_code == 401


def test_claims_returns_verified_org_and_manager_id(monkeypatch, keypair):
    private_key, public_key_b64 = keypair
    monkeypatch.setattr(settings, "INTERNAL_CLAIMS_PUBLIC_KEY", public_key_b64)
    token = _sign(private_key, org_id=uid(7), manager_id=uid(42))

    claims = verify_internal_claims(org_id=None, manager_id=None, x_internal_claims=token)

    assert claims.org_id == uid(7)
    assert claims.manager_id == uid(42)


def test_claims_ignore_query_params_when_key_configured(monkeypatch, keypair):
    """Regression test for the actual B2 fix: once a signing key is
    configured, a caller-supplied org_id/manager_id in the query string must
    never override — or even supplement — the verified token's claims."""
    private_key, public_key_b64 = keypair
    monkeypatch.setattr(settings, "INTERNAL_CLAIMS_PUBLIC_KEY", public_key_b64)
    token = _sign(private_key, org_id=uid(7), manager_id=None)

    claims = verify_internal_claims(org_id=uid(999), manager_id=uid(999), x_internal_claims=token)

    assert claims.org_id == uid(7)
    assert claims.manager_id is None


def test_claims_reject_a_valid_token_without_organization(monkeypatch, keypair):
    # Correctly signed by the backend's key, but no orgId: refused, never
    # turned into "every organization".
    private_key, public_b64 = keypair
    monkeypatch.setattr(settings, "INTERNAL_CLAIMS_PUBLIC_KEY", public_b64)
    token = _sign(private_key, org_id=None, manager_id=uid(42))

    org_id = uid(7)
    with pytest.raises(HTTPException) as exc:
        verify_internal_claims(org_id=org_id, manager_id=None, x_internal_claims=token)

    assert exc.value.status_code == 401
    assert exc.value.detail == "Missing organization in internal claims"


def test_claims_reject_missing_org_query_param_when_no_key_configured(monkeypatch):
    monkeypatch.setattr(settings, "INTERNAL_CLAIMS_PUBLIC_KEY", "")

    manager_id = uid(42)
    with pytest.raises(HTTPException) as exc:
        verify_internal_claims(org_id=None, manager_id=manager_id, x_internal_claims=None)

    assert exc.value.status_code == 401


@pytest.mark.parametrize("org_id", [7, "7", "not-a-uuid"])
def test_claims_reject_an_organization_that_is_not_a_uuid(monkeypatch, keypair, org_id):
    # Correctly signed, but a numeric id (tokens from before ids became UUIDs)
    # or any other non-UUID value: refused, never matched against the data.
    private_key, public_b64 = keypair
    monkeypatch.setattr(settings, "INTERNAL_CLAIMS_PUBLIC_KEY", public_b64)
    token = _sign(private_key, org_id=org_id)

    with pytest.raises(HTTPException) as exc:
        verify_internal_claims(org_id=None, manager_id=None, x_internal_claims=token)

    assert exc.value.status_code == 401
    assert exc.value.detail == "Invalid internal claims"


def test_claims_normalize_uuid_text_to_lowercase(monkeypatch, keypair):
    private_key, public_b64 = keypair
    monkeypatch.setattr(settings, "INTERNAL_CLAIMS_PUBLIC_KEY", public_b64)
    token = _sign(private_key, org_id=uid(171).upper(), manager_id=uid(42).upper())

    claims = verify_internal_claims(org_id=None, manager_id=None, x_internal_claims=token)

    assert (claims.org_id, claims.manager_id) == (uid(171), uid(42))
