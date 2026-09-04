import base64
from typing import Annotated

import jwt
from cryptography.hazmat.primitives import serialization
from fastapi import Header, HTTPException, Query, status
from pydantic import BaseModel

from .config import settings


def verify_internal_api_key(
    x_internal_api_key: Annotated[str | None, Header()] = None,
) -> None:
    if settings.INTERNAL_API_KEY and x_internal_api_key != settings.INTERNAL_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing internal API key",
        )


class InternalClaims(BaseModel):
    """The caller's verified org_id/manager_id scope for one request."""

    org_id: int | None = None
    manager_id: int | None = None


def verify_internal_claims(
    org_id: Annotated[int | None, Query()] = None,
    manager_id: Annotated[int | None, Query()] = None,
    x_internal_claims: Annotated[str | None, Header()] = None,
) -> InternalClaims:
    """Resolves the request's org_id/manager_id scope.

    When INTERNAL_CLAIMS_PUBLIC_KEY is configured, the query params are
    ignored entirely and only a verified X-Internal-Claims JWT (signed by the
    backend's private key, which this service never holds) is trusted —
    otherwise org_id/manager_id would still be a caller-supplied value an
    attacker in possession of X-Internal-Api-Key could set to anything.
    When no public key is configured (local dev), falls back to trusting the
    query params directly, matching verify_internal_api_key's empty-key
    dev posture.
    """
    if not settings.INTERNAL_CLAIMS_PUBLIC_KEY:
        return InternalClaims(org_id=org_id, manager_id=manager_id)

    if not x_internal_claims:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing internal claims",
        )
    try:
        public_key = serialization.load_der_public_key(
            base64.b64decode(settings.INTERNAL_CLAIMS_PUBLIC_KEY)
        )
        payload = jwt.decode(x_internal_claims, public_key, algorithms=["RS256"])
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid internal claims",
        ) from exc

    return InternalClaims(org_id=payload.get("orgId"), manager_id=payload.get("managerId"))
