# Copyright (c) 2025 Donato Corbacio. All rights reserved.
# Licensed under the terms of the LICENSE file at the repository root.

import base64
from typing import Annotated
from uuid import UUID

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
    """The caller's verified org_id/manager_id scope for one request.

    Both are UUIDs in canonical lowercase text (ADR-0009 in the backend), the
    form the CHAR(36) columns hold.
    """

    org_id: str | None = None
    manager_id: str | None = None


def verify_internal_claims(
    org_id: Annotated[UUID | None, Query()] = None,
    manager_id: Annotated[UUID | None, Query()] = None,
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

    Either way a scope without an organization is refused (401): no
    organization never means "every organization".
    """
    if not settings.INTERNAL_CLAIMS_PUBLIC_KEY:
        return _require_org_scope(InternalClaims(org_id=_text(org_id), manager_id=_text(manager_id)))

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

    try:
        claims = InternalClaims(org_id=_uuid_claim(payload.get("orgId")),
                                manager_id=_uuid_claim(payload.get("managerId")))
    except ValueError as exc:
        # Signed, but not a UUID (e.g. a numeric id from before ADR-0009): refused.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid internal claims",
        ) from exc
    return _require_org_scope(claims)


def _text(value: UUID | None) -> str | None:
    return str(value) if value is not None else None


def _uuid_claim(value: object) -> str | None:
    """A claim value as canonical UUID text; anything else that is present is invalid."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("claim is not a string")
    return str(UUID(value))


def _require_org_scope(claims: InternalClaims) -> InternalClaims:
    if claims.org_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing organization in internal claims",
        )
    return claims
