from typing import Annotated

from fastapi import Header, HTTPException, status

from .config import settings


def verify_internal_api_key(
    x_internal_api_key: Annotated[str | None, Header()] = None,
) -> None:
    if settings.INTERNAL_API_KEY and x_internal_api_key != settings.INTERNAL_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing internal API key",
        )
