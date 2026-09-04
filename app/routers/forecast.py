from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from ..database import get_db
from ..security import InternalClaims, verify_internal_api_key, verify_internal_claims
from ..services import forecasting

router = APIRouter(dependencies=[Depends(verify_internal_api_key)])


@router.get("/forecast")
def get_forecast(
    db: Annotated[Session, Depends(get_db)],
    claims: Annotated[InternalClaims, Depends(verify_internal_claims)],
    months: Annotated[int, Query(ge=1, le=24)] = 3,
):
    return forecasting.compute_forecast(db, months, claims.org_id, claims.manager_id)
