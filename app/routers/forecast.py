from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from ..database import get_db
from ..security import verify_internal_api_key
from ..services import forecasting

router = APIRouter(dependencies=[Depends(verify_internal_api_key)])


@router.get("/forecast")
def get_forecast(
    db: Annotated[Session, Depends(get_db)],
    months: Annotated[int, Query(ge=1, le=24)] = 3,
    org_id: Annotated[int | None, Query()] = None,
):
    return forecasting.compute_forecast(db, months, org_id)
