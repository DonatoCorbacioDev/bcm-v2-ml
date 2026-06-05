from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from ..database import get_db
from ..services import forecasting

router = APIRouter()


@router.get("/forecast")
def get_forecast(
    db: Annotated[Session, Depends(get_db)],
    months: Annotated[int, Query(ge=1, le=24)] = 3,
):
    return forecasting.compute_forecast(db, months)
