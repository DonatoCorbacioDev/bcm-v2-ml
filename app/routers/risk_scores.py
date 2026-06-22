from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from ..database import get_db
from ..security import verify_internal_api_key
from ..services import risk_scoring

router = APIRouter(dependencies=[Depends(verify_internal_api_key)])


@router.get("/risk-scores")
def get_risk_scores(
    db: Annotated[Session, Depends(get_db)],
    org_id: Annotated[int | None, Query()] = None,
):
    return risk_scoring.compute_risk_scores(db, org_id)
