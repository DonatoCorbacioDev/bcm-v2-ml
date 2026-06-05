from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..database import get_db
from ..services import risk_scoring

router = APIRouter()


@router.get("/risk-scores")
def get_risk_scores(db: Annotated[Session, Depends(get_db)]):
    return risk_scoring.compute_risk_scores(db)
