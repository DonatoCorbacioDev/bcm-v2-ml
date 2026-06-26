from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from ..database import get_db
from ..security import verify_internal_api_key
from ..services import ml_risk_scoring, risk_scoring

router = APIRouter(dependencies=[Depends(verify_internal_api_key)])


@router.get("/risk-scores")
def get_risk_scores(
    db: Annotated[Session, Depends(get_db)],
    org_id: Annotated[int | None, Query()] = None,
):
    results = risk_scoring.compute_risk_scores(db, org_id)

    ml_scores = ml_risk_scoring.compute_ml_risk_scores(db, org_id)
    if ml_scores:
        for item in results:
            ml = ml_scores.get(item["contractId"])
            if ml:
                item["mlScore"] = ml["mlScore"]
                item["mlLevel"] = ml["mlLevel"]

    return results
