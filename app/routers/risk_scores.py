from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..database import get_db
from ..security import InternalClaims, verify_internal_api_key, verify_internal_claims
from ..services import ml_risk_scoring, risk_scoring

router = APIRouter(dependencies=[Depends(verify_internal_api_key)])


@router.get("/risk-scores")
def get_risk_scores(
    db: Annotated[Session, Depends(get_db)],
    claims: Annotated[InternalClaims, Depends(verify_internal_claims)],
):
    results = risk_scoring.compute_risk_scores(db, claims.org_id, claims.manager_id)

    ml_scores = ml_risk_scoring.compute_ml_risk_scores(db, claims.org_id, claims.manager_id)
    if ml_scores:
        for item in results:
            ml = ml_scores.get(item["contractId"])
            if ml:
                item["mlScore"] = ml["mlScore"]
                item["mlLevel"] = ml["mlLevel"]

    return results
