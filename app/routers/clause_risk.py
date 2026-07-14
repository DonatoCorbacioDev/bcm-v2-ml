from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ..security import verify_internal_api_key
from ..services import clause_risk

router = APIRouter(dependencies=[Depends(verify_internal_api_key)])


class ClauseRiskRequest(BaseModel):
    text: str


@router.post("/clause-risk-analysis")
def analyze_clause_risk(request: ClauseRiskRequest):
    return clause_risk.analyze_clauses(request.text)
