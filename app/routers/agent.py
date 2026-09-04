from typing import Annotated

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..database import get_db
from ..security import InternalClaims, verify_internal_api_key, verify_internal_claims
from ..services import agent

router = APIRouter(dependencies=[Depends(verify_internal_api_key)])


@router.get("/agent/insights")
def get_agent_insights(
    db: Annotated[Session, Depends(get_db)],
    claims: Annotated[InternalClaims, Depends(verify_internal_claims)],
    months: Annotated[int, Query(ge=1, le=24)] = 3,
):
    return agent.generate_insights(db, months, claims.org_id, claims.manager_id)


class AskAgentRequest(BaseModel):
    question: str


@router.post("/agent/ask")
def post_ask_agent(
    request: AskAgentRequest,
    db: Annotated[Session, Depends(get_db)],
    claims: Annotated[InternalClaims, Depends(verify_internal_claims)],
):
    return agent.ask_agent(db, request.question, claims.org_id, claims.manager_id)
