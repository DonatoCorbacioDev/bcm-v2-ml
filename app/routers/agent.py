from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from ..database import get_db
from ..services import agent

router = APIRouter()


@router.get("/agent/insights")
def get_agent_insights(
    db: Annotated[Session, Depends(get_db)],
    months: Annotated[int, Query(ge=1, le=24)] = 3,
):
    return agent.generate_insights(db, months)
