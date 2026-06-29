from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from ..database import get_db
from ..security import verify_internal_api_key
from ..services import anomaly_detection

router = APIRouter(dependencies=[Depends(verify_internal_api_key)])


@router.get("/anomalies")
def get_anomalies(
    db: Annotated[Session, Depends(get_db)],
    org_id: Annotated[int | None, Query()] = None,
):
    return anomaly_detection.compute_anomalies(db, org_id)
