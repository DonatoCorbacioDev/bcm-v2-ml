from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..database import get_db
from ..security import InternalClaims, verify_internal_api_key, verify_internal_claims
from ..services import anomaly_detection

router = APIRouter(dependencies=[Depends(verify_internal_api_key)])


@router.get("/anomalies")
def get_anomalies(
    db: Annotated[Session, Depends(get_db)],
    claims: Annotated[InternalClaims, Depends(verify_internal_claims)],
):
    return anomaly_detection.compute_anomalies(db, claims.org_id, claims.manager_id)
