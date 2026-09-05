from datetime import date

from sqlalchemy import func
from sqlalchemy.orm import Session

from ..models import Contract, FinancialValue
from .risk_features import build_org_stats as _build_org_stats


def _expiry_score(end_date) -> tuple[float, list[str]]:
    if end_date is None:
        return 0.3, ["NO_END_DATE"]
    days = (end_date - date.today()).days
    if days < 0:
        return 1.0, ["EXPIRED"]
    if days < 30:
        return 0.8, ["EXPIRING_SOON"]
    if days < 90:
        return 0.5, []
    if days < 180:
        return 0.3, []
    return 0.1, []


def _z_score(total: float, org_id, org_stats: dict) -> tuple[float, list[str]]:
    if org_id and org_id in org_stats:
        mean, std = org_stats[org_id]
        z = (total - mean) / std
    else:
        z = 0.0
    return z, (["UNUSUAL_VALUE"] if abs(z) > 2 else [])


def _level(score: float) -> str:
    if score >= 0.65:
        return "HIGH"
    if score >= 0.35:
        return "MEDIUM"
    return "LOW"


def compute_risk_scores(db: Session, org_id: int | None = None, manager_id: int | None = None) -> list:
    contracts_query = db.query(Contract)
    if manager_id is not None:
        contracts_query = contracts_query.filter(Contract.manager_id == manager_id)
    elif org_id is not None:
        contracts_query = contracts_query.filter(Contract.organization_id == org_id)
    contracts = contracts_query.all()
    if not contracts:
        return []

    fv_query = db.query(FinancialValue.contract_id, func.sum(FinancialValue.financial_amount).label("total"))
    if manager_id is not None:
        # FinancialValue has no manager_id of its own — restrict totals to the
        # already-scoped contracts list instead.
        contract_ids = [c.id for c in contracts]
        fv_query = fv_query.filter(FinancialValue.contract_id.in_(contract_ids))
    elif org_id is not None:
        fv_query = fv_query.filter(FinancialValue.organization_id == org_id)
    fv_rows = fv_query.group_by(FinancialValue.contract_id).all()
    contract_totals = {row.contract_id: float(row.total) for row in fv_rows}
    org_stats = _build_org_stats(contracts, contract_totals)

    results = []
    for c in contracts:
        expiry, expiry_anomalies = _expiry_score(c.end_date)
        total = contract_totals.get(c.id, 0.0)
        z, value_anomalies = _z_score(total, c.organization_id, org_stats)
        risk_score = round(0.6 * expiry + 0.4 * min(abs(z) / 3.0, 1.0), 4)
        results.append({
            "contractId": c.id,
            "customerName": c.customer_name,
            "riskScore": risk_score,
            "level": _level(risk_score),
            "anomalies": expiry_anomalies + value_anomalies,
        })

    results.sort(key=lambda x: x["riskScore"], reverse=True)
    return results
