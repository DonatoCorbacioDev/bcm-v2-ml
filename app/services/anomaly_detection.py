"""
Anomaly detection on financial values using Isolation Forest.

Detects financial records whose amounts are statistically unusual compared to
the rest of the organization. Useful for spotting invoice fraud, data-entry errors,
or one-off outlier contracts.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import Contract, FinancialValue

_MIN_RECORDS = 5
_CONTAMINATION = 0.1


def _severity(score: float) -> str:
    """Convert Isolation Forest decision_function score to severity label.

    decision_function < 0 means anomaly; more negative = more anomalous.
    """
    if score < -0.2:
        return "HIGH"
    if score < -0.05:
        return "MEDIUM"
    return "LOW"


def compute_anomalies(db: Session, org_id: int | None = None) -> list[dict]:
    """Return financial records flagged as anomalous by Isolation Forest.

    Uses financial_amount and cyclical month encoding as features.
    Returns an empty list when fewer than _MIN_RECORDS records exist.
    Results are sorted by anomalyScore ascending (most anomalous first).
    """
    stmt = (
        select(
            FinancialValue.id,
            FinancialValue.contract_id,
            FinancialValue.month_value,
            FinancialValue.year_value,
            FinancialValue.financial_amount,
            Contract.customer_name,
        )
        .join(Contract, FinancialValue.contract_id == Contract.id)
    )
    if org_id is not None:
        stmt = stmt.where(FinancialValue.organization_id == org_id)

    rows = db.execute(stmt).all()
    if len(rows) < _MIN_RECORDS:
        return []

    amounts = np.array([r.financial_amount or 0.0 for r in rows], dtype=float)
    months = np.array([r.month_value for r in rows], dtype=float)
    month_sin = np.sin(2 * np.pi * months / 12)
    month_cos = np.cos(2 * np.pi * months / 12)
    X = np.column_stack([amounts, month_sin, month_cos])

    clf = IsolationForest(contamination=_CONTAMINATION, random_state=42, n_jobs=-1)
    clf.fit(X)
    labels = clf.predict(X)
    scores = clf.decision_function(X)

    results = [
        {
            "financialValueId": row.id,
            "contractId": row.contract_id,
            "customerName": row.customer_name,
            "month": row.month_value,
            "year": row.year_value,
            "financialAmount": float(row.financial_amount or 0.0),
            "anomalyScore": float(scores[i]),
            "severity": _severity(float(scores[i])),
        }
        for i, row in enumerate(rows)
        if labels[i] == -1
    ]
    results.sort(key=lambda x: x["anomalyScore"])
    return results
