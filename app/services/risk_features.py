"""
Single source of truth for the risk-scoring feature/label vocabulary.

Imported by the runtime scoring path (ml_risk_scoring.py) AND by the offline
training scripts (scripts/generate_training_data.py, scripts/train_risk_model.py)
so the feature list, status encoding and class labels can't silently drift
apart between what the model was trained on and what it's asked to score.
Before this module existed, three separate literal copies of these existed and
had already drifted (see docs/research/dataset_sources.md and README_ML.md,
corrected 2026-09-05).
"""

from __future__ import annotations

from datetime import date

import numpy as np

FEATURES = [
    "days_until_expiry",
    "status_code",
    "has_end_date",
    "total_financial_amount",
    "num_financial_records",
    "financial_std",
    "financial_zscore",
]

# Must match the `status` column values written by the backend (Contracts.status).
STATUS_CODE = {"ACTIVE": 0, "EXPIRED": 1, "CANCELLED": 2, "DRAFT": 3}

CLASS_NAMES = ["LOW", "MEDIUM", "HIGH"]
LEVEL_MAP = dict(enumerate(CLASS_NAMES))


def build_org_stats(contracts: list, contract_totals: dict) -> dict:
    """Per-organization (mean, std) of contract totals, used as the baseline
    for financial_zscore. Lives here (not in risk_scoring.py, which imports
    the SQLAlchemy ORM models) so scripts/generate_training_data.py can use
    the exact same baseline computation at training time without pulling in
    a database dependency it doesn't otherwise have — see the module
    docstring. risk_scoring.py and ml_risk_scoring.py both import this
    rather than each computing it their own way (they used to, and had
    already drifted slightly before being unified here)."""
    org_amounts: dict = {}
    for c in contracts:
        org_amounts.setdefault(c.organization_id, []).append(contract_totals.get(c.id, 0.0))
    stats = {}
    for org_id, amounts in org_amounts.items():
        arr = np.array(amounts, dtype=float)
        std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 1.0
        stats[org_id] = (float(np.mean(arr)), std if std > 0 else 1.0)
    return stats


def build_feature_row(
    end_date: date | None,
    status: str,
    total: float,
    count: float,
    std: float,
    org_id,
    org_stats: dict,
    today: date | None = None,
) -> list[float]:
    """Build one feature row (order matches FEATURES) for a single contract.

    This is THE feature computation for risk scoring — called both by
    ml_risk_scoring.py at serving time (from real ORM Contract/FinancialValue
    rows) and by scripts/generate_training_data.py at training time (from
    synthetic domain entities built the same shape). Never duplicate this
    logic elsewhere: a second copy is exactly how days_until_expiry/status_code/
    financial_zscore drifted out of sync with the docs before 2026-09-05.
    """
    today = today or date.today()

    if end_date is not None:
        days = float((end_date - today).days)
        has_end_date = 1.0
    else:
        days = 365.0
        has_end_date = 0.0

    status_code = float(STATUS_CODE.get(str(status), 0))

    if org_id in org_stats:
        mean, org_std = org_stats[org_id]
        z = (total - mean) / org_std
    else:
        z = 0.0

    return [days, status_code, has_end_date, float(total), float(count), float(std), z]
