"""
ML-based risk scoring service.

Loads a pre-trained scikit-learn Pipeline from model/risk_model.joblib at first call.
If the model file is not present (training script not yet run), all methods return
empty results and the router falls back to rule-based scores only.

Features used (must match scripts/generate_training_data.py):
  days_until_expiry, status_code, has_end_date, total_financial_amount,
  num_financial_records, financial_std, financial_zscore
"""

import logging
from datetime import date
from pathlib import Path

import numpy as np
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..models import Contract, FinancialValue

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent.parent.parent / "model" / "risk_model.joblib"

STATUS_CODE = {"ACTIVE": 0, "EXPIRED": 1, "CANCELLED": 2, "DRAFT": 3}
LEVEL_MAP = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}

_model = None
_model_loaded = False


def _load_model():
    global _model, _model_loaded
    if _model_loaded:
        return _model
    _model_loaded = True
    try:
        import joblib
        _model = joblib.load(MODEL_PATH)
        logger.info("ML risk model loaded from %s", MODEL_PATH)
    except FileNotFoundError:
        logger.warning("ML risk model not found at %s — run train_risk_model.py to enable ML scoring", MODEL_PATH)
    except Exception as exc:
        logger.warning("Failed to load ML risk model: %s", exc)
    return _model


def _build_feature_matrix(
    contracts: list,
    contract_totals: dict,
    contract_stds: dict,
    contract_counts: dict,
    org_stats: dict,
) -> np.ndarray:
    today = date.today()
    rows = []
    for c in contracts:
        if c.end_date is not None:
            days = float((c.end_date - today).days)
            has_end_date = 1.0
        else:
            days = 365.0
            has_end_date = 0.0

        status_code = float(STATUS_CODE.get(str(c.status), 0))
        total = float(contract_totals.get(c.id, 0.0))
        std = float(contract_stds.get(c.id, 0.0))
        count = float(contract_counts.get(c.id, 0))

        if c.organization_id in org_stats:
            mean, ostd = org_stats[c.organization_id]
            z = (total - mean) / ostd
        else:
            z = 0.0

        rows.append([days, status_code, has_end_date, total, count, std, z])

    return np.array(rows, dtype=float) if rows else np.empty((0, 7))


def compute_ml_risk_scores(db: Session, org_id: int | None = None) -> dict:
    """
    Returns a dict keyed by contract_id:
        {contract_id: {"mlScore": float, "mlLevel": str}}

    Returns an empty dict if the model is not loaded or the query returns no data.
    """
    model = _load_model()
    if model is None:
        return {}

    contracts_query = db.query(Contract)
    if org_id is not None:
        contracts_query = contracts_query.filter(Contract.organization_id == org_id)
    contracts = contracts_query.all()
    if not contracts:
        return {}

    contract_ids = [c.id for c in contracts]

    fv_rows = (
        db.query(
            FinancialValue.contract_id,
            func.sum(FinancialValue.financial_amount).label("total"),
            func.stddev(FinancialValue.financial_amount).label("std"),
            func.count(FinancialValue.id).label("count"),
        )
        .filter(FinancialValue.contract_id.in_(contract_ids))
        .group_by(FinancialValue.contract_id)
        .all()
    )
    contract_totals = {r.contract_id: float(r.total or 0.0) for r in fv_rows}
    contract_stds = {r.contract_id: float(r.std or 0.0) for r in fv_rows}
    contract_counts = {r.contract_id: int(r.count or 0) for r in fv_rows}

    # org-level stats for z-score
    org_amounts: dict = {}
    for c in contracts:
        org_amounts.setdefault(c.organization_id, []).append(contract_totals.get(c.id, 0.0))
    org_stats = {}
    for oid, amounts in org_amounts.items():
        arr = np.array(amounts, dtype=float)
        std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 1.0
        org_stats[oid] = (float(np.mean(arr)), max(std, 1e-9))

    X = _build_feature_matrix(contracts, contract_totals, contract_stds, contract_counts, org_stats)

    try:
        probs = model.predict_proba(X)   # shape (n, 3): P(LOW), P(MEDIUM), P(HIGH)
        preds = model.predict(X)
        return {
            contract_ids[i]: {
                "mlScore": round(float(probs[i][2]), 4),   # P(HIGH)
                "mlLevel": LEVEL_MAP.get(int(preds[i]), "LOW"),
            }
            for i in range(len(contract_ids))
        }
    except Exception as exc:
        logger.warning("ML prediction failed: %s", exc)
        return {}
