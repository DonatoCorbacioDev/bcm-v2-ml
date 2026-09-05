"""
ML-based risk scoring service.

Loads a pre-trained scikit-learn Pipeline from model/risk_model.joblib at first call.
If the model file is not present (training script not yet run), all methods return
empty results and the router falls back to rule-based scores only.

Feature computation lives in risk_features.build_feature_row — the same
function scripts/generate_training_data.py calls to build training rows, so
this module and the training pipeline can't compute features differently.
"""

import logging
from datetime import date
from pathlib import Path

import numpy as np
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..models import Contract, FinancialValue
from .risk_features import STATUS_CODE, LEVEL_MAP, build_feature_row
from .risk_features import build_org_stats as _build_org_stats

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent.parent.parent / "model" / "risk_model.joblib"

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
    today: date | None = None,
) -> np.ndarray:
    today = today or date.today()
    rows = [
        build_feature_row(
            c.end_date,
            c.status,
            contract_totals.get(c.id, 0.0),
            contract_counts.get(c.id, 0),
            contract_stds.get(c.id, 0.0),
            c.organization_id,
            org_stats,
            today,
        )
        for c in contracts
    ]
    return np.array(rows, dtype=float) if rows else np.empty((0, 7))


def compute_ml_risk_scores(db: Session, org_id: int | None = None, manager_id: int | None = None) -> dict:
    """
    Returns a dict keyed by contract_id:
        {contract_id: {"mlScore": float, "mlLevel": str}}

    Returns an empty dict if the model is not loaded or the query returns no data.
    """
    model = _load_model()
    if model is None:
        return {}

    contracts_query = db.query(Contract)
    if manager_id is not None:
        contracts_query = contracts_query.filter(Contract.manager_id == manager_id)
    elif org_id is not None:
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

    # org-level stats for z-score (shared with the rule-based scorer so the
    # two never compute this differently — see risk_scoring._build_org_stats)
    org_stats = _build_org_stats(contracts, contract_totals)

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
