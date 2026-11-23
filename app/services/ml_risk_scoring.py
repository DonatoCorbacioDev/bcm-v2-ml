# Copyright (c) 2025 Donato Corbacio. All rights reserved.
# Licensed under the terms of the LICENSE file at the repository root.

"""
ML-based risk scoring service.

Loads a pre-trained scikit-learn Pipeline from model/risk_model.joblib at first call,
only if its SHA-256 is the one in model/risk_model.sha256 (in Git; see _load_model).
If the model file is not present (training script not yet run), all methods return
empty results and the router falls back to rule-based scores only.

Feature computation lives in risk_features.build_feature_row — the same
function scripts/generate_training_data.py calls to build training rows, so
this module and the training pipeline can't compute features differently.
"""

import hashlib
import logging
from datetime import date
from pathlib import Path

import numpy as np
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..models import Contract, FinancialValue
from .risk_features import STATUS_CODE, LEVEL_MAP, SCORED_STATUSES, build_feature_row
from .risk_features import build_org_stats as _build_org_stats
from .scope import require_org

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent.parent.parent / "model" / "risk_model.joblib"

_model = None
_model_loaded = False


def expected_digest_path(model_path: Path) -> Path:
    """The model's expected SHA-256, kept in Git next to where the model goes."""
    return model_path.with_suffix(".sha256")


def file_digest(path: Path) -> str:
    """SHA-256 of a file, read in chunks (the model is tens of megabytes)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_digest(model_path: Path) -> str | None:
    """The hash in model/risk_model.sha256 (sha256sum's format), or None if absent or malformed."""
    try:
        text = expected_digest_path(model_path).read_text(encoding="ascii").split()
    except (FileNotFoundError, UnicodeDecodeError):
        return None
    if not text or len(text[0]) != 64 or any(c not in "0123456789abcdef" for c in text[0]):
        return None
    return text[0]


def _load_model():
    """
    Loads the model only if it is the one approved in Git (OWASP ASVS 1.5,
    safe deserialization; OWASP ML Top 10, ML06): joblib.load runs whatever
    code a pickle carries, so a file swapped, corrupted or simply the wrong
    one must never reach it. Any mismatch is an ERROR, never a quiet fallback
    to the rules: ops/maintenance/check-config.sh checks the same each week.
    No model at all (CI, a fresh laptop) stays a warning, as before.
    """
    global _model, _model_loaded
    if _model_loaded:
        return _model
    _model_loaded = True
    if not MODEL_PATH.exists():
        logger.warning("ML risk model not found at %s — run train_risk_model.py to enable ML scoring", MODEL_PATH)
        return _model
    expected = _expected_digest(MODEL_PATH)
    if expected is None:
        logger.error("ML risk model NOT loaded: no valid expected hash in %s", expected_digest_path(MODEL_PATH))
        return _model
    actual = file_digest(MODEL_PATH)
    if actual != expected:
        logger.error("ML risk model NOT loaded: its SHA-256 %s is not the approved %s (%s)",
                     actual, expected, expected_digest_path(MODEL_PATH))
        return _model
    try:
        import joblib
        _model = joblib.load(MODEL_PATH)
        logger.info("ML risk model loaded from %s (SHA-256 %s)", MODEL_PATH, actual)
    except Exception as exc:
        logger.error("Failed to load ML risk model: %s", exc)
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


def compute_ml_risk_scores(db: Session, org_id: str | None, manager_id: str | None = None) -> dict:
    """
    Returns a dict keyed by contract_id:
        {contract_id: {"mlScore": float, "mlLevel": str}}

    Returns an empty dict if the model is not loaded or the query returns no data.
    """
    org_id = require_org(org_id)
    model = _load_model()
    if model is None:
        return {}

    contracts_query = db.query(Contract).filter(Contract.organization_id == org_id)
    if manager_id is not None:
        contracts_query = contracts_query.filter(Contract.manager_id == manager_id)
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
        # The contracts are already this organization's: the organization is
        # filtered again anyway, like every other query on its tables.
        .filter(FinancialValue.organization_id == org_id, FinancialValue.contract_id.in_(contract_ids))
        .group_by(FinancialValue.contract_id)
        .all()
    )
    contract_totals = {r.contract_id: float(r.total or 0.0) for r in fv_rows}
    contract_stds = {r.contract_id: float(r.std or 0.0) for r in fv_rows}
    contract_counts = {r.contract_id: int(r.count or 0) for r in fv_rows}

    # org-level stats for z-score (shared with the rule-based scorer so the
    # two never compute this differently — see risk_scoring._build_org_stats)
    org_stats = _build_org_stats(contracts, contract_totals)

    scored = [c for c in contracts if c.status in SCORED_STATUSES]
    if not scored:
        return {}
    scored_ids = [c.id for c in scored]

    X = _build_feature_matrix(scored, contract_totals, contract_stds, contract_counts, org_stats)

    try:
        probs = model.predict_proba(X)   # shape (n, 3): P(LOW), P(MEDIUM), P(HIGH)
        preds = model.predict(X)
        return {
            scored_ids[i]: {
                "mlScore": round(float(probs[i][2]), 4),   # P(HIGH)
                "mlLevel": LEVEL_MAP.get(int(preds[i]), "LOW"),
            }
            for i in range(len(scored_ids))
        }
    except Exception as exc:
        logger.warning("ML prediction failed: %s", exc)
        return {}
