from datetime import date, timedelta
from unittest.mock import MagicMock
from app.services.risk_scoring import (
    _expiry_score, _z_score, _level, _build_org_stats, compute_risk_scores,
)


class ContractMock:
    def __init__(self, id, name, end_date, org_id):
        self.id = id
        self.customer_name = name
        self.end_date = end_date
        self.organization_id = org_id


class FVRowMock:
    def __init__(self, contract_id, total):
        self.contract_id = contract_id
        self.total = total


# ── _expiry_score ──────────────────────────────────────────────────────────────

def test_expiry_no_end_date():
    score, anomalies = _expiry_score(None)
    assert score == 0.3
    assert anomalies == ["NO_END_DATE"]

def test_expiry_expired():
    score, anomalies = _expiry_score(date.today() - timedelta(days=1))
    assert score == 1.0
    assert anomalies == ["EXPIRED"]

def test_expiry_expiring_soon():
    score, anomalies = _expiry_score(date.today() + timedelta(days=15))
    assert score == 0.8
    assert anomalies == ["EXPIRING_SOON"]

def test_expiry_within_90_days():
    score, anomalies = _expiry_score(date.today() + timedelta(days=60))
    assert score == 0.5
    assert anomalies == []

def test_expiry_within_180_days():
    score, anomalies = _expiry_score(date.today() + timedelta(days=120))
    assert score == 0.3
    assert anomalies == []

def test_expiry_far_future():
    score, anomalies = _expiry_score(date.today() + timedelta(days=365))
    assert score == 0.1
    assert anomalies == []


# ── _z_score ───────────────────────────────────────────────────────────────────

def test_z_score_unusual_high():
    stats = {1: (5000.0, 1000.0)}
    z, anomalies = _z_score(12000.0, 1, stats)
    assert anomalies == ["UNUSUAL_VALUE"]

def test_z_score_unusual_low():
    stats = {1: (5000.0, 1000.0)}
    z, anomalies = _z_score(0.0, 1, stats)
    assert anomalies == ["UNUSUAL_VALUE"]

def test_z_score_normal():
    stats = {1: (5000.0, 1000.0)}
    _, anomalies = _z_score(5000.0, 1, stats)
    assert anomalies == []

def test_z_score_no_org_id():
    z, anomalies = _z_score(5000.0, None, {})
    assert z == 0.0
    assert anomalies == []

def test_z_score_org_not_in_stats():
    z, anomalies = _z_score(5000.0, 99, {})
    assert z == 0.0


# ── _level ─────────────────────────────────────────────────────────────────────

def test_level_high():
    assert _level(0.65) == "HIGH"
    assert _level(1.0) == "HIGH"

def test_level_medium():
    assert _level(0.35) == "MEDIUM"
    assert _level(0.64) == "MEDIUM"

def test_level_low():
    assert _level(0.0) == "LOW"
    assert _level(0.34) == "LOW"


# ── _build_org_stats ───────────────────────────────────────────────────────────

def test_build_org_stats_single_contract():
    contracts = [ContractMock(1, "A", None, 1)]
    stats = _build_org_stats(contracts, {1: 5000.0})
    assert stats[1][0] == 5000.0
    assert stats[1][1] == 1.0  # default std when single item

def test_build_org_stats_multiple_contracts():
    contracts = [ContractMock(1, "A", None, 1), ContractMock(2, "B", None, 1)]
    stats = _build_org_stats(contracts, {1: 4000.0, 2: 6000.0})
    assert stats[1][0] == 5000.0
    assert stats[1][1] > 0

def test_build_org_stats_zero_std_fallback():
    contracts = [ContractMock(1, "A", None, 1), ContractMock(2, "B", None, 1)]
    stats = _build_org_stats(contracts, {1: 5000.0, 2: 5000.0})
    assert stats[1][1] == 1.0  # std=0 replaced with 1.0


# ── compute_risk_scores ────────────────────────────────────────────────────────

def make_risk_session(contracts, fv_rows):
    session = MagicMock()
    contracts_q = MagicMock()
    contracts_q.all.return_value = contracts
    fv_q = MagicMock()
    fv_q.group_by.return_value.all.return_value = fv_rows
    session.query.side_effect = [contracts_q, fv_q]
    return session


def test_compute_risk_scores_empty():
    session = MagicMock()
    session.query.return_value.all.return_value = []
    assert compute_risk_scores(session) == []


def test_compute_risk_scores_sorted_descending():
    today = date.today()
    contracts = [
        ContractMock(1, "Expired Corp", today - timedelta(days=10), 1),
        ContractMock(2, "Soon Ltd", today + timedelta(days=15), 1),
        ContractMock(3, "Far Inc", today + timedelta(days=365), 1),
        ContractMock(4, "No Date SA", None, 1),
    ]
    fv_rows = [FVRowMock(c.id, 10000.0) for c in contracts]
    result = compute_risk_scores(make_risk_session(contracts, fv_rows))
    assert len(result) == 4
    for i in range(len(result) - 1):
        assert result[i]["riskScore"] >= result[i + 1]["riskScore"]


def test_compute_risk_scores_structure():
    today = date.today()
    contracts = [ContractMock(1, "Acme", today - timedelta(days=5), 1)]
    fv_rows = [FVRowMock(1, 50000.0)]
    result = compute_risk_scores(make_risk_session(contracts, fv_rows))
    assert len(result) == 1
    item = result[0]
    assert item["contractId"] == 1
    assert item["customerName"] == "Acme"
    assert 0.0 <= item["riskScore"] <= 1.0
    assert item["level"] in ("HIGH", "MEDIUM", "LOW")
    assert "EXPIRED" in item["anomalies"]


def test_compute_risk_scores_unusual_value():
    today = date.today()
    # With 5 contracts at 1000 and 1 at 100000: z ≈ 2.04 > 2 → UNUSUAL_VALUE
    contracts = [ContractMock(i, f"Normal{i}", today + timedelta(days=365), 1) for i in range(1, 6)]
    contracts.append(ContractMock(6, "Outlier", today + timedelta(days=365), 1))
    fv_rows = [FVRowMock(i, 1000.0) for i in range(1, 6)]
    fv_rows.append(FVRowMock(6, 100000.0))
    result = compute_risk_scores(make_risk_session(contracts, fv_rows))
    outlier = next(r for r in result if r["contractId"] == 6)
    assert "UNUSUAL_VALUE" in outlier["anomalies"]


def make_org_risk_session(contracts, fv_rows):
    session = MagicMock()
    contracts_q = MagicMock()
    contracts_q.filter.return_value.all.return_value = contracts
    fv_q = MagicMock()
    fv_q.filter.return_value.group_by.return_value.all.return_value = fv_rows
    session.query.side_effect = [contracts_q, fv_q]
    return session


def test_compute_risk_scores_with_org_id_filters_query():
    today = date.today()
    contracts = [ContractMock(1, "Acme", today - timedelta(days=5), 7)]
    fv_rows = [FVRowMock(1, 50000.0)]
    session = make_org_risk_session(contracts, fv_rows)

    result = compute_risk_scores(session, org_id=7)

    assert len(result) == 1
    assert result[0]["contractId"] == 1
