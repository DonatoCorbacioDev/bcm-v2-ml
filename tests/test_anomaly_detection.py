from unittest.mock import MagicMock

from sqlalchemy.orm import Session

from app.services.anomaly_detection import _severity, compute_anomalies


def _make_db(rows):
    db = MagicMock(spec=Session)
    result = MagicMock()
    result.all.return_value = rows
    db.execute.return_value = result
    return db


def _row(fv_id, contract_id, month, year, amount, name="Client"):
    r = MagicMock()
    r.id = fv_id
    r.contract_id = contract_id
    r.month_value = month
    r.year_value = year
    r.financial_amount = amount
    r.customer_name = name
    return r


class TestSeverity:
    def test_high_below_minus_02(self):
        assert _severity(-0.5) == "HIGH"
        assert _severity(-0.21) == "HIGH"

    def test_medium_between_thresholds(self):
        assert _severity(-0.19) == "MEDIUM"
        assert _severity(-0.06) == "MEDIUM"

    def test_low_above_minus_005(self):
        assert _severity(-0.04) == "LOW"
        assert _severity(0.0) == "LOW"
        assert _severity(0.3) == "LOW"


class TestComputeAnomalies:
    def test_returns_empty_when_below_min_records(self):
        rows = [_row(i, 1, 1, 2025, 1000.0) for i in range(4)]
        assert compute_anomalies(_make_db(rows)) == []

    def test_returns_empty_when_no_records(self):
        assert compute_anomalies(_make_db([])) == []

    def test_detects_extreme_outlier(self):
        rows = [_row(i, 1, (i % 12) + 1, 2025, 1_000.0, "A") for i in range(10)]
        rows.append(_row(99, 2, 6, 2025, 10_000_000.0, "Outlier"))
        results = compute_anomalies(_make_db(rows))
        assert len(results) > 0
        assert "Outlier" in [r["customerName"] for r in results]

    def test_result_has_all_required_fields(self):
        rows = [_row(i, 1, (i % 12) + 1, 2025, 1_000.0 * (i + 1), "A") for i in range(10)]
        rows.append(_row(99, 2, 3, 2025, 9_999_999.0, "Big"))
        results = compute_anomalies(_make_db(rows))
        if results:
            r = results[0]
            assert all(k in r for k in [
                "financialValueId", "contractId", "customerName",
                "month", "year", "financialAmount", "anomalyScore", "severity",
            ])

    def test_results_sorted_by_score_ascending(self):
        rows = [_row(i, 1, (i % 12) + 1, 2025, 1_000.0, "A") for i in range(10)]
        rows.append(_row(97, 2, 3, 2025, 8_000_000.0, "B"))
        rows.append(_row(98, 3, 4, 2025, 9_000_000.0, "C"))
        results = compute_anomalies(_make_db(rows))
        scores = [r["anomalyScore"] for r in results]
        assert scores == sorted(scores)

    def test_anomaly_score_and_amount_are_float(self):
        rows = [_row(i, 1, (i % 12) + 1, 2025, 500.0 * (i + 1), "A") for i in range(10)]
        rows.append(_row(99, 2, 6, 2025, 5_000_000.0, "Anomaly"))
        results = compute_anomalies(_make_db(rows))
        for r in results:
            assert isinstance(r["anomalyScore"], float)
            assert isinstance(r["financialAmount"], float)

    def test_severity_in_valid_values(self):
        rows = [_row(i, 1, (i % 12) + 1, 2025, 500.0 * (i + 1), "A") for i in range(10)]
        rows.append(_row(99, 2, 6, 2025, 5_000_000.0, "Anomaly"))
        results = compute_anomalies(_make_db(rows))
        for r in results:
            assert r["severity"] in ("LOW", "MEDIUM", "HIGH")

    def test_handles_none_financial_amount(self):
        rows = [_row(i, 1, 1, 2025, None if i == 5 else 1_000.0, "A") for i in range(10)]
        rows.append(_row(99, 2, 6, 2025, 5_000_000.0, "Big"))
        results = compute_anomalies(_make_db(rows))
        assert isinstance(results, list)

    def test_db_execute_called_once(self):
        db = _make_db([])
        compute_anomalies(db)
        db.execute.assert_called_once()

    def test_db_execute_called_with_org_id(self):
        db = _make_db([])
        compute_anomalies(db, org_id=5)
        db.execute.assert_called_once()
