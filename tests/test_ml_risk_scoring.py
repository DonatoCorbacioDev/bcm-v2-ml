"""Unit tests for app/services/ml_risk_scoring.py"""

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.services import ml_risk_scoring
from app.services.ml_risk_scoring import (
    LEVEL_MAP,
    STATUS_CODE,
    _build_feature_matrix,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_contract(cid, status="ACTIVE", end_date=None, org_id=1):
    c = MagicMock()
    c.id = cid
    c.status = status
    c.end_date = end_date
    c.organization_id = org_id
    return c


def _reset_model_cache():
    ml_risk_scoring._model = None
    ml_risk_scoring._model_loaded = False


# ---------------------------------------------------------------------------
# _build_feature_matrix
# ---------------------------------------------------------------------------

class TestBuildFeatureMatrix:
    def test_empty_contracts_returns_empty_array(self):
        X = _build_feature_matrix([], {}, {}, {}, {})
        assert X.shape == (0, 7)

    def test_active_contract_with_end_date(self):
        future = date.today() + timedelta(days=60)
        c = _make_contract(1, end_date=future)
        X = _build_feature_matrix([c], {1: 10000.0}, {1: 500.0}, {1: 12}, {1: (10000.0, 1000.0)})
        row = X[0]
        assert row[0] == pytest.approx(60.0, abs=1)   # days_until_expiry
        assert row[1] == 0.0                            # ACTIVE
        assert row[2] == 1.0                            # has_end_date
        assert row[3] == pytest.approx(10000.0)         # total
        assert row[4] == pytest.approx(12.0)            # count
        assert row[5] == pytest.approx(500.0)           # std

    def test_contract_without_end_date_uses_365(self):
        c = _make_contract(1, end_date=None)
        X = _build_feature_matrix([c], {}, {}, {}, {})
        assert X[0, 0] == pytest.approx(365.0)
        assert X[0, 2] == 0.0  # has_end_date = 0

    def test_expired_contract_has_negative_days(self):
        past = date.today() - timedelta(days=10)
        c = _make_contract(1, status="EXPIRED", end_date=past)
        X = _build_feature_matrix([c], {}, {}, {}, {})
        assert X[0, 0] < 0
        assert X[0, 1] == float(STATUS_CODE["EXPIRED"])

    def test_z_score_computed_from_org_stats(self):
        c = _make_contract(1, org_id=5)
        org_stats = {5: (5000.0, 1000.0)}
        X = _build_feature_matrix([c], {1: 7000.0}, {}, {}, org_stats)
        assert X[0, 6] == pytest.approx(2.0)  # (7000 - 5000) / 1000

    def test_z_score_zero_when_no_org_stats(self):
        c = _make_contract(1, org_id=99)
        X = _build_feature_matrix([c], {1: 5000.0}, {}, {}, {})
        assert X[0, 6] == pytest.approx(0.0)

    def test_unknown_status_maps_to_zero(self):
        c = _make_contract(1, status="UNKNOWN")
        X = _build_feature_matrix([c], {}, {}, {}, {})
        assert X[0, 1] == 0.0

    def test_multiple_contracts_correct_row_count(self):
        contracts = [_make_contract(i) for i in range(5)]
        X = _build_feature_matrix(contracts, {}, {}, {}, {})
        assert X.shape == (5, 7)


# ---------------------------------------------------------------------------
# _load_model
# ---------------------------------------------------------------------------

class TestLoadModel:
    def setup_method(self):
        _reset_model_cache()

    def teardown_method(self):
        _reset_model_cache()

    def test_returns_none_when_model_file_missing(self, tmp_path):
        with patch("app.services.ml_risk_scoring.MODEL_PATH", tmp_path / "missing.joblib"):
            result = ml_risk_scoring._load_model()
        assert result is None

    def test_loads_model_when_file_present(self, tmp_path):
        import joblib
        model_file = tmp_path / "risk_model.joblib"
        joblib.dump({"dummy": True}, model_file)
        with patch("app.services.ml_risk_scoring.MODEL_PATH", model_file):
            result = ml_risk_scoring._load_model()
        assert result is not None

    def test_caches_model_after_first_load(self, tmp_path):
        with patch("app.services.ml_risk_scoring.MODEL_PATH", tmp_path / "missing.joblib"):
            r1 = ml_risk_scoring._load_model()
            r2 = ml_risk_scoring._load_model()  # second call: no re-attempt
        assert r1 is None
        assert r2 is None
        assert ml_risk_scoring._model_loaded is True

    def test_returns_none_on_corrupt_file(self, tmp_path):
        bad_file = tmp_path / "bad.joblib"
        bad_file.write_bytes(b"not a joblib file")
        with patch("app.services.ml_risk_scoring.MODEL_PATH", bad_file):
            result = ml_risk_scoring._load_model()
        assert result is None


# ---------------------------------------------------------------------------
# compute_ml_risk_scores
# ---------------------------------------------------------------------------

class TestComputeMlRiskScores:
    def setup_method(self):
        _reset_model_cache()

    def teardown_method(self):
        _reset_model_cache()

    def test_returns_empty_dict_when_model_not_loaded(self):
        with patch.object(ml_risk_scoring, "_load_model", return_value=None):
            db = MagicMock()
            result = ml_risk_scoring.compute_ml_risk_scores(db, org_id=1)
        assert result == {}

    def test_returns_empty_dict_when_no_contracts(self):
        mock_model = MagicMock()
        with patch.object(ml_risk_scoring, "_load_model", return_value=mock_model):
            db = MagicMock()
            db.query.return_value.filter.return_value.all.return_value = []
            db.query.return_value.all.return_value = []
            result = ml_risk_scoring.compute_ml_risk_scores(db, org_id=1)
        assert result == {}

    def test_returns_ml_scores_for_contracts(self):
        future = date.today() + timedelta(days=200)
        contracts = [_make_contract(1, end_date=future, org_id=1)]

        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.6, 0.3, 0.1]])
        mock_model.predict.return_value = np.array([0])  # LOW

        fv_row = MagicMock()
        fv_row.contract_id = 1
        fv_row.total = 10000.0
        fv_row.std = 500.0
        fv_row.count = 12

        db = MagicMock()
        db.query.return_value.all.return_value = contracts
        db.query.return_value.filter.return_value.all.return_value = contracts
        db.query.return_value.filter.return_value.group_by.return_value.all.return_value = [fv_row]

        with patch.object(ml_risk_scoring, "_load_model", return_value=mock_model):
            result = ml_risk_scoring.compute_ml_risk_scores(db, org_id=1)

        assert 1 in result
        assert result[1]["mlScore"] == pytest.approx(0.1, abs=1e-4)
        assert result[1]["mlLevel"] == "LOW"

    def test_returns_empty_dict_when_prediction_raises(self):
        future = date.today() + timedelta(days=100)
        contracts = [_make_contract(1, end_date=future)]

        mock_model = MagicMock()
        mock_model.predict_proba.side_effect = RuntimeError("prediction error")

        fv_row = MagicMock()
        fv_row.contract_id = 1
        fv_row.total = 5000.0
        fv_row.std = 0.0
        fv_row.count = 1

        db = MagicMock()
        db.query.return_value.all.return_value = contracts
        db.query.return_value.filter.return_value.all.return_value = contracts
        db.query.return_value.filter.return_value.group_by.return_value.all.return_value = [fv_row]

        with patch.object(ml_risk_scoring, "_load_model", return_value=mock_model):
            result = ml_risk_scoring.compute_ml_risk_scores(db, org_id=1)

        assert result == {}

    def test_high_risk_contract_maps_correctly(self):
        past = date.today() - timedelta(days=5)
        contracts = [_make_contract(1, status="EXPIRED", end_date=past, org_id=1)]

        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.05, 0.10, 0.85]])
        mock_model.predict.return_value = np.array([2])  # HIGH

        fv_row = MagicMock()
        fv_row.contract_id = 1
        fv_row.total = 50000.0
        fv_row.std = 0.0
        fv_row.count = 6

        db = MagicMock()
        db.query.return_value.all.return_value = contracts
        db.query.return_value.filter.return_value.all.return_value = contracts
        db.query.return_value.filter.return_value.group_by.return_value.all.return_value = [fv_row]

        with patch.object(ml_risk_scoring, "_load_model", return_value=mock_model):
            result = ml_risk_scoring.compute_ml_risk_scores(db, org_id=1)

        assert result[1]["mlScore"] == pytest.approx(0.85, abs=1e-4)
        assert result[1]["mlLevel"] == "HIGH"

    def test_no_org_id_filter_queries_all_contracts(self):
        mock_model = MagicMock()
        with patch.object(ml_risk_scoring, "_load_model", return_value=mock_model):
            db = MagicMock()
            db.query.return_value.all.return_value = []
            ml_risk_scoring.compute_ml_risk_scores(db, org_id=None)
        # Without org_id, .filter() should NOT be called on Contract query
        db.query.return_value.filter.assert_not_called()
