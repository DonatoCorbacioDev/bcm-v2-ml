"""Tests for scripts/train_risk_model.py's non-trivial pure-function pieces:
the group-aware split (must not leak an organization across splits) and the
rule-based baseline (must match app/services/risk_scoring.py's formula)."""

import numpy as np
import pytest

from app.services.risk_features import FEATURES
from scripts.train_risk_model import (
    _group_train_val_test_split,
    _rule_based_baseline_predict,
)


class TestGroupTrainValTestSplit:
    def _make_data(self, n_orgs=20, per_org=10):
        n = n_orgs * per_org
        X = np.random.default_rng(1).normal(size=(n, 7))
        y = np.random.default_rng(2).integers(0, 3, size=n)
        groups = np.repeat(np.arange(n_orgs), per_org)
        return X, y, groups

    def test_splits_cover_all_rows_without_overlap(self):
        X, y, groups = self._make_data()
        train_idx, val_idx, test_idx = _group_train_val_test_split(X, y, groups)

        all_idx = np.concatenate([train_idx, val_idx, test_idx])
        assert len(all_idx) == len(X)
        assert len(set(all_idx.tolist())) == len(X)  # no duplicates

    def test_no_organization_spans_more_than_one_split(self):
        X, y, groups = self._make_data()
        train_idx, val_idx, test_idx = _group_train_val_test_split(X, y, groups)

        train_orgs = set(groups[train_idx].tolist())
        val_orgs = set(groups[val_idx].tolist())
        test_orgs = set(groups[test_idx].tolist())

        assert train_orgs.isdisjoint(val_orgs)
        assert train_orgs.isdisjoint(test_orgs)
        assert val_orgs.isdisjoint(test_orgs)

    def test_split_sizes_roughly_match_requested_proportions(self):
        X, y, groups = self._make_data(n_orgs=50, per_org=20)
        train_idx, val_idx, test_idx = _group_train_val_test_split(
            X, y, groups, val_size=0.20, test_size=0.20
        )
        n = len(X)
        # Group-based splitting can't hit exact proportions (whole orgs move
        # together) - just check it's in a sane ballpark.
        assert 0.5 < len(train_idx) / n < 0.75
        assert 0.1 < len(val_idx) / n < 0.30
        assert 0.1 < len(test_idx) / n < 0.30


class TestRuleBasedBaselinePredict:
    def _row(self, days, has_end_date, z):
        row = [0.0] * len(FEATURES)
        row[FEATURES.index("days_until_expiry")] = days
        row[FEATURES.index("has_end_date")] = has_end_date
        row[FEATURES.index("financial_zscore")] = z
        return row

    def test_expired_contract_alone_is_medium_not_high(self):
        # expiry_score=1.0 (EXPIRED), value_score=0.0 ->
        # risk_score = 0.6*1.0 + 0.4*0.0 = 0.6 -> MEDIUM (needs >= 0.65 for HIGH).
        # Unlike the ML training label rule, the rule-based score is not an
        # unconditional EXPIRED -> HIGH mapping - it also weighs the amount.
        X = np.array([self._row(days=-10, has_end_date=1, z=0.0)])
        assert _rule_based_baseline_predict(X)[0] == 1

    def test_expired_contract_with_unusual_amount_is_high(self):
        X = np.array([self._row(days=-10, has_end_date=1, z=3.0)])
        assert _rule_based_baseline_predict(X)[0] == 2

    def test_no_end_date_is_low_or_medium_not_high(self):
        X = np.array([self._row(days=365, has_end_date=0, z=0.0)])
        # expiry_score=0.3 -> risk_score=0.18 -> LOW
        assert _rule_based_baseline_predict(X)[0] == 0

    def test_far_future_low_zscore_is_low(self):
        X = np.array([self._row(days=400, has_end_date=1, z=0.1)])
        assert _rule_based_baseline_predict(X)[0] == 0

    def test_extreme_zscore_alone_reaches_medium_not_high(self):
        # expiry_score=0.1 (far future), value_score=1.0 (|z|>=3) ->
        # risk_score = 0.6*0.1 + 0.4*1.0 = 0.46 -> MEDIUM, not HIGH alone.
        X = np.array([self._row(days=400, has_end_date=1, z=5.0)])
        assert _rule_based_baseline_predict(X)[0] == 1

    def test_expiring_soon_and_high_zscore_combine_to_high(self):
        # expiry_score=0.8 (<30d), value_score=1.0 -> risk_score=0.88 -> HIGH
        X = np.array([self._row(days=10, has_end_date=1, z=5.0)])
        assert _rule_based_baseline_predict(X)[0] == 2

    def test_returns_one_label_per_row(self):
        X = np.array([
            self._row(days=-5, has_end_date=1, z=0.0),
            self._row(days=400, has_end_date=1, z=0.0),
            self._row(days=100, has_end_date=1, z=0.0),
        ])
        labels = _rule_based_baseline_predict(X)
        assert len(labels) == 3
        assert set(labels.tolist()) <= {0, 1, 2}
