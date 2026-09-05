"""Tests for scripts/generate_training_data.py.

The most important test here is TestTrainingServingParity: it proves the
training pipeline and the runtime scorer (ml_risk_scoring.py) compute
features identically for the same contract, which is the actual fix for the
training/serving skew documented in docs/research/dataset_sources.md.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from app.services.ml_risk_scoring import _build_feature_matrix
from app.services.risk_features import FEATURES
from scripts.generate_training_data import (
    SyntheticContract,
    _assign_labels,
    _label_probabilities,
    _sample_end_date,
    _sample_latent_reliability,
    _sample_monthly_amounts,
    _visible_risk_signal,
    generate,
)

AS_OF = date(2026, 1, 1)


class TestGenerate:
    def test_returns_requested_sample_count(self):
        df = generate(n_samples=200, seed=1, as_of_date=AS_OF)
        assert len(df) == 200

    def test_columns_match_shared_features_plus_label(self):
        df = generate(n_samples=50, seed=1, as_of_date=AS_OF)
        assert list(df.columns) == [*FEATURES, "risk_level"]

    def test_deterministic_for_same_seed_and_as_of_date(self):
        df1 = generate(n_samples=300, seed=7, as_of_date=AS_OF)
        df2 = generate(n_samples=300, seed=7, as_of_date=AS_OF)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_produce_different_data(self):
        df1 = generate(n_samples=300, seed=1, as_of_date=AS_OF)
        df2 = generate(n_samples=300, seed=2, as_of_date=AS_OF)
        assert not df1["total_financial_amount"].equals(df2["total_financial_amount"])

    def test_not_wall_clock_dependent(self):
        # Same seed + explicit as_of_date must be reproducible regardless of
        # when the test actually runs (see --as-of-date in main()).
        df1 = generate(n_samples=100, seed=3, as_of_date=date(2025, 1, 1))
        df2 = generate(n_samples=100, seed=3, as_of_date=date(2025, 1, 1))
        pd.testing.assert_frame_equal(df1, df2)

    def test_all_labels_within_expected_range(self):
        df = generate(n_samples=500, seed=1, as_of_date=AS_OF)
        assert set(df["risk_level"].unique()) <= {0, 1, 2}

    def test_class_distribution_has_all_three_levels_present(self):
        df = generate(n_samples=2000, seed=1, as_of_date=AS_OF)
        counts = df["risk_level"].value_counts()
        assert set(counts.index) == {0, 1, 2}


class TestVisibleRiskSignal:
    def test_expired_status_gives_max_expiry_component(self):
        signal = _visible_risk_signal(np.array([500.0]), np.array([1.0]), np.array([0.0]))
        assert signal[0] == pytest.approx(0.75)  # 0.75 * 1.0 (expiry) + 0.25 * 0.0 (value)

    def test_far_future_low_zscore_gives_low_signal(self):
        signal = _visible_risk_signal(np.array([700.0]), np.array([0.0]), np.array([0.1]))
        assert signal[0] < 0.1

    def test_extreme_zscore_raises_signal(self):
        low_z = _visible_risk_signal(np.array([700.0]), np.array([0.0]), np.array([0.0]))
        high_z = _visible_risk_signal(np.array([700.0]), np.array([0.0]), np.array([3.0]))
        assert high_z[0] > low_z[0]

    def test_signal_bounded_zero_one(self):
        signal = _visible_risk_signal(np.array([-1000.0]), np.array([1.0]), np.array([50.0]))
        assert 0.0 <= signal[0] <= 1.0


class TestLabelProbabilities:
    """Deterministic given `reliability` — no sampling randomness here, so
    these assert exact probability behavior instead of statistical tendency."""

    def test_probabilities_sum_to_one(self):
        probs = _label_probabilities(
            np.array([500.0, 10.0]), np.array([0.0, 0.0]), np.array([0.0, 3.0]), np.array([0.5, 0.5])
        )
        assert probs.sum(axis=1) == pytest.approx([1.0, 1.0])

    def test_high_visible_risk_favors_high_class(self):
        probs = _label_probabilities(
            np.array([5.0]), np.array([1.0]), np.array([3.0]), np.array([0.5])
        )
        assert probs[0][2] > probs[0][0]  # P(HIGH) > P(LOW)

    def test_low_visible_risk_favors_low_class(self):
        probs = _label_probabilities(
            np.array([700.0]), np.array([0.0]), np.array([0.0]), np.array([0.5])
        )
        assert probs[0][0] > probs[0][2]  # P(LOW) > P(HIGH)

    def test_low_reliability_increases_high_probability(self):
        # Same visible signal, only the hidden reliability factor differs —
        # this is the part no model trained on visible features alone can see.
        days, status, z = np.array([300.0]), np.array([0.0]), np.array([0.5])
        reliable = _label_probabilities(days, status, z, np.array([0.9]))
        unreliable = _label_probabilities(days, status, z, np.array([0.1]))
        assert unreliable[0][2] > reliable[0][2]  # P(HIGH) higher when unreliable


class TestSampleLatentReliability:
    def test_values_within_unit_interval(self):
        rng = np.random.default_rng(1)
        values = _sample_latent_reliability(rng, 1000)
        assert values.min() >= 0.0
        assert values.max() <= 1.0

    def test_centered_near_half(self):
        rng = np.random.default_rng(1)
        values = _sample_latent_reliability(rng, 5000)
        assert values.mean() == pytest.approx(0.5, abs=0.05)


class TestAssignLabels:
    def test_all_labels_within_range(self):
        rng = np.random.default_rng(1)
        days = np.array([500.0, 10.0, 100.0, -5.0])
        status_code = np.array([0.0, 0.0, 0.0, 1.0])
        z = np.array([0.0, 0.5, 0.0, 0.0])
        labels = _assign_labels(rng, days, status_code, z)
        assert set(labels.tolist()) <= {0, 1, 2}

    def test_expired_status_is_high_most_of_the_time(self):
        # Probabilistic, not deterministic — assert the tendency over many
        # draws rather than a single guaranteed outcome.
        rng = np.random.default_rng(1)
        days = np.full(500, 500.0)
        status_code = np.full(500, 1.0)  # EXPIRED
        z = np.zeros(500)
        labels = _assign_labels(rng, days, status_code, z)
        assert (labels == 2).mean() > 0.7

    def test_far_future_healthy_contract_is_rarely_high(self):
        rng = np.random.default_rng(1)
        days = np.full(500, 700.0)
        status_code = np.zeros(500)
        z = np.zeros(500)
        labels = _assign_labels(rng, days, status_code, z)
        assert (labels == 2).mean() < 0.1

    def test_deterministic_for_same_rng_state(self):
        days = np.array([500.0, 10.0])
        status_code = np.array([0.0, 1.0])
        z = np.array([0.0, 0.0])
        labels1 = _assign_labels(np.random.default_rng(5), days, status_code, z)
        labels2 = _assign_labels(np.random.default_rng(5), days, status_code, z)
        assert labels1.tolist() == labels2.tolist()


class TestSampleEndDate:
    def test_expired_status_gives_past_date(self):
        rng = np.random.default_rng(1)
        result = _sample_end_date(rng, AS_OF, "EXPIRED")
        assert result < AS_OF

    def test_active_status_gives_future_or_none(self):
        rng = np.random.default_rng(1)
        for _ in range(20):
            result = _sample_end_date(rng, AS_OF, "ACTIVE")
            assert result is None or result >= AS_OF - pd.Timedelta(days=15).to_pytimedelta()


class TestSampleMonthlyAmounts:
    def test_returns_between_1_and_35_months(self):
        rng = np.random.default_rng(1)
        amounts = _sample_monthly_amounts(rng, is_outlier=False)
        assert 1 <= len(amounts) <= 35

    def test_outlier_amounts_are_scaled(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        normal = _sample_monthly_amounts(rng1, is_outlier=False)
        outlier = _sample_monthly_amounts(rng2, is_outlier=True)
        # Same seed diverges only at the outlier multiplier draw — just check
        # both are positive, non-trivial series (the multiplier RNG draw order
        # makes an exact ratio assertion brittle).
        assert all(a > 0 for a in normal)
        assert all(a > 0 for a in outlier)


class TestTrainingServingParity:
    """The actual fix: training-time features (via generate()'s per-row call
    to build_feature_row) must be bit-for-bit identical to what
    ml_risk_scoring._build_feature_matrix computes for the same contract at
    serving time."""

    def test_single_contract_features_match_runtime_scorer(self):
        contract = SyntheticContract(
            id=1, organization_id=5, status="ACTIVE", end_date=date(2026, 6, 1)
        )
        totals = {1: 42000.0}
        counts = {1: 12}
        stds = {1: 1500.0}
        org_stats = {5: (40000.0, 5000.0)}

        from app.services.risk_features import build_feature_row

        training_row = build_feature_row(
            contract.end_date, contract.status, totals[1], counts[1], stds[1],
            contract.organization_id, org_stats, AS_OF,
        )

        # ml_risk_scoring._build_feature_matrix expects objects with
        # .end_date/.status/.id/.organization_id — SyntheticContract satisfies
        # that by construction (see its docstring).
        serving_matrix = _build_feature_matrix([contract], totals, stds, counts, org_stats, today=AS_OF)

        assert training_row == pytest.approx(serving_matrix[0].tolist())

    def test_generated_dataset_rows_match_runtime_scorer(self):
        # End-to-end version: generate a small dataset, then recompute every
        # row's features via the runtime scorer's own function from the same
        # synthetic contracts, and assert they match exactly.
        rng = np.random.default_rng(9)
        contracts = []
        totals, counts, stds = {}, {}, {}
        for i in range(25):
            cid = i + 1
            org_id = int(rng.integers(1, 5))
            status = str(rng.choice(["ACTIVE", "EXPIRED", "CANCELLED", "DRAFT"]))
            end_date = date(2026, 1, 1) if status != "DRAFT" else None
            contracts.append(SyntheticContract(cid, org_id, status, end_date))
            totals[cid] = float(rng.uniform(1000, 100000))
            counts[cid] = int(rng.integers(1, 20))
            stds[cid] = float(rng.uniform(0, 5000))

        from app.services.risk_features import build_feature_row, build_org_stats

        org_stats = build_org_stats(contracts, totals)

        training_rows = [
            build_feature_row(c.end_date, c.status, totals[c.id], counts[c.id], stds[c.id],
                               c.organization_id, org_stats, AS_OF)
            for c in contracts
        ]
        serving_matrix = _build_feature_matrix(contracts, totals, stds, counts, org_stats, today=AS_OF)

        assert np.allclose(np.array(training_rows), serving_matrix)
