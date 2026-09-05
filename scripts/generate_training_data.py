"""
Generate a synthetic contract dataset for training the ML risk scoring model.

Generates synthetic domain entities (contracts + monthly financial values)
first, then derives the 7 model features by calling the exact same
app.services.risk_features.build_feature_row function ml_risk_scoring.py uses
at serving time. This closes the training/serving skew that existed when this
script fabricated the 7 feature columns directly (see
docs/research/dataset_sources.md for the history).

Labels are sampled from class probabilities driven by the visible signal
PLUS a latent "counterparty reliability" factor that is never exposed as a
feature (see _sample_latent_reliability) — unlike the previous deterministic
threshold rule (identical to a subset of the visible features, hence
trivially learnable), no model fit only on the 7 visible features can
reproduce this exactly, by construction.

Output: data/synthetic_contracts.csv + data/synthetic_contracts_manifest.json

Run:
    python scripts/generate_training_data.py [--samples N] [--seed S] [--as-of-date YYYY-MM-DD]

No database connection required. All data is generated statistically.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# app/ is a sibling package, not installed — make it importable so training
# can't compute features differently than ml_risk_scoring.py does at serving
# time (see app/services/risk_features.py for why this module exists).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.services.risk_features import (  # noqa: E402
    FEATURES,
    STATUS_CODE,
    build_feature_row,
    build_org_stats,
)

STATUSES = list(STATUS_CODE.keys())  # ACTIVE, EXPIRED, CANCELLED, DRAFT
STATUS_WEIGHTS = [0.60, 0.25, 0.10, 0.05]
N_ORGS = 20

# LogNormal(mu, sigma) calibrated on ANAC "forniture e servizi" open data
# (median ~EUR 44,000, P90 ~EUR 350,000). Duplicated as plain constants here
# rather than imported from scripts/seed_synthetic_data.py, which is a
# DB-writing production tool this dependency-free script must not depend on.
# See scripts/seed_synthetic_data.py for the full sourcing/license notes.
ANAC_LOGNORMAL_MU = 10.69
ANAC_LOGNORMAL_SIGMA = 1.52


@dataclass
class SyntheticContract:
    """Shape-compatible with the ORM Contract just enough for
    build_feature_row and risk_scoring._build_org_stats to treat it exactly
    like a real one — that compatibility is what makes the feature-parity
    test below meaningful."""

    id: int
    organization_id: int
    status: str
    end_date: date | None


def _sample_status(rng: np.random.Generator) -> str:
    return str(rng.choice(STATUSES, p=STATUS_WEIGHTS))


def _sample_end_date(rng: np.random.Generator, today: date, status: str) -> date | None:
    if status == "EXPIRED":
        return today - timedelta(days=int(rng.integers(1, 366)))
    if status in ("CANCELLED", "DRAFT"):
        if rng.random() < 0.5:
            return None
        return today + timedelta(days=int(rng.integers(30, 731)))
    # ACTIVE: mostly future; a fraction genuinely open-ended (no end date).
    if rng.random() < 0.15:
        return None
    return today + timedelta(days=int(rng.integers(-15, 731)))


def _sample_monthly_amounts(rng: np.random.Generator, is_outlier: bool) -> list[float]:
    """One financial_values-shaped monthly series per contract: a log-normal
    total split across 1-35 months with local noise. Mirrors the shape (not
    the exact seasonality model, which is DB/org-specific) of the real
    demo-data generator in scripts/seed_synthetic_data.py."""
    base = float(rng.lognormal(mean=ANAC_LOGNORMAL_MU, sigma=ANAC_LOGNORMAL_SIGMA))
    base = max(1_000.0, min(base, 5_000_000.0))
    if is_outlier:
        base *= float(rng.choice([0.25, 0.40, 1.80, 2.50, 3.50]))
    n_months = int(rng.integers(1, 36))
    noise = rng.uniform(0.65, 1.35, size=n_months)
    return (base / n_months * noise).tolist()


def _visible_risk_signal(days: np.ndarray, status_code: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Continuous risk signal in [0, 1] from the same visible information the
    old deterministic rule used — expiry proximity and financial z-score."""
    expiry_signal = np.where(
        status_code == STATUS_CODE["EXPIRED"],
        1.0,
        np.clip(1.0 - days / 200.0, 0.0, 1.0),
    )
    value_signal = np.clip(np.abs(z) / 3.0, 0.0, 1.0)
    # Weighted so an expired contract alone (expiry_signal=1.0) already sits
    # comfortably above the HIGH sigmoid midpoint (0.65) before the latent
    # reliability factor is even applied — an expired contract is close to
    # unconditionally high-risk, same as the old deterministic rule treated it.
    return 0.75 * expiry_signal + 0.25 * value_signal


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _sample_latent_reliability(rng: np.random.Generator, n: int) -> np.ndarray:
    """A per-contract factor that influences the true outcome but is NEVER
    exposed as a feature — a stand-in for real-world information no
    synthetic dataset built from visible fields alone can capture
    (counterparty financial health, relationship history, market
    conditions). Beta(2, 2) centers near 0.5 with most mass away from the
    extremes, so it meaningfully perturbs the label without swamping the
    visible signal entirely."""
    return rng.beta(2.0, 2.0, size=n)


def _label_probabilities(
    days: np.ndarray, status_code: np.ndarray, z: np.ndarray, reliability: np.ndarray
) -> np.ndarray:
    """P(LOW), P(MEDIUM), P(HIGH) per row — deterministic given all inputs,
    including `reliability` (the randomness lives only in how that latent
    value itself is drawn; see _sample_latent_reliability). Soft sigmoid
    boundaries instead of hard cutoffs mean two contracts with an identical
    visible signal can land in different classes, same as real outcomes
    would — this replaces the old approach of a hard threshold rule plus 5%
    uniform label noise bolted on afterward."""
    visible = _visible_risk_signal(days, status_code, z)
    # Low reliability (unreliable counterparty) pushes risk up; high
    # reliability pulls it down. The model never sees `reliability`, so even
    # a perfect fit to the visible features can't reproduce this exactly —
    # that irreducible gap is the point (see docs/research/dataset_sources.md).
    combined = np.clip(visible + 0.35 * (0.5 - reliability), 0.0, 1.0)

    p_high = _sigmoid((combined - 0.65) / 0.08)
    p_low = 1.0 - _sigmoid((combined - 0.35) / 0.08)
    p_medium = np.clip(1.0 - p_high - p_low, 0.0, 1.0)

    probs = np.stack([p_low, p_medium, p_high], axis=1)
    return probs / probs.sum(axis=1, keepdims=True)


def _assign_labels(rng: np.random.Generator, days: np.ndarray, status_code: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Ground-truth label — sampled from _label_probabilities, not a
    deterministic threshold. The label mechanism (visible signal + latent
    reliability + soft boundaries) is the single source of truth also
    documented in docs/research/dataset_sources.md and README_ML.md; keep
    all three in sync if this changes."""
    reliability = _sample_latent_reliability(rng, len(days))
    probs = _label_probabilities(days, status_code, z, reliability)
    return np.array([rng.choice(3, p=p) for p in probs])


def generate(n_samples: int = 5000, seed: int = 42, as_of_date: date | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    today = as_of_date or date.today()

    contracts: list[SyntheticContract] = []
    totals: dict[int, float] = {}
    counts: dict[int, int] = {}
    stds: dict[int, float] = {}

    outlier_ids = set(
        rng.choice(n_samples, size=max(1, int(0.10 * n_samples)), replace=False).tolist()
    )

    for i in range(n_samples):
        contract_id = i + 1
        org_id = int(rng.integers(1, N_ORGS + 1))
        status = _sample_status(rng)
        end_date = _sample_end_date(rng, today, status)
        contracts.append(
            SyntheticContract(id=contract_id, organization_id=org_id, status=status, end_date=end_date)
        )

        monthly = _sample_monthly_amounts(rng, i in outlier_ids)
        totals[contract_id] = float(np.sum(monthly))
        counts[contract_id] = len(monthly)
        stds[contract_id] = float(np.std(monthly)) if len(monthly) > 1 else 0.0

    # Same helper risk_scoring.py/ml_risk_scoring.py use at serving time —
    # training and serving compute the per-organization z-score baseline
    # identically.
    org_stats = build_org_stats(contracts, totals)

    feature_rows = [
        build_feature_row(
            c.end_date, c.status, totals[c.id], counts[c.id], stds[c.id],
            c.organization_id, org_stats, today,
        )
        for c in contracts
    ]
    X = np.array(feature_rows, dtype=float)
    df = pd.DataFrame(X, columns=FEATURES)

    days = df["days_until_expiry"].to_numpy()
    status_code = df["status_code"].to_numpy()
    z = df["financial_zscore"].to_numpy()
    labels = _assign_labels(rng, days, status_code, z)

    # Not a model feature - kept only so train_risk_model.py can split
    # train/validation/test by organization instead of by row (a random
    # per-row split would let contracts from the same organization leak
    # across splits via the shared org-level z-score baseline).
    df["organization_id"] = [c.organization_id for c in contracts]
    df["risk_level"] = labels.astype(int)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic contract training data")
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--as-of-date",
        type=str,
        default=None,
        help="Reference date (YYYY-MM-DD) days_until_expiry/status are computed relative to. "
             "Defaults to today. Pin this for a dataset that is reproducible independent of "
             "the day the script happens to run.",
    )
    args = parser.parse_args()

    as_of = date.fromisoformat(args.as_of_date) if args.as_of_date else date.today()
    df = generate(n_samples=args.samples, seed=args.seed, as_of_date=as_of)

    out_dir = Path(__file__).parent.parent / "data"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "synthetic_contracts.csv"
    manifest_path = out_dir / "synthetic_contracts_manifest.json"

    df.to_csv(csv_path, index=False)

    class_counts = df["risk_level"].value_counts().sort_index()
    manifest = {
        "seed": args.seed,
        "n_samples": args.samples,
        "as_of_date": as_of.isoformat(),
        "features": FEATURES,
        "class_distribution": {
            "LOW": int(class_counts.get(0, 0)),
            "MEDIUM": int(class_counts.get(1, 0)),
            "HIGH": int(class_counts.get(2, 0)),
        },
        "generator": "scripts/generate_training_data.py",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"Generated {len(df):,} samples -> {csv_path}")
    print(f"Manifest -> {manifest_path}")
    print("\nClass distribution:")
    print(class_counts.rename({0: "LOW", 1: "MEDIUM", 2: "HIGH"}))
    print(f"\nFeatures: {FEATURES}")


if __name__ == "__main__":
    main()
