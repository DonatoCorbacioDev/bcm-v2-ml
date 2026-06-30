"""
Generate a synthetic contract dataset for training the ML risk scoring model.

Output: data/synthetic_contracts.csv

Run:
    python scripts/generate_training_data.py [--samples N] [--seed S]

No database connection required. All data is generated statistically.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = [
    "days_until_expiry",
    "status_code",
    "has_end_date",
    "total_financial_amount",
    "num_financial_records",
    "financial_std",
    "financial_zscore",
]
STATUS_LABELS = {0: "ACTIVE", 1: "EXPIRED", 2: "CANCELLED", 3: "DRAFT"}
N_ORGS = 20


def _assign_labels(days: np.ndarray, status: np.ndarray, z: np.ndarray) -> np.ndarray:
    labels = np.zeros(len(days), dtype=int)
    medium = (days < 180) | (np.abs(z) > 1.0)
    high = (status == 1) | (days < 30) | (np.abs(z) > 2.5)
    labels[medium] = 1
    labels[high] = 2
    return labels


def generate(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # --- status ---
    status = rng.choice([0, 1, 2, 3], size=n_samples, p=[0.60, 0.25, 0.10, 0.05])

    # --- days_until_expiry ---
    days = np.where(
        status == 1,
        rng.integers(-365, 0, size=n_samples),   # EXPIRED: past
        rng.integers(-15, 730, size=n_samples),   # others: mostly future
    )
    # CANCELLED/DRAFT: positive expiry or no real expiry
    days = np.where(
        np.isin(status, [2, 3]),
        rng.integers(30, 730, size=n_samples),
        days,
    )

    # --- has_end_date ---
    has_end_date = np.where(
        np.isin(status, [2, 3]),
        rng.integers(0, 2, size=n_samples),  # 50% no end date for CANCELLED/DRAFT
        1,
    )
    # A fraction of ACTIVE contracts also lack an end date
    active_no_end = (status == 0) & (rng.random(n_samples) < 0.05)
    has_end_date[active_no_end] = 0
    days[has_end_date == 0] = 365  # placeholder when no end date

    # --- financial amounts (log-normal, EUR) ---
    base = rng.lognormal(mean=10.0, sigma=1.2, size=n_samples)
    outlier_mask = rng.random(n_samples) < 0.10
    amounts = np.where(outlier_mask, base * rng.uniform(5, 20, size=n_samples), base)

    # --- num_financial_records ---
    num_records = rng.integers(1, 36, size=n_samples)

    # --- financial std (fraction of base amount) ---
    fin_std = amounts * rng.uniform(0.05, 0.35, size=n_samples)

    # --- z-score per organisation ---
    org_ids = rng.integers(1, N_ORGS + 1, size=n_samples)
    df_tmp = pd.DataFrame({"org_id": org_ids, "amount": amounts})
    org_stats = df_tmp.groupby("org_id")["amount"].agg(["mean", "std"]).fillna(1.0)
    org_mean = np.array([org_stats.loc[o, "mean"] if o in org_stats.index else 1.0 for o in org_ids])
    org_std = np.array([max(org_stats.loc[o, "std"], 1e-9) if o in org_stats.index else 1.0 for o in org_ids])
    z_scores = (amounts - org_mean) / org_std

    # --- ground truth labels ---
    labels = _assign_labels(days, status, z_scores)

    # 5% label noise
    noise_idx = rng.choice(n_samples, size=int(0.05 * n_samples), replace=False)
    labels[noise_idx] = rng.integers(0, 3, size=len(noise_idx))

    df = pd.DataFrame({
        "days_until_expiry": days.astype(float),
        "status_code": status.astype(int),
        "has_end_date": has_end_date.astype(int),
        "total_financial_amount": amounts,
        "num_financial_records": num_records.astype(int),
        "financial_std": fin_std,
        "financial_zscore": z_scores,
        "risk_level": labels.astype(int),
    })
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic contract training data")
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = generate(n_samples=args.samples, seed=args.seed)

    out = Path(__file__).parent.parent / "data" / "synthetic_contracts.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)

    print(f"Generated {len(df):,} samples → {out}")
    print("\nClass distribution:")
    print(df["risk_level"].value_counts().sort_index().rename({0: "LOW", 1: "MEDIUM", 2: "HIGH"}))
    print(f"\nFeatures: {FEATURES}")


if __name__ == "__main__":
    main()
