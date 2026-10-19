# Copyright (c) 2025 Donato Corbacio. All rights reserved.
# Licensed under the terms of the LICENSE file at the repository root.

"""
Backtest Prophet's financial forecast (app/services/forecasting.py) against
held-out synthetic monthly series, to answer a question MODEL_CARD.md's
section 1 ("Financial forecasting") never had an answer to: HOW accurate is
the forecast, in the same regression-error terms (MAE, RMSE, MAPE, RAE, RSE)
the risk classifier already reports for its own task (section 2)?

Unlike this project's classifier (train/val/test split over many synthetic
contracts), there is no training here to evaluate - Prophet fits fresh on
whatever history it's given, every time (see forecasting.py's own
docstring). "Backtesting" here means: hide the last `holdout` months of a
known series, fit on what's left, forecast forward, and compare the
forecast against the real (hidden) values - walk-forward validation,
standard practice for time series since there's no other way to get a
held-out answer for "the future" than to pretend part of the past still is
one.

Calls the REAL `_prophet_forecast` (app.services.forecasting) directly on
a plain ds/y DataFrame - no database, no FastAPI app, same "exercise
production code, don't reimplement it" posture as the other eval scripts
in this directory. Scenarios are hand-crafted synthetic series (trend +
yearly seasonality + noise, numpy, fixed seed) rather than real financial
data, so results describe how well Prophet recovers a KNOWN synthetic
pattern - a sanity check on the method, not a claim about real portfolios
(same caveat MODEL_CARD.md already makes about the risk classifier's
synthetic training data).

Every metric is also computed for a trivial "repeat the last known month"
baseline - Prophet has to beat that to be worth running at all, same
posture as section 2.2's "baselines the ML model must beat to justify
existing."

Run:
    python scripts/backtest_forecast.py
"""

import os

os.environ.setdefault("DB_URL", "sqlite:///:memory:")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.forecasting import _prophet_forecast  # noqa: E402

HOLDOUT_MONTHS = 3


def synthetic_series(
    n_months: int, trend_per_month: float, yearly_amplitude: float, noise_std: float, seed: int,
) -> pd.DataFrame:
    """A monthly ds/y series with a known trend + yearly seasonal pattern +
    Gaussian noise - the "ground truth" a correct forecast should recover.
    Fixed seed per scenario so results are reproducible run to run."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n_months, freq="MS")
    t = np.arange(n_months)
    base = 20000 + trend_per_month * t
    seasonal = yearly_amplitude * np.sin(2 * np.pi * t / 12)
    noise = rng.normal(0, noise_std, n_months)
    y = np.clip(base + seasonal + noise, 0, None)
    return pd.DataFrame({"ds": dates, "y": y})


SCENARIOS = [
    {
        "name": "clean-trend-seasonality",
        "series": synthetic_series(30, trend_per_month=300, yearly_amplitude=4000, noise_std=500, seed=1),
    },
    {
        "name": "high-noise",
        "series": synthetic_series(30, trend_per_month=300, yearly_amplitude=4000, noise_std=4000, seed=2),
    },
    {
        "name": "short-history-no-seasonality-fit",
        # Below the n>=24 threshold _prophet_forecast uses to enable yearly
        # seasonality (see forecasting.py) - this is the regime
        # MODEL_CARD.md's `reliable: false` flag exists to warn callers
        # about; this scenario measures how much worse it actually is.
        "series": synthetic_series(14, trend_per_month=300, yearly_amplitude=4000, noise_std=500, seed=3),
    },
]


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs((actual - predicted) / actual)) * 100)


def rae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Relative Absolute Error: total absolute error vs. the error a
    "always predict the mean" baseline would make. < 1.0 means the
    forecast beats that trivial baseline; >= 1.0 means it does not."""
    mean_baseline_error = np.sum(np.abs(actual - np.mean(actual)))
    if mean_baseline_error == 0:
        return float("nan")
    return float(np.sum(np.abs(actual - predicted)) / mean_baseline_error)


def rse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Relative Squared Error - same idea as RAE, squared (equivalent to
    1 - R^2 against the mean baseline)."""
    mean_baseline_error = np.sum((actual - np.mean(actual)) ** 2)
    if mean_baseline_error == 0:
        return float("nan")
    return float(np.sum((actual - predicted) ** 2) / mean_baseline_error)


def all_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    return {
        "MAE": mae(actual, predicted),
        "RMSE": rmse(actual, predicted),
        "MAPE%": mape(actual, predicted),
        "RAE": rae(actual, predicted),
        "RSE": rse(actual, predicted),
    }


def naive_forecast(train: pd.DataFrame, h: int) -> np.ndarray:
    """Trivial baseline: repeat the last known month's value h times."""
    return np.full(h, train["y"].iloc[-1])


def run_scenario(scenario: dict) -> dict:
    series = scenario["series"]
    train = series.iloc[:-HOLDOUT_MONTHS]
    holdout = series.iloc[-HOLDOUT_MONTHS:]
    actual = holdout["y"].to_numpy()

    prophet_result = _prophet_forecast(train, HOLDOUT_MONTHS)
    prophet_predicted = np.array([row["amount"] for row in prophet_result])
    naive_predicted = naive_forecast(train, HOLDOUT_MONTHS)

    return {
        "scenario": scenario["name"],
        "train_months": len(train),
        "prophet": all_metrics(actual, prophet_predicted),
        "naive": all_metrics(actual, naive_predicted),
    }


def print_result(result: dict) -> None:
    print(f"--- {result['scenario']} (trained on {result['train_months']} months) ---")
    print(f"{'metric':<8} {'prophet':>12} {'naive (last-value)':>20}")
    for key in result["prophet"]:
        print(f"{key:<8} {result['prophet'][key]:>12.2f} {result['naive'][key]:>20.2f}")
    beats_naive = result["prophet"]["RMSE"] < result["naive"]["RMSE"]
    print(f"Prophet {'BEATS' if beats_naive else 'DOES NOT BEAT'} the naive baseline on RMSE.")
    print()


def main() -> None:
    results = [run_scenario(s) for s in SCENARIOS]
    for result in results:
        print_result(result)

    print("=== Summary ===")
    for result in results:
        beats_naive = result["prophet"]["RMSE"] < result["naive"]["RMSE"]
        print(
            f"{result['scenario']}: RMSE={result['prophet']['RMSE']:.2f} "
            f"(naive={result['naive']['RMSE']:.2f}), "
            f"{'beats' if beats_naive else 'DOES NOT beat'} naive baseline"
        )


if __name__ == "__main__":
    main()
