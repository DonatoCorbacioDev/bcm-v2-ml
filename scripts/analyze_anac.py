#!/usr/bin/env python3
"""
Analyze ANAC (Autorità Nazionale Anticorruzione) open-data CSV files and
extract statistical calibration parameters for the BCM synthetic data generator.

Data source: https://dati.anticorruzione.it/opendata/dataset
License: CC BY 4.0 / Italian Open Data License 2.0 (IODL 2.0)
This script reads the file locally and prints calibration parameters only.
The original ANAC data is NOT stored in or distributed with this project.

Usage:
    python scripts/analyze_anac.py --file path/to/anac_contratti.csv
    python scripts/analyze_anac.py --file path/to/anac_contratti.csv --amount-col importoAggiudicazione
    python scripts/analyze_anac.py --help

Typical ANAC CSV columns (may vary by dataset year):
    importoAggiudicazione  — contract award amount (EUR)
    dataInizioContratto    — contract start date (YYYY-MM-DD or DD/MM/YYYY)
    cpv                    — product/service category code
    codiceCIG              — unique contract identifier

The script outputs ready-to-paste Python constants for seed_synthetic_data.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", required=True, help="Path to the ANAC CSV file")
    p.add_argument("--amount-col", default="importoAggiudicazione",
                   help="Column name for contract amounts (default: importoAggiudicazione)")
    p.add_argument("--date-col", default="dataInizioContratto",
                   help="Column name for contract start date (default: dataInizioContratto)")
    p.add_argument("--sep", default=";", help="CSV separator (default: ;)")
    p.add_argument("--min-amount", type=float, default=1_000.0,
                   help="Exclude amounts below this value (noise filter, default: 1000)")
    p.add_argument("--max-amount", type=float, default=50_000_000.0,
                   help="Exclude amounts above this value (mega-projects skew SME params, default: 50M)")
    p.add_argument("--sample", type=int, default=0,
                   help="Analyze a random sample of N rows (0 = all rows)")
    return p.parse_args()


def load_csv(path: str, sep: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep=sep, low_memory=False, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, sep=sep, low_memory=False, encoding="latin-1")
    print(f"Loaded {len(df):,} rows from {Path(path).name}")
    print(f"Columns: {list(df.columns)}\n")
    return df


def fit_lognormal(amounts: np.ndarray) -> tuple[float, float]:
    log_amounts = np.log(amounts)
    mu = float(np.mean(log_amounts))
    sigma = float(np.std(log_amounts, ddof=1))
    return mu, sigma


def compute_monthly_seasonality(df: pd.DataFrame, date_col: str, amount_col: str) -> list[float]:
    """
    Compute monthly seasonality indices from contract start dates and amounts.
    Index > 1.0 means that month has above-average activity.
    Normalized so the 12-element list sums to 12.0 (average = 1.0).
    """
    dates = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    valid_mask = dates.notna() & df[amount_col].notna()
    monthly_amounts = pd.Series(
        df.loc[valid_mask, amount_col].values,
        index=dates[valid_mask],
        name="amount",
    )
    monthly_sums = monthly_amounts.groupby(monthly_amounts.index.month).sum()
    counts = monthly_amounts.groupby(monthly_amounts.index.month).count()
    avg_per_month = monthly_sums / counts

    overall_avg = float(avg_per_month.mean())
    indices = []
    for m in range(1, 13):
        if m in avg_per_month.index:
            indices.append(round(float(avg_per_month[m]) / overall_avg, 3))
        else:
            indices.append(1.0)

    # Normalize so sum = 12
    total = sum(indices)
    indices = [round(v * 12.0 / total, 3) for v in indices]
    return indices


def print_report(
    amounts: np.ndarray,
    mu: float,
    sigma: float,
    seasonality: list[float] | None,
    dataset_path: str,
) -> None:
    median = float(np.exp(mu))
    mean = float(np.exp(mu + sigma ** 2 / 2))
    p10 = float(np.exp(mu - 1.28 * sigma))
    p90 = float(np.exp(mu + 1.28 * sigma))

    print("=" * 60)
    print("ANAC CALIBRATION REPORT")
    print(f"Source: {Path(dataset_path).name} (CC BY 4.0 / IODL 2.0)")
    print(f"Sample size: {len(amounts):,} contracts (after filters)")
    print("=" * 60)
    print()
    print("-- Amount distribution (LogNormal fit) --")
    print(f"  mu    = {mu:.4f}   (log-mean)")
    print(f"  sigma = {sigma:.4f}   (log-std)")
    print(f"  Implied median : EUR {median:,.0f}")
    print(f"  Implied mean   : EUR {mean:,.0f}")
    print(f"  Implied P10    : EUR {p10:,.0f}")
    print(f"  Implied P90    : EUR {p90:,.0f}")
    print()

    if seasonality:
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        print("-- Monthly seasonality indices --")
        for name, idx in zip(months, seasonality):
            bar = "#" * int(idx * 20)
            print(f"  {name}: {idx:.3f}  {bar}")
        print(f"  Sum: {sum(seasonality):.2f} (should be ~12.0)")
        print()

    print("=" * 60)
    print("PASTE INTO seed_synthetic_data.py:")
    print("=" * 60)
    print()
    print("# Calibrated from ANAC open data (CC BY 4.0 / IODL 2.0)")
    print("# Run scripts/analyze_anac.py to regenerate these parameters.")
    print(f"ANAC_LOGNORMAL_MU    = {mu:.4f}   # log-mean of contract amounts (EUR)")
    print(f"ANAC_LOGNORMAL_SIGMA = {sigma:.4f}   # log-std of contract amounts (EUR)")
    if seasonality:
        print(f"ANAC_MONTHLY_SEASONALITY = {seasonality}")
    print()


def main() -> None:
    args = parse_args()

    if not Path(args.file).exists():
        print(f"Error: file not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    df = load_csv(args.file, args.sep)

    if args.amount_col not in df.columns:
        print(f"Error: column '{args.amount_col}' not found.", file=sys.stderr)
        print(f"Available columns: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    if args.sample > 0 and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=42)
        print(f"Using random sample of {args.sample:,} rows.")

    # Clean and filter amounts
    amounts_raw = pd.to_numeric(df[args.amount_col], errors="coerce")
    amounts = amounts_raw.dropna()
    amounts = amounts[(amounts >= args.min_amount) & (amounts <= args.max_amount)]
    amounts = amounts.values.astype(float)

    if len(amounts) < 100:
        print(f"Warning: only {len(amounts)} valid amounts after filtering. Results may not be reliable.")

    mu, sigma = fit_lognormal(amounts)

    seasonality = None
    if args.date_col in df.columns:
        try:
            seasonality = compute_monthly_seasonality(df, args.date_col, args.amount_col)
        except Exception as e:
            print(f"Warning: could not compute seasonality from '{args.date_col}': {e}")

    print_report(amounts, mu, sigma, seasonality, args.file)


if __name__ == "__main__":
    main()
