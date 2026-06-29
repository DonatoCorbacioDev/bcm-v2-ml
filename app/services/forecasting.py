"""
Financial forecasting using Facebook Prophet.

Prophet handles trend + yearly seasonality automatically and produces
calibrated 95% confidence intervals. For series shorter than 2 data points
a simple flat fallback is used since Prophet requires at least 2 observations.
"""

from __future__ import annotations

import logging

import pandas as pd
from prophet import Prophet
from sqlalchemy.orm import Session

from ..models import FinancialValue

# Prophet / Stan emit verbose INFO logs during fitting — suppress them.
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


def _flat_forecast(level: float, h: int) -> list[dict]:
    """Return a flat forecast with ±10% CI when not enough data for Prophet."""
    ci = level * 0.1
    return [
        {
            "amount": round(level, 2),
            "lower": round(max(0.0, level - ci), 2),
            "upper": round(level + ci, 2),
        }
        for _ in range(h)
    ]


def _prophet_forecast(df: pd.DataFrame, h: int) -> list[dict]:
    """Fit Prophet on a ds/y DataFrame and return h future monthly periods."""
    n = len(df)
    m = Prophet(
        yearly_seasonality=n >= 24,
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.95,
        seasonality_mode="additive",
    )
    m.fit(df)

    future = m.make_future_dataframe(periods=h, freq="MS")
    forecast = m.predict(future)

    results = []
    for _, row in forecast.tail(h).iterrows():
        amount = max(0.0, float(row["yhat"]))
        lower = max(0.0, float(row["yhat_lower"]))
        upper = float(row["yhat_upper"])
        results.append({
            "month": row["ds"].strftime("%Y-%m"),
            "amount": round(amount, 2),
            "lower": round(lower, 2),
            "upper": round(upper, 2),
        })

    return results


def compute_forecast(db: Session, months: int, org_id: int | None = None) -> dict:
    """Return historical monthly totals and a Prophet forecast for N months ahead."""
    query = db.query(FinancialValue)
    if org_id is not None:
        query = query.filter(FinancialValue.organization_id == org_id)
    rows = query.all()

    if not rows:
        return {"historical": [], "forecast": []}

    df = pd.DataFrame(
        [{"year": r.year_value, "month": r.month_value, "amount": r.financial_amount}
         for r in rows]
    )
    monthly = (
        df.groupby(["year", "month"])["amount"]
        .sum()
        .reset_index()
        .sort_values(["year", "month"])
        .reset_index(drop=True)
    )
    monthly["date"] = pd.to_datetime(
        {"year": monthly["year"], "month": monthly["month"], "day": 1}
    )

    historical = [
        {"month": row["date"].strftime("%Y-%m"), "amount": round(float(row["amount"]), 2)}
        for _, row in monthly.iterrows()
    ]

    safe_months = min(max(months, 1), 24)
    last_date = monthly["date"].iloc[-1]

    if len(monthly) < 2:
        flat = _flat_forecast(float(monthly["amount"].iloc[-1]), safe_months)
        forecast = [
            {"month": (last_date + pd.DateOffset(months=i)).strftime("%Y-%m"), **f}
            for i, f in enumerate(flat, start=1)
        ]
        return {"historical": historical, "forecast": forecast}

    prophet_df = monthly.rename(columns={"date": "ds", "amount": "y"})[["ds", "y"]]
    try:
        forecast = _prophet_forecast(prophet_df, safe_months)
    except Exception as exc:
        logger.warning("Prophet failed (%s), falling back to flat forecast", exc)
        flat = _flat_forecast(float(monthly["amount"].iloc[-1]), safe_months)
        forecast = [
            {"month": (last_date + pd.DateOffset(months=i)).strftime("%Y-%m"), **f}
            for i, f in enumerate(flat, start=1)
        ]

    return {"historical": historical, "forecast": forecast}
