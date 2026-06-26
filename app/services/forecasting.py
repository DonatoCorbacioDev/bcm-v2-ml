import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from ..models import FinancialValue


def _holt_dampened(
    y: np.ndarray,
    h: int,
    alpha: float = 0.3,
    beta: float = 0.1,
    phi: float = 0.85,
) -> tuple[list[float], float]:
    """
    Holt's dampened-trend exponential smoothing.

    Dampening (phi < 1) prevents the trend from exploding on short or noisy
    series — the critical difference from plain linear extrapolation.
    Returns (point forecasts for h steps, residual std for CI computation).
    """
    n = len(y)
    if n == 0:
        return [], 0.0
    if n == 1:
        # No trend information: flat forecast, CI = 10% of level
        return [float(y[0])] * h, float(y[0]) * 0.1

    l = float(y[0])
    b = float(y[1] - y[0])
    residuals: list[float] = []

    for t in range(1, n):
        fitted = l + phi * b
        residuals.append(float(y[t]) - fitted)
        l_new = alpha * float(y[t]) + (1 - alpha) * fitted
        b = beta * (l_new - l) + (1 - beta) * phi * b
        l = l_new

    std_resid = (
        float(np.std(residuals, ddof=1)) if len(residuals) > 1
        else abs(residuals[0]) if residuals
        else 0.0
    )

    forecasts: list[float] = []
    phi_cumsum = 0.0
    for _ in range(h):
        phi_cumsum = phi_cumsum * phi + phi   # phi + phi^2 + ... + phi^i
        forecasts.append(l + phi_cumsum * b)

    return forecasts, std_resid


def compute_forecast(db: Session, months: int, org_id: int | None = None) -> dict:
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

    y = monthly["amount"].values.astype(float)
    safe_months = min(max(months, 1), 24)

    point_forecasts, std_resid = _holt_dampened(y, h=safe_months)
    ci = 1.96 * std_resid

    historical = [
        {"month": row["date"].strftime("%Y-%m"), "amount": round(float(row["amount"]), 2)}
        for _, row in monthly.iterrows()
    ]

    last_date = monthly["date"].iloc[-1]
    forecast = []
    for i, amount in enumerate(point_forecasts, start=1):
        clamped = max(0.0, amount)
        next_month = (last_date + pd.DateOffset(months=i)).strftime("%Y-%m")
        forecast.append({
            "month": next_month,
            "amount": round(clamped, 2),
            "lower": round(max(0.0, amount - ci), 2),
            "upper": round(amount + ci, 2),
        })

    return {"historical": historical, "forecast": forecast}
