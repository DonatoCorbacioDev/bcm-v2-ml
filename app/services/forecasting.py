import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from ..models import FinancialValue


def compute_forecast(db: Session, months: int, org_id: int | None = None) -> dict:
    query = db.query(FinancialValue)
    if org_id is not None:
        query = query.filter(FinancialValue.organization_id == org_id)
    rows = query.all()

    if not rows:
        return {"historical": [], "forecast": []}

    df = pd.DataFrame(
        [{"year": r.year_value, "month": r.month_value, "amount": r.financial_amount} for r in rows]
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

    t = np.arange(len(monthly), dtype=float)
    y = monthly["amount"].values.astype(float)

    if len(t) >= 2:
        A = np.column_stack([t, np.ones(len(t))])
        coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
        slope, intercept = coeffs
        std_resid = np.std(y - (slope * t + intercept), ddof=1) if len(t) > 2 else 0.0
    else:
        slope, intercept = 0.0, float(y[0]) if len(y) else 0.0
        std_resid = 0.0

    historical = [
        {"month": row["date"].strftime("%Y-%m"), "amount": round(float(row["amount"]), 2)}
        for _, row in monthly.iterrows()
    ]

    last_date = monthly["date"].iloc[-1]
    last_t = float(len(t) - 1)
    ci = 1.96 * std_resid
    safe_months = min(max(months, 1), 24)
    forecast = []
    for i in range(1, safe_months + 1):
        amount = slope * (last_t + i) + intercept
        next_month = (last_date + pd.DateOffset(months=i)).strftime("%Y-%m")
        forecast.append({
            "month": next_month,
            "amount": round(max(0.0, amount), 2),
            "lower": round(max(0.0, amount - ci), 2),
            "upper": round(amount + ci, 2),
        })

    return {"historical": historical, "forecast": forecast}
