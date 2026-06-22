import httpx
from sqlalchemy.orm import Session

from ..config import settings
from . import forecasting, risk_scoring

TOP_RISK_CONTRACTS = 5


def _format_risk_scores(risk_scores: list) -> str:
    if not risk_scores:
        return "No contracts found."

    lines = []
    for item in risk_scores[:TOP_RISK_CONTRACTS]:
        anomalies = ", ".join(item["anomalies"]) if item["anomalies"] else "none"
        lines.append(
            f"- {item['customerName']} (contract #{item['contractId']}): "
            f"risk score {item['riskScore']}, level {item['level']}, anomalies: {anomalies}"
        )
    return "\n".join(lines)


def _format_forecast(forecast: dict) -> str:
    historical = forecast.get("historical", [])
    forecast_months = forecast.get("forecast", [])

    if not historical and not forecast_months:
        return "No financial data available."

    lines = []
    if historical:
        recent = historical[-3:]
        recent_str = ", ".join(f"{h['month']}: {h['amount']}" for h in recent)
        lines.append(f"Recent months: {recent_str}")

    if forecast_months:
        forecast_str = ", ".join(
            f"{f['month']}: {f['amount']} (range {f['lower']}-{f['upper']})"
            for f in forecast_months
        )
        lines.append(f"Forecast: {forecast_str}")

    return "\n".join(lines)


def _build_prompt(risk_scores: list, forecast: dict, language: str) -> str:
    return (
        f"You are a financial assistant for a contract management system. "
        f"Write your entire response in {language}.\n\n"
        f"Here are the contracts with the highest risk scores:\n"
        f"{_format_risk_scores(risk_scores)}\n\n"
        f"Here is the financial trend (historical and forecast amounts):\n"
        f"{_format_forecast(forecast)}\n\n"
        f"Write a concise report with three sections: "
        f"1) the highest-risk contracts and why, "
        f"2) the financial trend for the upcoming months, "
        f"3) recommended actions."
    )


def _call_ollama(prompt: str) -> str:
    response = httpx.post(
        f"{settings.OLLAMA_URL}/api/generate",
        json={"model": settings.OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=settings.OLLAMA_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()["response"]


def generate_insights(db: Session, months: int, org_id: int | None = None) -> dict:
    risk_scores = risk_scoring.compute_risk_scores(db, org_id)
    forecast = forecasting.compute_forecast(db, months, org_id)

    report = None
    error = None
    try:
        prompt = _build_prompt(risk_scores, forecast, settings.REPORT_LANGUAGE)
        report = _call_ollama(prompt)
    except httpx.HTTPError as exc:
        error = f"Ollama service unavailable: {exc}"

    return {
        "riskScores": risk_scores,
        "forecast": forecast,
        "report": report,
        "error": error,
    }
