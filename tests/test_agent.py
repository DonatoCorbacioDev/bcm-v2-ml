from unittest.mock import MagicMock, patch

import httpx

from app.services.agent import (
    _build_prompt, _call_ollama, _format_forecast, _format_risk_scores, generate_insights,
)


# ── _format_risk_scores ──────────────────────────────────────────────────────

def test_format_risk_scores_empty():
    assert _format_risk_scores([]) == "No contracts found."


def test_format_risk_scores_with_anomalies():
    risk_scores = [
        {"contractId": 1, "customerName": "Acme", "riskScore": 0.9, "level": "HIGH", "anomalies": ["EXPIRED"]},
    ]
    result = _format_risk_scores(risk_scores)
    assert "Acme" in result
    assert "HIGH" in result
    assert "EXPIRED" in result


def test_format_risk_scores_without_anomalies():
    risk_scores = [
        {"contractId": 2, "customerName": "Beta", "riskScore": 0.1, "level": "LOW", "anomalies": []},
    ]
    result = _format_risk_scores(risk_scores)
    assert "none" in result


def test_format_risk_scores_limits_to_top_n():
    risk_scores = [
        {"contractId": i, "customerName": f"Org{i}", "riskScore": 1.0, "level": "HIGH", "anomalies": []}
        for i in range(10)
    ]
    result = _format_risk_scores(risk_scores)
    assert result.count("Org") == 5


# ── _format_forecast ──────────────────────────────────────────────────────────

def test_format_forecast_empty():
    assert _format_forecast({"historical": [], "forecast": []}) == "No financial data available."


def test_format_forecast_with_data():
    forecast = {
        "historical": [{"month": "2024-01", "amount": 1000.0}],
        "forecast": [{"month": "2024-02", "amount": 1100.0, "lower": 900.0, "upper": 1300.0}],
    }
    result = _format_forecast(forecast)
    assert "2024-01" in result
    assert "2024-02" in result
    assert "900.0-1300.0" in result


# ── _build_prompt ──────────────────────────────────────────────────────────────

def test_build_prompt_includes_language_and_data():
    risk_scores = [{"contractId": 1, "customerName": "Acme", "riskScore": 0.9, "level": "HIGH", "anomalies": []}]
    forecast = {"historical": [], "forecast": []}
    prompt = _build_prompt(risk_scores, forecast, "italian")
    assert "italian" in prompt
    assert "Acme" in prompt
    assert "No financial data available." in prompt


# ── _call_ollama ────────────────────────────────────────────────────────────────

def test_call_ollama_success():
    mock_response = MagicMock()
    mock_response.json.return_value = {"response": "Generated report"}
    with patch("app.services.agent.httpx.post", return_value=mock_response) as mock_post:
        result = _call_ollama("some prompt")
    assert result == "Generated report"
    mock_post.assert_called_once()


def test_call_ollama_connection_error():
    with patch("app.services.agent.httpx.post", side_effect=httpx.ConnectError("refused")):
        try:
            _call_ollama("some prompt")
            assert False, "expected httpx.ConnectError"
        except httpx.ConnectError:
            pass


def test_call_ollama_http_status_error():
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "error", request=MagicMock(), response=mock_response
    )
    with patch("app.services.agent.httpx.post", return_value=mock_response):
        try:
            _call_ollama("some prompt")
            assert False, "expected httpx.HTTPStatusError"
        except httpx.HTTPStatusError:
            pass


# ── generate_insights ────────────────────────────────────────────────────────

def test_generate_insights_success():
    db = MagicMock()
    risk_scores = [{"contractId": 1, "customerName": "Acme", "riskScore": 0.9, "level": "HIGH", "anomalies": []}]
    forecast = {"historical": [], "forecast": []}
    with patch("app.services.agent.risk_scoring.compute_risk_scores", return_value=risk_scores), \
         patch("app.services.agent.forecasting.compute_forecast", return_value=forecast), \
         patch("app.services.agent._call_ollama", return_value="Generated report"):
        result = generate_insights(db, 3)
    assert result["riskScores"] == risk_scores
    assert result["forecast"] == forecast
    assert result["report"] == "Generated report"
    assert result["error"] is None


def test_generate_insights_ollama_unavailable():
    db = MagicMock()
    with patch("app.services.agent.risk_scoring.compute_risk_scores", return_value=[]), \
         patch("app.services.agent.forecasting.compute_forecast", return_value={"historical": [], "forecast": []}), \
         patch("app.services.agent._call_ollama", side_effect=httpx.ConnectError("refused")):
        result = generate_insights(db, 3)
    assert result["report"] is None
    assert "Ollama service unavailable" in result["error"]
