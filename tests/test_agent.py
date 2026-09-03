from unittest.mock import MagicMock, patch

import httpx

from app.services.agent import (
    _build_prompt, _call_ollama, _call_ollama_chat, _format_forecast, _format_risk_scores,
    _run_tool, ask_agent, generate_insights,
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
    assert "Servizio AI non disponibile" in result["error"]


def test_generate_insights_passes_org_id_through():
    db = MagicMock()
    with patch("app.services.agent.risk_scoring.compute_risk_scores", return_value=[]) as mock_risk, \
         patch("app.services.agent.forecasting.compute_forecast", return_value={"historical": [], "forecast": []}) as mock_forecast, \
         patch("app.services.agent._call_ollama", return_value="Generated report"):
        generate_insights(db, 3, org_id=9)
    mock_risk.assert_called_once_with(db, 9)
    mock_forecast.assert_called_once_with(db, 3, 9)


# ── _run_tool ─────────────────────────────────────────────────────────────────

def test_run_tool_get_risk_scores_uses_bound_org_id():
    db = MagicMock()
    with patch("app.services.agent.risk_scoring.compute_risk_scores", return_value=["scores"]) as mock_risk:
        result = _run_tool("get_risk_scores", {}, db, org_id=7)
    mock_risk.assert_called_once_with(db, 7)
    assert result == ["scores"]


def test_run_tool_get_forecast_uses_bound_org_id_and_model_supplied_months():
    db = MagicMock()
    with patch("app.services.agent.forecasting.compute_forecast", return_value={"historical": []}) as mock_forecast:
        result = _run_tool("get_forecast", {"months": 6}, db, org_id=7)
    mock_forecast.assert_called_once_with(db, 6, 7)
    assert result == {"historical": []}


def test_run_tool_get_forecast_clamps_out_of_range_months():
    db = MagicMock()
    with patch("app.services.agent.forecasting.compute_forecast", return_value={}) as mock_forecast:
        _run_tool("get_forecast", {"months": 999}, db, org_id=1)
    mock_forecast.assert_called_once_with(db, 24, 1)


def test_run_tool_get_forecast_defaults_months_when_missing():
    db = MagicMock()
    with patch("app.services.agent.forecasting.compute_forecast", return_value={}) as mock_forecast:
        _run_tool("get_forecast", {}, db, org_id=1)
    mock_forecast.assert_called_once_with(db, 3, 1)


def test_run_tool_ignores_org_id_supplied_by_the_model():
    """A model-supplied org_id in tool arguments must never override the
    server-bound one — this is the tenant-isolation boundary for tool calls."""
    db = MagicMock()
    with patch("app.services.agent.risk_scoring.compute_risk_scores", return_value=[]) as mock_risk:
        _run_tool("get_risk_scores", {"org_id": 999}, db, org_id=7)
    mock_risk.assert_called_once_with(db, 7)


def test_run_tool_unknown_tool_returns_error_dict():
    result = _run_tool("delete_everything", {}, MagicMock(), org_id=1)
    assert "error" in result


# ── _call_ollama_chat ────────────────────────────────────────────────────────

def test_call_ollama_chat_includes_tools_when_provided():
    mock_response = MagicMock()
    mock_response.json.return_value = {"message": {"role": "assistant", "content": "hi"}}
    with patch("app.services.agent.httpx.post", return_value=mock_response) as mock_post:
        result = _call_ollama_chat([{"role": "user", "content": "hi"}], tools=[{"type": "function"}])
    assert result == {"role": "assistant", "content": "hi"}
    assert "tools" in mock_post.call_args.kwargs["json"]


def test_call_ollama_chat_omits_tools_when_none():
    mock_response = MagicMock()
    mock_response.json.return_value = {"message": {"role": "assistant", "content": "hi"}}
    with patch("app.services.agent.httpx.post", return_value=mock_response) as mock_post:
        _call_ollama_chat([{"role": "user", "content": "hi"}], tools=None)
    assert "tools" not in mock_post.call_args.kwargs["json"]


# ── ask_agent ─────────────────────────────────────────────────────────────────

def test_ask_agent_answers_directly_without_tool_calls():
    db = MagicMock()
    with patch("app.services.agent._call_ollama_chat",
               return_value={"role": "assistant", "content": "42 contracts total."}):
        result = ask_agent(db, "How many contracts?", org_id=1)
    assert result == {"answer": "42 contracts total.", "error": None}


def test_ask_agent_executes_a_requested_tool_then_answers():
    db = MagicMock()
    tool_request = {
        "role": "assistant", "content": "",
        "tool_calls": [{"function": {"name": "get_risk_scores", "arguments": {}}}],
    }
    final_answer = {"role": "assistant", "content": "Acme is your riskiest contract."}
    with patch("app.services.agent._call_ollama_chat", side_effect=[tool_request, final_answer]), \
         patch("app.services.agent.risk_scoring.compute_risk_scores",
               return_value=[{"customerName": "Acme"}]) as mock_risk:
        result = ask_agent(db, "Which contract is riskiest?", org_id=5)
    mock_risk.assert_called_once_with(db, 5)
    assert result == {"answer": "Acme is your riskiest contract.", "error": None}


def test_ask_agent_forces_a_final_answer_after_max_iterations():
    db = MagicMock()
    always_wants_tool = {
        "role": "assistant", "content": "",
        "tool_calls": [{"function": {"name": "get_risk_scores", "arguments": {}}}],
    }
    forced_final = {"role": "assistant", "content": "Best I can say without more tool calls."}
    with patch("app.services.agent._call_ollama_chat",
               side_effect=[always_wants_tool, always_wants_tool, always_wants_tool, forced_final]), \
         patch("app.services.agent.risk_scoring.compute_risk_scores", return_value=[]):
        result = ask_agent(db, "Loop forever?", org_id=1)
    assert result == {"answer": "Best I can say without more tool calls.", "error": None}


def test_ask_agent_returns_error_when_ollama_unavailable():
    db = MagicMock()
    with patch("app.services.agent._call_ollama_chat", side_effect=httpx.ConnectError("refused")):
        result = ask_agent(db, "Any question", org_id=1)
    assert result["answer"] is None
    assert "Servizio AI non disponibile" in result["error"]
