import itertools
from unittest.mock import MagicMock, patch

import httpx

from app.services.agent import (
    _build_prompt, _call_ollama, _call_ollama_chat, _format_amount, _format_forecast,
    _format_risk_scores, _looks_like_leaked_tool_call, _resolve_placeholders, _run_tool,
    _templatize_forecast, _templatize_risk_scores, _templatize_tool_result, ask_agent,
    generate_insights,
)


# ── _format_risk_scores ──────────────────────────────────────────────────────

def test_format_risk_scores_empty():
    assert _format_risk_scores([], {}) == "No contracts found."


def test_format_risk_scores_with_anomalies():
    risk_scores = [
        {"contractId": 1, "customerName": "Acme", "riskScore": 0.9, "level": "HIGH", "anomalies": ["EXPIRED"]},
    ]
    placeholders = {}
    result = _format_risk_scores(risk_scores, placeholders)
    assert "Acme" in result
    assert "HIGH" in result
    assert "EXPIRED" in result
    assert placeholders["[R1]"] == "0.9"


def test_format_risk_scores_score_is_a_token_not_the_raw_number():
    # The whole point of the templating refactor: the raw decimal must never
    # reach the model as text, only an opaque token - see _resolve_placeholders.
    risk_scores = [{"contractId": 1, "customerName": "Acme", "riskScore": 0.91, "level": "HIGH", "anomalies": []}]
    result = _format_risk_scores(risk_scores, {})
    assert "0.91" not in result
    assert "[R1]" in result


def test_format_risk_scores_without_anomalies():
    risk_scores = [
        {"contractId": 2, "customerName": "Beta", "riskScore": 0.1, "level": "LOW", "anomalies": []},
    ]
    result = _format_risk_scores(risk_scores, {})
    assert "none" in result


def test_format_risk_scores_limits_to_top_n():
    risk_scores = [
        {"contractId": i, "customerName": f"Org{i}", "riskScore": 1.0, "level": "HIGH", "anomalies": []}
        for i in range(10)
    ]
    placeholders = {}
    result = _format_risk_scores(risk_scores, placeholders)
    assert result.count("Org") == 5
    assert len(placeholders) == 5  # one score token per shown contract, not per all 10


def test_format_risk_scores_notes_remaining_high_risk_contracts():
    # 7 HIGH: 5 are shown by name, the other 2 must still be surfaced -
    # silently dropping a HIGH contract from the report is not acceptable
    # just because it didn't make the top 5 by score.
    risk_scores = [
        {"contractId": i, "customerName": f"Org{i}", "riskScore": 1.0 - i * 0.01, "level": "HIGH", "anomalies": []}
        for i in range(7)
    ]
    result = _format_risk_scores(risk_scores, {})
    assert "2 more HIGH-risk contract(s) not shown above" in result


def test_format_risk_scores_no_note_when_overflow_is_not_high():
    # 5 HIGH shown + 2 LOW cut off: omitting a LOW contract is fine (see
    # scripts/eval_agent_insights.py's name_coverage for the same rule), so
    # no "more HIGH-risk" note should appear.
    risk_scores = [
        {"contractId": i, "customerName": f"Org{i}", "riskScore": 1.0 - i * 0.01, "level": "HIGH", "anomalies": []}
        for i in range(5)
    ] + [
        {"contractId": 50 + i, "customerName": f"Low{i}", "riskScore": 0.1, "level": "LOW", "anomalies": []}
        for i in range(2)
    ]
    result = _format_risk_scores(risk_scores, {})
    assert "more HIGH-risk" not in result


# ── _format_amount ────────────────────────────────────────────────────────────

def test_format_amount_rounds_and_uses_thousands_separator():
    assert _format_amount(45230.567891) == "45.231"


def test_format_amount_small_value_no_separator():
    assert _format_amount(500.0) == "500"


# ── _format_forecast ──────────────────────────────────────────────────────────

def test_format_forecast_empty():
    assert _format_forecast({"historical": [], "forecast": []}, {}) == "No financial data available."


def test_format_forecast_with_data():
    forecast = {
        "historical": [{"month": "2024-01", "amount": 1000.0}],
        "forecast": [{"month": "2024-02", "amount": 1100.0, "lower": 900.0, "upper": 1300.0}],
    }
    placeholders = {}
    result = _format_forecast(forecast, placeholders)
    assert "2024-01" in result
    assert "2024-02" in result
    assert "1.000" not in result  # raw amount must never leak into the text itself
    assert placeholders["[H1]"] == "1.000"
    assert placeholders["[F1]"] == "1.100"
    assert placeholders["[F1L]"] == "900"
    assert placeholders["[F1U]"] == "1.300"


def test_format_forecast_cleans_up_long_raw_floats():
    # The actual bug being fixed originally: a long raw float is exactly
    # what a small local LLM garbles most often - the placeholder's
    # resolved VALUE must still be short and clean (the model itself never
    # sees the float at all now, but the final report's figure does).
    forecast = {"historical": [{"month": "2024-01", "amount": 45230.567891234}], "forecast": []}
    placeholders = {}
    _format_forecast(forecast, placeholders)
    assert placeholders["[H1]"] == "45.231"


# ── _build_prompt ──────────────────────────────────────────────────────────────

def test_build_prompt_includes_language_and_data():
    prompt = _build_prompt("- Acme (contract #1): risk score [R1]", "No financial data available.", "italian")
    assert "italian" in prompt
    assert "Acme" in prompt
    assert "No financial data available." in prompt


# ── _resolve_placeholders ────────────────────────────────────────────────────

def test_resolve_placeholders_substitutes_every_token():
    report = "Il contratto Acme ha punteggio [R1], il fatturato di giugno è [H1]."
    result = _resolve_placeholders(report, {"[R1]": "0.91", "[H1]": "48.230"})
    assert result == "Il contratto Acme ha punteggio 0.91, il fatturato di giugno è 48.230."


def test_resolve_placeholders_leaves_unknown_bracket_text_untouched():
    # Not every "[...]" in a report is one of ours - resolution must not
    # touch text it doesn't recognize.
    report = "Vedi [nota 1] per i dettagli, punteggio [R1]."
    result = _resolve_placeholders(report, {"[R1]": "0.91"})
    assert result == "Vedi [nota 1] per i dettagli, punteggio 0.91."


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


def test_generate_insights_handles_malformed_ollama_json():
    """Regression test for A4: a 2xx response body that isn't valid JSON must
    degrade gracefully, not crash the request with an unhandled ValueError."""
    db = MagicMock()
    with patch("app.services.agent.risk_scoring.compute_risk_scores", return_value=[]), \
         patch("app.services.agent.forecasting.compute_forecast", return_value={"historical": [], "forecast": []}), \
         patch("app.services.agent._call_ollama", side_effect=ValueError("Expecting value: line 1 column 1")):
        result = generate_insights(db, 3)
    assert result["report"] is None
    assert "Servizio AI non disponibile" in result["error"]


def test_generate_insights_handles_unexpected_ollama_response_shape():
    """Regression test for A4: valid JSON missing the expected key must
    degrade gracefully, not crash the request with an unhandled KeyError."""
    db = MagicMock()
    with patch("app.services.agent.risk_scoring.compute_risk_scores", return_value=[]), \
         patch("app.services.agent.forecasting.compute_forecast", return_value={"historical": [], "forecast": []}), \
         patch("app.services.agent._call_ollama", side_effect=KeyError("response")):
        result = generate_insights(db, 3)
    assert result["report"] is None
    assert "Servizio AI non disponibile" in result["error"]


def test_generate_insights_passes_org_id_through():
    db = MagicMock()
    with patch("app.services.agent.risk_scoring.compute_risk_scores", return_value=[]) as mock_risk, \
         patch("app.services.agent.forecasting.compute_forecast", return_value={"historical": [], "forecast": []}) as mock_forecast, \
         patch("app.services.agent._call_ollama", return_value="Generated report"):
        generate_insights(db, 3, org_id=9)
    mock_risk.assert_called_once_with(db, 9, None)
    mock_forecast.assert_called_once_with(db, 3, 9, None)


def test_generate_insights_passes_manager_id_through():
    db = MagicMock()
    with patch("app.services.agent.risk_scoring.compute_risk_scores", return_value=[]) as mock_risk, \
         patch("app.services.agent.forecasting.compute_forecast", return_value={"historical": [], "forecast": []}) as mock_forecast, \
         patch("app.services.agent._call_ollama", return_value="Generated report"):
        generate_insights(db, 3, org_id=9, manager_id=42)
    mock_risk.assert_called_once_with(db, 9, 42)
    mock_forecast.assert_called_once_with(db, 3, 9, 42)


# ── _run_tool ─────────────────────────────────────────────────────────────────

def test_run_tool_get_risk_scores_uses_bound_org_id():
    db = MagicMock()
    with patch("app.services.agent.risk_scoring.compute_risk_scores", return_value=["scores"]) as mock_risk:
        result = _run_tool("get_risk_scores", {}, db, org_id=7)
    mock_risk.assert_called_once_with(db, 7, None)
    assert result == ["scores"]


def test_run_tool_get_risk_scores_uses_bound_manager_id():
    db = MagicMock()
    with patch("app.services.agent.risk_scoring.compute_risk_scores", return_value=["scores"]) as mock_risk:
        result = _run_tool("get_risk_scores", {}, db, org_id=7, manager_id=42)
    mock_risk.assert_called_once_with(db, 7, 42)
    assert result == ["scores"]


def test_run_tool_get_forecast_uses_bound_org_id_and_model_supplied_months():
    db = MagicMock()
    with patch("app.services.agent.forecasting.compute_forecast", return_value={"historical": []}) as mock_forecast:
        result = _run_tool("get_forecast", {"months": 6}, db, org_id=7)
    mock_forecast.assert_called_once_with(db, 6, 7, None)
    assert result == {"historical": []}


def test_run_tool_get_forecast_uses_bound_manager_id():
    db = MagicMock()
    with patch("app.services.agent.forecasting.compute_forecast", return_value={}) as mock_forecast:
        _run_tool("get_forecast", {"months": 6}, db, org_id=7, manager_id=42)
    mock_forecast.assert_called_once_with(db, 6, 7, 42)


def test_run_tool_get_forecast_clamps_out_of_range_months():
    db = MagicMock()
    with patch("app.services.agent.forecasting.compute_forecast", return_value={}) as mock_forecast:
        _run_tool("get_forecast", {"months": 999}, db, org_id=1)
    mock_forecast.assert_called_once_with(db, 24, 1, None)


def test_run_tool_get_forecast_defaults_months_when_missing():
    db = MagicMock()
    with patch("app.services.agent.forecasting.compute_forecast", return_value={}) as mock_forecast:
        _run_tool("get_forecast", {}, db, org_id=1)
    mock_forecast.assert_called_once_with(db, 3, 1, None)


def test_run_tool_ignores_org_id_supplied_by_the_model():
    """A model-supplied org_id in tool arguments must never override the
    server-bound one — this is the tenant-isolation boundary for tool calls."""
    db = MagicMock()
    with patch("app.services.agent.risk_scoring.compute_risk_scores", return_value=[]) as mock_risk:
        _run_tool("get_risk_scores", {"org_id": 999}, db, org_id=7)
    mock_risk.assert_called_once_with(db, 7, None)


def test_run_tool_ignores_manager_id_supplied_by_the_model():
    """Same tenant-isolation boundary as org_id above, for manager_id."""
    db = MagicMock()
    with patch("app.services.agent.risk_scoring.compute_risk_scores", return_value=[]) as mock_risk:
        _run_tool("get_risk_scores", {"manager_id": 999}, db, org_id=7, manager_id=42)
    mock_risk.assert_called_once_with(db, 7, 42)


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


# ── _templatize_risk_scores / _templatize_forecast / _templatize_tool_result ──

def test_templatize_risk_scores_replaces_score_with_a_token():
    risk_scores = [{"customerName": "Acme", "contractId": 1, "riskScore": 0.91, "level": "HIGH", "anomalies": []}]
    placeholders = {}
    counter = itertools.count(1)
    result = _templatize_risk_scores(risk_scores, placeholders, counter)
    assert result == [{"customerName": "Acme", "contractId": 1, "riskScore": "[T1]", "level": "HIGH", "anomalies": []}]
    assert placeholders == {"[T1]": "0.91"}


def test_templatize_risk_scores_gives_each_contract_its_own_token():
    risk_scores = [
        {"customerName": "A", "contractId": 1, "riskScore": 0.9, "level": "HIGH", "anomalies": []},
        {"customerName": "B", "contractId": 2, "riskScore": 0.3, "level": "LOW", "anomalies": []},
    ]
    placeholders = {}
    result = _templatize_risk_scores(risk_scores, placeholders, itertools.count(1))
    assert result[0]["riskScore"] == "[T1]"
    assert result[1]["riskScore"] == "[T2]"
    assert placeholders == {"[T1]": "0.9", "[T2]": "0.3"}


def test_templatize_forecast_replaces_amounts_with_tokens():
    forecast = {
        "historical": [{"month": "2026-01", "amount": 1000.0}],
        "forecast": [{"month": "2026-02", "amount": 1100.0, "lower": 900.0, "upper": 1300.0}],
    }
    placeholders = {}
    result = _templatize_forecast(forecast, placeholders, itertools.count(1))
    assert result["historical"] == [{"month": "2026-01", "amount": "[T1]"}]
    assert result["forecast"] == [{"month": "2026-02", "amount": "[T2]", "lower": "[T3]", "upper": "[T4]"}]
    assert placeholders == {"[T1]": "1.000", "[T2]": "1.100", "[T3]": "900", "[T4]": "1.300"}


def test_templatize_forecast_and_risk_scores_share_one_counter_without_collision():
    # ask_agent shares one counter across every tool call in a conversation -
    # a risk-score token and a forecast token must never end up with the
    # same number just because each templatizer started counting from 1.
    placeholders = {}
    counter = itertools.count(1)
    risk_result = _templatize_risk_scores(
        [{"customerName": "Acme", "contractId": 1, "riskScore": 0.9, "level": "HIGH", "anomalies": []}],
        placeholders, counter,
    )
    forecast_result = _templatize_forecast(
        {"historical": [{"month": "2026-01", "amount": 500.0}], "forecast": []}, placeholders, counter,
    )
    assert risk_result[0]["riskScore"] == "[T1]"
    assert forecast_result["historical"][0]["amount"] == "[T2]"


def test_templatize_tool_result_dispatches_by_name():
    placeholders = {}
    counter = itertools.count(1)
    risk_scores = [{"customerName": "Acme", "contractId": 1, "riskScore": 0.9, "level": "HIGH", "anomalies": []}]
    result = _templatize_tool_result("get_risk_scores", risk_scores, placeholders, counter)
    assert result[0]["riskScore"] == "[T1]"


def test_templatize_tool_result_passes_through_unknown_tools_unchanged():
    placeholders = {}
    reminder_result = {"proposedAction": {"type": "CREATE_REMINDER", "contractId": 1, "message": "hi"}}
    result = _templatize_tool_result("propose_reminder", reminder_result, placeholders, itertools.count(1))
    assert result == reminder_result
    assert placeholders == {}


# ── _looks_like_leaked_tool_call ────────────────────────────────────────────────

def test_looks_like_leaked_tool_call_detects_the_real_observed_shape():
    # The exact text a live run produced (scripts/eval_ask_agent.py, 2026-10-19).
    assert _looks_like_leaked_tool_call('{"name":"get_risk_scores","parameters":{"}}') is True


def test_looks_like_leaked_tool_call_allows_leading_whitespace():
    assert _looks_like_leaked_tool_call('  \n{"name": "get_forecast", "parameters": {}}') is True


def test_looks_like_leaked_tool_call_does_not_flag_a_normal_answer():
    assert _looks_like_leaked_tool_call("Il contratto Acme ha un punteggio di 0.91.") is False


def test_looks_like_leaked_tool_call_does_not_flag_json_mentioned_in_prose():
    # Narrow on purpose: only Ollama's own tool-call JSON shape, not any
    # text that happens to contain braces or the word "name".
    assert _looks_like_leaked_tool_call('Il campo si chiama "name" nel contratto.') is False


# ── ask_agent ─────────────────────────────────────────────────────────────────

def test_ask_agent_answers_directly_without_tool_calls():
    db = MagicMock()
    with patch("app.services.agent._call_ollama_chat",
               return_value={"role": "assistant", "content": "42 contracts total."}):
        result = ask_agent(db, "How many contracts?", org_id=1)
    assert result == {"answer": "42 contracts total.", "error": None, "proposedAction": None}


def test_ask_agent_executes_a_requested_tool_then_answers():
    db = MagicMock()
    tool_request = {
        "role": "assistant", "content": "",
        "tool_calls": [{"function": {"name": "get_risk_scores", "arguments": {}}}],
    }
    final_answer = {"role": "assistant", "content": "Acme is your riskiest contract."}
    with patch("app.services.agent._call_ollama_chat", side_effect=[tool_request, final_answer]), \
         patch("app.services.agent.risk_scoring.compute_risk_scores",
               return_value=[{"customerName": "Acme", "contractId": 1, "riskScore": 0.9,
                              "level": "HIGH", "anomalies": []}]) as mock_risk:
        result = ask_agent(db, "Which contract is riskiest?", org_id=5)
    mock_risk.assert_called_once_with(db, 5, None)
    assert result == {"answer": "Acme is your riskiest contract.", "error": None, "proposedAction": None}


def test_ask_agent_sends_a_token_to_the_model_and_resolves_it_in_the_answer():
    """The actual fix: the tool result the model sees must carry a token,
    never the raw risk score, and whatever token the model echoes back must
    resolve to the real figure in the final answer - same number-fidelity
    guarantee generate_insights has, applied to the free-text Q&A flow."""
    db = MagicMock()
    tool_request = {
        "role": "assistant", "content": "",
        "tool_calls": [{"function": {"name": "get_risk_scores", "arguments": {}}}],
    }
    # The model echoes the token it was given, exactly as instructed.
    final_answer = {"role": "assistant", "content": "Il contratto Acme ha un punteggio di rischio di [T1]."}
    with patch("app.services.agent._call_ollama_chat", side_effect=[tool_request, final_answer]) as mock_chat, \
         patch("app.services.agent.risk_scoring.compute_risk_scores",
               return_value=[{"customerName": "Acme", "contractId": 1, "riskScore": 0.91,
                              "level": "HIGH", "anomalies": []}]):
        result = ask_agent(db, "Which contract is riskiest?", org_id=5)

    # The raw score must never have reached the model in the tool message.
    tool_message_sent = mock_chat.call_args_list[1].args[0][-1]
    assert tool_message_sent["role"] == "tool"
    assert "0.91" not in tool_message_sent["content"]
    assert "[T1]" in tool_message_sent["content"]

    # The final answer has the real figure substituted back in, not the token.
    assert result["answer"] == "Il contratto Acme ha un punteggio di rischio di 0.91."


def test_ask_agent_passes_manager_id_through_to_tools():
    db = MagicMock()
    tool_request = {
        "role": "assistant", "content": "",
        "tool_calls": [{"function": {"name": "get_risk_scores", "arguments": {}}}],
    }
    final_answer = {"role": "assistant", "content": "Acme is your riskiest contract."}
    with patch("app.services.agent._call_ollama_chat", side_effect=[tool_request, final_answer]), \
         patch("app.services.agent.risk_scoring.compute_risk_scores",
               return_value=[{"customerName": "Acme", "contractId": 1, "riskScore": 0.9,
                              "level": "HIGH", "anomalies": []}]) as mock_risk:
        ask_agent(db, "Which contract is riskiest?", org_id=5, manager_id=42)
    mock_risk.assert_called_once_with(db, 5, 42)


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
    assert result == {"answer": "Best I can say without more tool calls.", "error": None, "proposedAction": None}


def test_ask_agent_treats_a_leaked_tool_call_as_an_error_not_an_answer():
    """Regression test for the real failure found live via
    scripts/eval_ask_agent.py (2026-10-19): the forced final call (no tools
    offered) still produced tool-call-shaped JSON as plain text instead of
    a natural-language answer. That must never reach the user as if it
    were a valid response."""
    db = MagicMock()
    always_wants_tool = {
        "role": "assistant", "content": "",
        "tool_calls": [{"function": {"name": "get_risk_scores", "arguments": {}}}],
    }
    garbled_final = {"role": "assistant", "content": '{"name":"get_risk_scores","parameters":{"}}'}
    with patch("app.services.agent._call_ollama_chat",
               side_effect=[always_wants_tool, always_wants_tool, always_wants_tool, garbled_final]), \
         patch("app.services.agent.risk_scoring.compute_risk_scores", return_value=[]):
        result = ask_agent(db, "Loop forever?", org_id=1)
    assert result["answer"] is None
    assert "non è riuscito a generare una risposta valida" in result["error"]


def test_ask_agent_treats_a_leaked_tool_call_as_an_error_on_the_first_turn_too():
    db = MagicMock()
    garbled_first_turn = {"role": "assistant", "content": '{"name": "get_forecast", "parameters": {}}'}
    with patch("app.services.agent._call_ollama_chat", return_value=garbled_first_turn):
        result = ask_agent(db, "Any question", org_id=1)
    assert result["answer"] is None
    assert "non è riuscito a generare una risposta valida" in result["error"]


def test_ask_agent_returns_error_when_ollama_unavailable():
    db = MagicMock()
    with patch("app.services.agent._call_ollama_chat", side_effect=httpx.ConnectError("refused")):
        result = ask_agent(db, "Any question", org_id=1)
    assert result["answer"] is None
    assert "Servizio AI non disponibile" in result["error"]


def test_ask_agent_handles_malformed_ollama_json():
    """Regression test for A4: same malformed-JSON handling as generate_insights,
    for the tool-calling chat endpoint."""
    db = MagicMock()
    with patch("app.services.agent._call_ollama_chat", side_effect=ValueError("Expecting value: line 1 column 1")):
        result = ask_agent(db, "Any question", org_id=1)
    assert result["answer"] is None
    assert result["proposedAction"] is None
    assert "Servizio AI non disponibile" in result["error"]


def test_ask_agent_handles_unexpected_ollama_response_shape():
    """Regression test for A4: valid JSON missing the "message" key must
    degrade gracefully instead of raising an unhandled KeyError."""
    db = MagicMock()
    with patch("app.services.agent._call_ollama_chat", side_effect=KeyError("message")):
        result = ask_agent(db, "Any question", org_id=1)
    assert result["answer"] is None
    assert result["proposedAction"] is None
    assert "Servizio AI non disponibile" in result["error"]


def test_ask_agent_handles_malformed_json_on_forced_final_call():
    """Regression test for A4: the forced final call after MAX_TOOL_ITERATIONS
    goes through the same except clause as the main loop."""
    db = MagicMock()
    always_wants_tool = {
        "role": "assistant", "content": "",
        "tool_calls": [{"function": {"name": "get_risk_scores", "arguments": {}}}],
    }
    with patch("app.services.agent._call_ollama_chat",
               side_effect=[always_wants_tool, always_wants_tool, always_wants_tool, ValueError("bad json")]), \
         patch("app.services.agent.risk_scoring.compute_risk_scores", return_value=[]):
        result = ask_agent(db, "Loop forever?", org_id=1)
    assert result["answer"] is None
    assert "Servizio AI non disponibile" in result["error"]


# ── _run_tool("propose_reminder") / _propose_reminder ─────────────────────────

class CounterpartyMock:
    def __init__(self, name):
        self.name = name


class ContractMock:
    def __init__(self, id, customer_name):
        self.id = id
        self.counterparty = CounterpartyMock(customer_name) if customer_name else None


def test_propose_reminder_requires_contract_id_and_message():
    db = MagicMock()
    assert "error" in _run_tool("propose_reminder", {}, db, org_id=1)
    assert "error" in _run_tool("propose_reminder", {"contract_id": 1}, db, org_id=1)
    assert "error" in _run_tool("propose_reminder", {"message": "hi"}, db, org_id=1)


def test_propose_reminder_scopes_lookup_by_org_id():
    db = MagicMock()
    contract = ContractMock(1, "Acme")
    db.query.return_value.filter.return_value.filter.return_value.first.return_value = contract

    result = _run_tool("propose_reminder", {"contract_id": 1, "message": "Rinnovo in scadenza"}, db, org_id=7)

    assert result == {
        "proposedAction": {
            "type": "CREATE_REMINDER",
            "contractId": 1,
            "customerName": "Acme",
            "message": "Rinnovo in scadenza",
        }
    }


def test_propose_reminder_returns_error_when_contract_not_in_org():
    """A contract_id belonging to another organization must resolve to nothing —
    never leak that other org's customer name into the tool result."""
    db = MagicMock()
    db.query.return_value.filter.return_value.filter.return_value.first.return_value = None

    result = _run_tool("propose_reminder", {"contract_id": 999, "message": "hi"}, db, org_id=7)

    assert "error" in result
    assert "proposedAction" not in result


def test_propose_reminder_scopes_lookup_by_manager_id_over_org_id():
    """Regression test: when both are given (a MANAGER caller), manager_id
    must be the scope actually applied — matches _propose_reminder's org_id
    precedence contract."""
    db = MagicMock()
    contract = ContractMock(1, "Acme")
    db.query.return_value.filter.return_value.filter.return_value.first.return_value = contract

    result = _run_tool(
        "propose_reminder", {"contract_id": 1, "message": "Rinnovo in scadenza"}, db, org_id=7, manager_id=42
    )

    assert result["proposedAction"]["contractId"] == 1
    # Exactly one scoping .filter() beyond the id filter — proves org_id's
    # branch was skipped in favor of manager_id's.
    db.query.return_value.filter.return_value.filter.assert_called_once()


def test_propose_reminder_returns_error_when_contract_not_managers():
    """A contract_id belonging to another manager (even in the same org) must
    resolve to nothing."""
    db = MagicMock()
    db.query.return_value.filter.return_value.filter.return_value.first.return_value = None

    result = _run_tool(
        "propose_reminder", {"contract_id": 1, "message": "hi"}, db, org_id=7, manager_id=42
    )

    assert "error" in result
    assert "proposedAction" not in result


def test_propose_reminder_without_org_id_still_looks_up_the_contract():
    db = MagicMock()
    contract = ContractMock(2, "Beta")
    db.query.return_value.filter.return_value.first.return_value = contract

    result = _run_tool("propose_reminder", {"contract_id": 2, "message": "hi"}, db, org_id=None)

    assert result["proposedAction"]["contractId"] == 2


# ── ask_agent surfaces proposedAction ──────────────────────────────────────────

def test_ask_agent_surfaces_proposed_action_alongside_the_answer():
    db = MagicMock()
    tool_request = {
        "role": "assistant", "content": "",
        "tool_calls": [{"function": {
            "name": "propose_reminder",
            "arguments": {"contract_id": 1, "message": "Rinnovo in scadenza"},
        }}],
    }
    final_answer = {"role": "assistant", "content": "Ho preparato un promemoria, confermi?"}
    contract = ContractMock(1, "Acme")
    with patch("app.services.agent._call_ollama_chat", side_effect=[tool_request, final_answer]):
        db.query.return_value.filter.return_value.filter.return_value.first.return_value = contract
        result = ask_agent(db, "Ricordami di rinnovare il contratto Acme", org_id=7)

    assert result["answer"] == "Ho preparato un promemoria, confermi?"
    assert result["proposedAction"] == {
        "type": "CREATE_REMINDER",
        "contractId": 1,
        "customerName": "Acme",
        "message": "Rinnovo in scadenza",
    }


def test_ask_agent_proposed_action_is_none_when_no_such_tool_is_called():
    db = MagicMock()
    with patch("app.services.agent._call_ollama_chat",
               return_value={"role": "assistant", "content": "42 contracts total."}):
        result = ask_agent(db, "How many contracts?", org_id=1)
    assert result["proposedAction"] is None
