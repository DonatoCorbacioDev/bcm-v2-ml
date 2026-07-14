from unittest.mock import MagicMock, patch

import httpx

from app.services.clause_risk import MAX_TEXT_CHARS, _build_prompt, _call_ollama, analyze_clauses


# ── _build_prompt ────────────────────────────────────────────────────────────

def test_build_prompt_includes_language_and_text():
    prompt = _build_prompt("Some contract clause", "italian")
    assert "italian" in prompt
    assert "Some contract clause" in prompt
    assert "clauses" in prompt


# ── _call_ollama ─────────────────────────────────────────────────────────────

def test_call_ollama_success():
    mock_response = MagicMock()
    mock_response.json.return_value = {"response": '{"clauses": []}'}
    with patch("app.services.clause_risk.httpx.post", return_value=mock_response) as mock_post:
        result = _call_ollama("some prompt")
    assert result == '{"clauses": []}'
    mock_post.assert_called_once()
    assert mock_post.call_args.kwargs["json"]["format"] == "json"


def test_call_ollama_http_status_error():
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "error", request=MagicMock(), response=mock_response
    )
    with patch("app.services.clause_risk.httpx.post", return_value=mock_response):
        try:
            _call_ollama("some prompt")
            assert False, "expected httpx.HTTPStatusError"
        except httpx.HTTPStatusError:
            pass


# ── analyze_clauses ──────────────────────────────────────────────────────────

def test_analyze_clauses_empty_text():
    result = analyze_clauses("")
    assert result == {"clauses": [], "error": "No text provided for analysis."}


def test_analyze_clauses_blank_text():
    result = analyze_clauses("   \n  ")
    assert result["clauses"] == []
    assert result["error"] == "No text provided for analysis."


def test_analyze_clauses_success():
    fake_clauses = [
        {"category": "auto-renewal", "excerpt": "renews automatically", "riskLevel": "HIGH", "reasoning": "..."}
    ]
    with patch(
        "app.services.clause_risk._call_ollama",
        return_value='{"clauses": ' + str(fake_clauses).replace("'", '"') + "}",
    ):
        result = analyze_clauses("Some contract text")
    assert result["error"] is None
    assert result["clauses"] == fake_clauses


def test_analyze_clauses_ollama_unavailable():
    with patch("app.services.clause_risk._call_ollama", side_effect=httpx.ConnectError("refused")):
        result = analyze_clauses("Some contract text")
    assert result["clauses"] == []
    assert "Ollama service unavailable" in result["error"]


def test_analyze_clauses_malformed_json_response():
    with patch("app.services.clause_risk._call_ollama", return_value="not json"):
        result = analyze_clauses("Some contract text")
    assert result["clauses"] == []
    assert result["error"] == "Could not parse the AI response."


def test_analyze_clauses_missing_clauses_key():
    with patch("app.services.clause_risk._call_ollama", return_value='{"foo": "bar"}'):
        result = analyze_clauses("Some contract text")
    assert result["clauses"] == []
    assert result["error"] == "Could not parse the AI response."


def test_analyze_clauses_drops_entries_with_empty_excerpt():
    raw_clauses = [
        {"category": "auto-renewal", "excerpt": "renews automatically", "riskLevel": "HIGH", "reasoning": "..."},
        {"category": "indemnification", "excerpt": "", "riskLevel": "", "reasoning": ""},
    ]
    with patch(
        "app.services.clause_risk._call_ollama",
        return_value='{"clauses": ' + str(raw_clauses).replace("'", '"') + "}",
    ):
        result = analyze_clauses("Some contract text")
    assert result["error"] is None
    assert len(result["clauses"]) == 1
    assert result["clauses"][0]["category"] == "auto-renewal"


def test_analyze_clauses_truncates_long_text():
    long_text = "a" * (MAX_TEXT_CHARS * 2)
    with patch("app.services.clause_risk._call_ollama", return_value='{"clauses": []}') as mock_call:
        analyze_clauses(long_text)
    prompt_arg = mock_call.call_args.args[0]
    assert long_text not in prompt_arg
    assert "a" * MAX_TEXT_CHARS in prompt_arg
