# Copyright (c) 2025 Donato Corbacio. All rights reserved.
# Licensed under the terms of the LICENSE file at the repository root.

import itertools
import json

import httpx
from sqlalchemy.orm import Session

from ..config import settings
from ..models import Contract
from . import forecasting, risk_scoring

TOP_RISK_CONTRACTS = 5

# Raised by _call_ollama/_call_ollama_chat when Ollama is unreachable
# (httpx.HTTPError) or returns a 2xx response we can't use — malformed JSON
# (ValueError, which json.JSONDecodeError subclasses) or valid JSON missing
# the expected "response"/"message" key (KeyError).
_OLLAMA_ERRORS = (httpx.HTTPError, ValueError, KeyError)

# Bounds how many tool-call round trips a single question can trigger, so a
# model stuck requesting tools forever can't turn one HTTP request into an
# unbounded number of Ollama calls.
MAX_TOOL_ITERATIONS = 3

# Tool schemas shown to the model. Deliberately no org/tenant parameter here:
# org_id is bound server-side from the authenticated caller (see ask_agent)
# and passed straight to the underlying compute_* functions — the model can
# never choose or override which organization's data a tool call reads.
_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_risk_scores",
            "description": (
                "Returns the caller's contracts at highest risk (expiry + unusual "
                "value anomalies), sorted highest risk first."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_forecast",
            "description": (
                "Returns the caller's historical monthly financial totals and a "
                "forecast for the given number of future months."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "months": {
                        "type": "integer",
                        "description": "Number of months to forecast ahead, 1-24.",
                    },
                },
                "required": ["months"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "propose_reminder",
            "description": (
                "Prepares a suggested in-app reminder about one contract for the user to "
                "review. This does NOT create or send anything by itself — the user must "
                "explicitly confirm the suggestion in the app before it becomes a real "
                "reminder. Use the contract_id of a contract already seen via another tool "
                "call in this conversation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "contract_id": {
                        "type": "integer",
                        "description": "id of the contract this reminder is about.",
                    },
                    "message": {
                        "type": "string",
                        "description": "Short reminder text to suggest to the user.",
                    },
                },
                "required": ["contract_id", "message"],
            },
        },
    },
]


def _format_risk_scores(risk_scores: list, placeholders: dict) -> str:
    """Formats the top contracts for the prompt. Each risk score is replaced
    with an opaque placeholder token (e.g. "[R1]") instead of being written
    as a number - the model is asked to copy the token verbatim, never to
    transcribe a decimal itself (see _build_prompt/_resolve_placeholders for
    why: no amount of prompt instruction makes an LLM 100% reliable at
    reproducing a decimal number unchanged, but it has nothing to "correct"
    in a short opaque code). `placeholders` is mutated in place, mapping
    each token to the real value it must resolve to."""
    if not risk_scores:
        return "No contracts found."

    shown = risk_scores[:TOP_RISK_CONTRACTS]
    lines = []
    for i, item in enumerate(shown):
        anomalies = ", ".join(item["anomalies"]) if item["anomalies"] else "none"
        score_token = f"[R{i + 1}]"
        placeholders[score_token] = str(item["riskScore"])
        lines.append(
            f"- {item['customerName']} (contract #{item['contractId']}): "
            f"risk score {score_token}, level {item['level']}, anomalies: {anomalies}"
        )

    # risk_scores is already sorted by riskScore descending (risk_scoring.py),
    # so `shown` is always the TOP_RISK_CONTRACTS most urgent ones - but an
    # org with more than TOP_RISK_CONTRACTS HIGH contracts would otherwise
    # have the rest silently never mentioned anywhere in the report. Never
    # hide that a limit was hit (same posture as `reliable: false` on
    # /forecast) - count only HIGH ones beyond what's shown, since omitting
    # a MEDIUM/LOW one here isn't a correctness problem (see
    # scripts/eval_agent_insights.py's name_coverage for the same distinction).
    remaining_high = sum(1 for item in risk_scores[TOP_RISK_CONTRACTS:] if item["level"] == "HIGH")
    if remaining_high:
        lines.append(f"... and {remaining_high} more HIGH-risk contract(s) not shown above.")

    return "\n".join(lines)


def _format_amount(value: float) -> str:
    """Round to the nearest euro and use a thousands separator - this is now
    only ever a PLACEHOLDER's resolved value (see _resolve_placeholders),
    never text the model itself has to copy, but kept short and clean
    regardless since it's still what ends up in the final report."""
    return f"{round(value):,}".replace(",", ".")


def _format_forecast(forecast: dict, placeholders: dict) -> str:
    historical = forecast.get("historical", [])
    forecast_months = forecast.get("forecast", [])

    if not historical and not forecast_months:
        return "No financial data available."

    lines = []
    if historical:
        recent = historical[-3:]
        recent_parts = []
        for i, h in enumerate(recent):
            token = f"[H{i + 1}]"
            placeholders[token] = _format_amount(h["amount"])
            recent_parts.append(f"{h['month']}: {token}")
        lines.append(f"Recent months: {', '.join(recent_parts)}")

    if forecast_months:
        forecast_parts = []
        for i, f in enumerate(forecast_months):
            amount_token, lower_token, upper_token = f"[F{i + 1}]", f"[F{i + 1}L]", f"[F{i + 1}U]"
            placeholders[amount_token] = _format_amount(f["amount"])
            placeholders[lower_token] = _format_amount(f["lower"])
            placeholders[upper_token] = _format_amount(f["upper"])
            forecast_parts.append(f"{f['month']}: {amount_token} (range {lower_token}-{upper_token})")
        lines.append(f"Forecast: {', '.join(forecast_parts)}")

    return "\n".join(lines)


def _build_prompt(formatted_risk_scores: str, formatted_forecast: str, language: str) -> str:
    return (
        f"You are a financial assistant for a contract management system. "
        f"Write your entire response in {language}.\n\n"
        f"Here are the contracts with the highest risk scores:\n"
        f"{formatted_risk_scores}\n\n"
        f"Here is the financial trend (historical and forecast amounts):\n"
        f"{formatted_forecast}\n\n"
        f"Write a concise report with three sections: "
        f"1) the highest-risk contracts and why, "
        f"2) the financial trend for the upcoming months, "
        f"3) recommended actions. "
        f"Every risk score and amount above is written as a short code in square "
        f"brackets, like [R1] or [F2L] - these are placeholder tokens, not numbers, "
        f"and will be substituted with the real figure automatically after you respond. "
        f"Copy each token EXACTLY as shown, square brackets included, every time you "
        f"refer to that score or amount - never write out a number yourself in its "
        f"place, never translate, reformat, round, or guess what a token's value is. "
        f"Write like a professional financial analyst addressing a colleague directly: "
        f"plain, specific sentences, no filler phrases ('it is important to note', "
        f"'we remind you that'), no restating the same point twice in different words, "
        f"no generic advice that isn't tied to a specific contract or token above. "
        f"If the contract list above ends with a line stating more HIGH-risk contracts "
        f"exist beyond those shown, explicitly state that count in section 1 - never "
        f"let the reader assume the ones named are the only high-risk contracts there are."
    )


def _resolve_placeholders(report: str, placeholders: dict) -> str:
    """Replaces every placeholder token with its real value, deterministically.
    This - not the prompt instruction above - is what actually guarantees
    number fidelity: the instruction only makes the model MORE likely to copy
    a token correctly, a short opaque code like "[R1]" is far less tempting
    for a model to "helpfully" reformat than a decimal number would be, but
    nothing here depends on the model behaving well. If a token comes back
    malformed (wrong case, a stray space), it fails safely: the bracket
    survives unresolved and is visibly wrong in the final text, instead of a
    plausible-looking but silently incorrect number."""
    for token, value in placeholders.items():
        report = report.replace(token, value)
    return report


def _call_ollama(prompt: str) -> str:
    response = httpx.post(
        f"{settings.OLLAMA_URL}/api/generate",
        json={"model": settings.OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=settings.OLLAMA_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()["response"]


def generate_insights(
    db: Session, months: int, org_id: int | None = None, manager_id: int | None = None
) -> dict:
    risk_scores = risk_scoring.compute_risk_scores(db, org_id, manager_id)
    forecast = forecasting.compute_forecast(db, months, org_id, manager_id)

    report = None
    error = None
    try:
        placeholders: dict[str, str] = {}
        formatted_risk_scores = _format_risk_scores(risk_scores, placeholders)
        formatted_forecast = _format_forecast(forecast, placeholders)
        prompt = _build_prompt(formatted_risk_scores, formatted_forecast, settings.REPORT_LANGUAGE)
        report = _resolve_placeholders(_call_ollama(prompt), placeholders)
    except _OLLAMA_ERRORS as exc:
        error = f"Servizio AI non disponibile: {exc}"

    return {
        "riskScores": risk_scores,
        "forecast": forecast,
        "report": report,
        "error": error,
    }


def _run_tool(
    name: str, arguments: dict, db: Session, org_id: int | None, manager_id: int | None = None
) -> object:
    # org_id/manager_id always come from ask_agent's own parameters, never from
    # `arguments` (the model's tool-call input) — that boundary is what keeps a
    # tool call scoped to the caller's own organization/manager regardless of
    # what the model asks for.
    if name == "get_risk_scores":
        return risk_scoring.compute_risk_scores(db, org_id, manager_id)
    if name == "get_forecast":
        months = int(arguments.get("months") or 3)
        months = min(max(months, 1), 24)
        return forecasting.compute_forecast(db, months, org_id, manager_id)
    if name == "propose_reminder":
        return _propose_reminder(arguments, db, org_id, manager_id)
    return {"error": f"Unknown tool: {name}"}


def _propose_reminder(
    arguments: dict, db: Session, org_id: int | None, manager_id: int | None = None
) -> dict:
    contract_id = arguments.get("contract_id")
    message = str(arguments.get("message") or "").strip()
    if not contract_id or not message:
        return {"error": "contract_id and message are required"}

    # Scoped by manager_id/org_id exactly like every read tool above — a
    # contract_id the model saw (or guessed) for a different manager/org must
    # resolve to nothing, never leak that customer name into this response.
    query = db.query(Contract).filter(Contract.id == contract_id)
    if manager_id is not None:
        query = query.filter(Contract.manager_id == manager_id)
    elif org_id is not None:
        query = query.filter(Contract.organization_id == org_id)
    contract = query.first()
    if contract is None:
        return {"error": f"Contract {contract_id} not found"}

    # Not a write: this is a suggestion surfaced to the user in ask_agent's
    # response. The actual notification is only created if the user confirms
    # it via the backend's own authenticated, re-validated endpoint.
    return {
        "proposedAction": {
            "type": "CREATE_REMINDER",
            "contractId": contract.id,
            "customerName": contract.counterparty.name if contract.counterparty else None,
            "message": message,
        }
    }


def _next_token(counter: "itertools.count") -> str:
    return f"[T{next(counter)}]"


def _templatize_risk_scores(risk_scores: list, placeholders: dict, counter: "itertools.count") -> list:
    """Same idea as _format_risk_scores for generate_insights, applied to a
    tool RESULT instead of prompt text: ask_agent lets the model call tools
    in any order on a free-text question, so there's no fixed [R1]/[R2]
    position scheme to reuse here - every score gets the next number in a
    single counter shared across the whole ask_agent call, so two different
    tool calls never collide on the same token. contractId is left as a
    plain number, same as generate_insights - it's an identifier, not a
    figure the model could plausibly mistranscribe."""
    templatized = []
    for item in risk_scores:
        token = _next_token(counter)
        placeholders[token] = str(item["riskScore"])
        templatized.append({**item, "riskScore": token})
    return templatized


def _templatize_forecast(forecast: dict, placeholders: dict, counter: "itertools.count") -> dict:
    def tokenize(value: float) -> str:
        token = _next_token(counter)
        placeholders[token] = _format_amount(value)
        return token

    return {
        "historical": [{**h, "amount": tokenize(h["amount"])} for h in forecast.get("historical", [])],
        "forecast": [
            {**f, "amount": tokenize(f["amount"]), "lower": tokenize(f["lower"]), "upper": tokenize(f["upper"])}
            for f in forecast.get("forecast", [])
        ],
    }


def _templatize_tool_result(name: str, result: object, placeholders: dict, counter: "itertools.count") -> object:
    """Dispatches to the right templatizer by tool name, so ask_agent's loop
    doesn't need to know which tools carry figures worth protecting.
    propose_reminder's result (and anything else) passes through unchanged -
    its only text field is one the model itself supplied as an argument, not
    a number read from our data."""
    if name == "get_risk_scores" and isinstance(result, list):
        return _templatize_risk_scores(result, placeholders, counter)
    if name == "get_forecast" and isinstance(result, dict):
        return _templatize_forecast(result, placeholders, counter)
    return result


def _call_ollama_chat(messages: list, tools: list | None) -> dict:
    payload = {"model": settings.OLLAMA_MODEL, "messages": messages, "stream": False}
    if tools:
        payload["tools"] = tools
    response = httpx.post(f"{settings.OLLAMA_URL}/api/chat", json=payload, timeout=settings.OLLAMA_TIMEOUT)
    response.raise_for_status()
    return response.json()["message"]


def ask_agent(
    db: Session, question: str, org_id: int | None = None, manager_id: int | None = None
) -> dict:
    """Answers a free-text question by letting the model call read-only tools
    (risk scores, forecast) grounded in the caller's own data, instead of
    generating a fixed report. org_id/manager_id are bound here, once, from the
    authenticated request — see _run_tool for why they never flow through the
    model's tool-call arguments.

    Every risk score/amount a tool returns is templated into an opaque
    token (_templatize_tool_result) before the model ever sees it, and
    resolved back to the real value (_resolve_placeholders) only in the
    final answer text - same guarantee as generate_insights, adapted to a
    multi-turn, open-ended conversation instead of one fixed report: a
    single `placeholders` map and token counter are shared across every
    tool call this question triggers, so two different tool calls in the
    same conversation never collide on the same token number."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a financial assistant for a contract management system. "
                f"Write your entire response in {settings.REPORT_LANGUAGE}. "
                "Use the available tools to look up real data before answering; "
                "never invent contract names, amounts, or dates. "
                "Risk scores and amounts returned by tools are given as short "
                "placeholder tokens in square brackets, like [T3] - these are not "
                "numbers, copy them verbatim whenever you refer to that score or "
                "amount, never write out a number yourself in their place."
            ),
        },
        {"role": "user", "content": question},
    ]
    proposed_action = None
    placeholders: dict[str, str] = {}
    token_counter = itertools.count(1)

    try:
        for _ in range(MAX_TOOL_ITERATIONS):
            message = _call_ollama_chat(messages, _TOOLS)
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                content = message.get("content")
                answer = _resolve_placeholders(content, placeholders) if content else content
                return {"answer": answer, "error": None, "proposedAction": proposed_action}

            messages.append(message)
            for call in tool_calls:
                function = call.get("function", {})
                name = function.get("name", "")
                result = _run_tool(name, function.get("arguments") or {}, db, org_id, manager_id)
                if isinstance(result, dict) and "proposedAction" in result:
                    proposed_action = result["proposedAction"]
                templatized = _templatize_tool_result(name, result, placeholders, token_counter)
                messages.append({"role": "tool", "content": json.dumps(templatized)})

        # Still requesting tools after MAX_TOOL_ITERATIONS — ask once more
        # without tools so the model is forced to answer with what it has.
        final = _call_ollama_chat(messages, tools=None)
        final_content = final.get("content")
        answer = _resolve_placeholders(final_content, placeholders) if final_content else final_content
        return {"answer": answer, "error": None, "proposedAction": proposed_action}
    except _OLLAMA_ERRORS as exc:
        return {"answer": None, "error": f"Servizio AI non disponibile: {exc}", "proposedAction": None}
