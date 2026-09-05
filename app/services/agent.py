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


def _format_amount(value: float) -> str:
    """Round to the nearest euro and use a thousands separator, so the model
    is given a short, clean token to copy instead of a long raw float (e.g.
    "45.231" instead of "45230.567891...") - small local models garble long
    unformatted numbers far more often than short clean ones. Italian-style
    "." thousands separator matches settings.REPORT_LANGUAGE=italian, so it
    doesn't collide with Italian's own "," decimal separator convention."""
    return f"{round(value):,}".replace(",", ".")


def _format_forecast(forecast: dict) -> str:
    historical = forecast.get("historical", [])
    forecast_months = forecast.get("forecast", [])

    if not historical and not forecast_months:
        return "No financial data available."

    lines = []
    if historical:
        recent = historical[-3:]
        recent_str = ", ".join(f"{h['month']}: {_format_amount(h['amount'])}" for h in recent)
        lines.append(f"Recent months: {recent_str}")

    if forecast_months:
        forecast_str = ", ".join(
            f"{f['month']}: {_format_amount(f['amount'])} "
            f"(range {_format_amount(f['lower'])}-{_format_amount(f['upper'])})"
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
        f"3) recommended actions. "
        f"When you mention any amount or score, copy the exact digits given above "
        f"verbatim - never recalculate, round differently, or retype a number from memory. "
        f"Write like a professional financial analyst addressing a colleague directly: "
        f"plain, specific sentences, no filler phrases ('it is important to note', "
        f"'we remind you that'), no restating the same point twice in different words, "
        f"no generic advice that isn't tied to a specific contract or number above."
    )


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
        prompt = _build_prompt(risk_scores, forecast, settings.REPORT_LANGUAGE)
        report = _call_ollama(prompt)
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
            "customerName": contract.customer_name,
            "message": message,
        }
    }


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
    model's tool-call arguments."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a financial assistant for a contract management system. "
                f"Write your entire response in {settings.REPORT_LANGUAGE}. "
                "Use the available tools to look up real data before answering; "
                "never invent contract names, amounts, or dates."
            ),
        },
        {"role": "user", "content": question},
    ]
    proposed_action = None

    try:
        for _ in range(MAX_TOOL_ITERATIONS):
            message = _call_ollama_chat(messages, _TOOLS)
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                return {"answer": message.get("content"), "error": None, "proposedAction": proposed_action}

            messages.append(message)
            for call in tool_calls:
                function = call.get("function", {})
                result = _run_tool(
                    function.get("name", ""), function.get("arguments") or {}, db, org_id, manager_id
                )
                if isinstance(result, dict) and "proposedAction" in result:
                    proposed_action = result["proposedAction"]
                messages.append({"role": "tool", "content": json.dumps(result)})

        # Still requesting tools after MAX_TOOL_ITERATIONS — ask once more
        # without tools so the model is forced to answer with what it has.
        final = _call_ollama_chat(messages, tools=None)
        return {"answer": final.get("content"), "error": None, "proposedAction": proposed_action}
    except _OLLAMA_ERRORS as exc:
        return {"answer": None, "error": f"Servizio AI non disponibile: {exc}", "proposedAction": None}
