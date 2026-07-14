import json

import httpx

from ..config import settings

MAX_TEXT_CHARS = 8000

_PROMPT_TEMPLATE = (
    "You are a contract review assistant for a contract management system. "
    "Write every text field in your response in {language}.\n\n"
    "Read the following contract text and identify clauses that carry legal or "
    "financial risk for the party relying on this system, focusing on categories "
    "such as: automatic renewal, unilateral termination, penalties or liquidated "
    "damages, liability limitation or cap, exclusivity, non-compete, jurisdiction "
    "or governing law, indemnification, and price escalation.\n\n"
    "Contract text:\n"
    "{text}\n\n"
    "Respond with ONLY a JSON object in this exact shape, no other text:\n"
    '{{"clauses": [{{"category": string, "excerpt": string, "riskLevel": '
    '"HIGH"|"MEDIUM"|"LOW", "reasoning": string}}]}}\n'
    "If no risky clauses are found, respond with {{\"clauses\": []}}."
)


def _build_prompt(text: str, language: str) -> str:
    return _PROMPT_TEMPLATE.format(language=language, text=text)


def _call_ollama(prompt: str) -> str:
    response = httpx.post(
        f"{settings.OLLAMA_URL}/api/generate",
        json={
            "model": settings.OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
        },
        timeout=settings.OLLAMA_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()["response"]


def analyze_clauses(text: str) -> dict:
    if not text or not text.strip():
        return {"clauses": [], "error": "No text provided for analysis."}

    truncated = text[:MAX_TEXT_CHARS]
    prompt = _build_prompt(truncated, settings.REPORT_LANGUAGE)

    try:
        raw = _call_ollama(prompt)
    except httpx.HTTPError as exc:
        return {"clauses": [], "error": f"Ollama service unavailable: {exc}"}

    try:
        parsed = json.loads(raw)
        clauses = parsed["clauses"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return {"clauses": [], "error": "Could not parse the AI response."}

    return {"clauses": clauses, "error": None}
