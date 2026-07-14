import json
import re

import httpx

from ..config import settings

MAX_TEXT_CHARS = 8000

_CATEGORIES_IT = (
    "Rinnovo automatico, Recesso unilaterale, Penali o danni liquidati, "
    "Limitazione di responsabilità, Esclusiva, Non concorrenza, "
    "Foro competente o legge applicabile, Manleva, Adeguamento prezzi"
)

_PROMPT_TEMPLATE = (
    "You are a contract review assistant for a contract management system.\n\n"
    "Read the following contract text and identify clauses that carry legal or "
    "financial risk for the party relying on this system. Look for clauses "
    "matching these categories: {categories}.\n\n"
    "Contract text:\n"
    "{text}\n\n"
    "Respond with ONLY a JSON object in this exact shape, no other text:\n"
    '{{"clauses": [{{"category": string, "excerpt": string, "riskLevel": '
    '"HIGH"|"MEDIUM"|"LOW", "reasoning": string}}]}}\n'
    "If no risky clauses are found, respond with {{\"clauses\": []}}.\n\n"
    'IMPORTANT: the "category" value must be exactly one of the category '
    "names given above, in {language}, never translated to English. The "
    '"reasoning" value must also be written in {language}. The "excerpt" '
    "value must be copied verbatim from the contract text above, in its "
    "original language."
)


def _build_prompt(text: str, language: str) -> str:
    return _PROMPT_TEMPLATE.format(language=language, text=text, categories=_CATEGORIES_IT)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _is_grounded(excerpt: str, source_text: str) -> bool:
    return _normalize_whitespace(excerpt) in _normalize_whitespace(source_text)


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

    # The model occasionally emits a placeholder entry for a category it
    # considered but found no matching text for (empty excerpt/reasoning), or
    # fabricates a plausible-sounding excerpt that isn't actually in the
    # source text (more likely on short/sparse OCR'd input). Neither is an
    # actionable, trustworthy finding, so drop both here.
    clauses = [
        c for c in clauses
        if isinstance(c, dict) and c.get("excerpt") and _is_grounded(c["excerpt"], truncated)
    ]

    return {"clauses": clauses, "error": None}
