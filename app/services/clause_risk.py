# Copyright (c) 2025 Donato Corbacio. All rights reserved.
# Licensed under the terms of the LICENSE file at the repository root.

import json
import re

import httpx

from ..config import settings
from .llm_errors import unavailable

MAX_TEXT_CHARS = 8000
# What the model writes is checked like any input from another service (OWASP
# API10): the frontend shows the category and reasoning as text, and an unknown
# risk level as LOW, which would understate a risk.
RISK_LEVELS = {"HIGH": "HIGH", "MEDIUM": "MEDIUM", "LOW": "LOW", "ALTO": "HIGH", "MEDIO": "MEDIUM", "BASSO": "LOW"}
MAX_CATEGORY_CHARS = 80
MAX_REASONING_CHARS = 1000
# The small model walks through every category, making up excerpts for those
# it doesn't find, so its answer runs anywhere from 200 to 900+ tokens and,
# at about 11 tokens a second on the VM's CPU, sometimes past the timeout.
# The cap keeps each analysis under a minute; the clauses complete when the
# answer is cut off are kept (_complete_clauses).
MAX_OUTPUT_TOKENS = 500

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


def _clean_clause(clause: object, source_text: str) -> dict | None:
    """The clause with only the fields the frontend shows, each of the expected
    type and size, or None when it can't be trusted: no excerpt actually in
    the text, no category, or a risk level outside HIGH/MEDIUM/LOW."""
    if not isinstance(clause, dict):
        return None
    excerpt, category = clause.get("excerpt"), clause.get("category")
    if not isinstance(excerpt, str) or not excerpt.strip() or not _is_grounded(excerpt, source_text):
        return None
    if not isinstance(category, str) or not category.strip():
        return None
    level = RISK_LEVELS.get(str(clause.get("riskLevel") or "").strip().upper())
    if level is None:
        return None
    reasoning = clause.get("reasoning")
    return {
        "category": category.strip()[:MAX_CATEGORY_CHARS],
        "excerpt": excerpt,
        "riskLevel": level,
        "reasoning": reasoning.strip()[:MAX_REASONING_CHARS] if isinstance(reasoning, str) else "",
    }


def _call_ollama(prompt: str) -> str:
    response = httpx.post(
        f"{settings.OLLAMA_URL}/api/generate",
        json={
            "model": settings.OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"num_predict": MAX_OUTPUT_TOKENS},
        },
        timeout=settings.OLLAMA_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()["response"]


def _complete_clauses(raw: str) -> list | None:
    """The clauses written in full before the answer was cut off, read one at
    a time; the one cut in half is dropped. None when not even one is whole."""
    start = re.search(r'"clauses"\s*:\s*\[', raw)
    if start is None:
        return None
    decoder = json.JSONDecoder()
    clauses, pos = [], start.end()
    while True:
        while pos < len(raw) and raw[pos] in " \t\r\n,":
            pos += 1
        try:
            clause, pos = decoder.raw_decode(raw, pos)
        except json.JSONDecodeError:
            return clauses or None
        clauses.append(clause)


def _parse_clauses(raw: str) -> list | None:
    """The list of clauses in the model's answer, or None when it can't be read."""
    try:
        clauses = json.loads(raw)["clauses"]
    except json.JSONDecodeError:
        return _complete_clauses(raw)
    except (KeyError, TypeError):
        return None
    return clauses if isinstance(clauses, list) else None


def analyze_clauses(text: str) -> dict:
    if not text or not text.strip():
        return {"clauses": [], "error": "Nessun testo disponibile per l'analisi."}

    truncated = text[:MAX_TEXT_CHARS]
    prompt = _build_prompt(truncated, settings.REPORT_LANGUAGE)

    try:
        raw = _call_ollama(prompt)
    except httpx.HTTPError as exc:
        return {"clauses": [], "error": unavailable(exc)}

    clauses = _parse_clauses(raw)
    if clauses is None:
        return {"clauses": [], "error": "Impossibile interpretare la risposta dell'AI."}

    # The model occasionally emits a placeholder entry for a category it
    # considered but found no matching text for (empty excerpt/reasoning), or
    # fabricates a plausible-sounding excerpt that isn't actually in the
    # source text (more likely on short/sparse OCR'd input). Neither is an
    # actionable, trustworthy finding, so drop both here.
    clauses = [
        clean for clean in (_clean_clause(c, truncated) for c in clauses) if clean is not None
    ]

    return {"clauses": clauses, "error": None}
