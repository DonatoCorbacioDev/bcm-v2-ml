# Copyright (c) 2025 Donato Corbacio. All rights reserved.
# Licensed under the terms of the LICENSE file at the repository root.

"""
Evaluate /agent/insights report quality against a handful of hand-crafted
scenarios, without a database or the FastAPI app running.

Reuses the REAL prompt-building and Ollama-calling code from
app.services.agent (_build_prompt/_call_ollama/_format_risk_scores/
_format_forecast) instead of reimplementing them, so this eval exercises
exactly what production runs. The prompt itself instructs the model to
"copy the exact digits given above verbatim - never recalculate... or
retype a number from memory" (see agent.py); this script checks whether it
actually does.

Two metrics, both computed from exact-substring matching (same idea as
clause_risk.py's _is_grounded check, applied to names/numbers instead of a
quoted excerpt):

- name_coverage: of the top-N customer names fed into the prompt, what
  fraction appear verbatim in the generated report. Low coverage usually
  means the model summarized vaguely instead of naming contracts.
- number_fidelity: of the numbers the report contains, what fraction were
  actually given to it (vs. invented or recalculated). This is a simple
  regex-based check, not a full parser - a model that reformats "48.230"
  as "48230" (no thousands separator) will be flagged as "unknown", which
  is a false positive worth knowing about, not a bug to silently hide.

Run:
    python scripts/eval_agent_insights.py
    python scripts/eval_agent_insights.py --models llama3.2,llama3.1:8b,mistral

Requires Ollama running locally (OLLAMA_URL in .env, default
http://localhost:11434) with each requested model already pulled. Does NOT
require MySQL/Docker - the scenarios below are hand-crafted, not read from
a database.
"""

import os

# app.services.agent -> app.models -> app.database creates a SQLAlchemy
# engine from settings.DB_URL at IMPORT time (see app/database.py). This
# script never runs a query, so a syntactically valid placeholder is enough
# - we only need pydantic to stop demanding a real one. setdefault() leaves
# a real DB_URL alone if one is already set in the environment.
os.environ.setdefault("DB_URL", "sqlite:///:memory:")

import argparse  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.services import agent as agent_service  # noqa: E402

SCENARIOS = [
    {
        "name": "clear-cut-expired",
        "risk_scores": [
            {"customerName": "Rossi Costruzioni Srl", "contractId": 101, "riskScore": 0.91,
             "level": "HIGH", "anomalies": ["EXPIRED"]},
            {"customerName": "Bianchi Logistica Spa", "contractId": 102, "riskScore": 0.72,
             "level": "HIGH", "anomalies": ["EXPIRED", "UNUSUAL_VALUE"]},
            {"customerName": "Verdi Consulting", "contractId": 103, "riskScore": 0.40,
             "level": "MEDIUM", "anomalies": []},
            {"customerName": "Neri Impianti", "contractId": 104, "riskScore": 0.15,
             "level": "LOW", "anomalies": []},
        ],
        "forecast": {
            "historical": [
                {"month": "2026-04", "amount": 48230.0},
                {"month": "2026-05", "amount": 51010.0},
                {"month": "2026-06", "amount": 46500.0},
            ],
            "forecast": [
                {"month": "2026-07", "amount": 49800.0, "lower": 42000.0, "upper": 57500.0},
                {"month": "2026-08", "amount": 50200.0, "lower": 43000.0, "upper": 58000.0},
            ],
        },
    },
    {
        "name": "no-high-risk",
        "risk_scores": [
            {"customerName": "Gialli Trasporti", "contractId": 201, "riskScore": 0.20,
             "level": "LOW", "anomalies": []},
            {"customerName": "Blu Energia Srl", "contractId": 202, "riskScore": 0.10,
             "level": "LOW", "anomalies": []},
        ],
        "forecast": {
            "historical": [
                {"month": "2026-05", "amount": 12000.0},
                {"month": "2026-06", "amount": 12500.0},
            ],
            "forecast": [
                {"month": "2026-07", "amount": 12700.0, "lower": 10000.0, "upper": 15000.0},
            ],
        },
    },
]

# Matches both the Italian-style "48.230" _format_amount produces and a
# plain "0.91"/"123" fallback, so a number the model reformats slightly
# differently from the input is still recognized as the SAME number, not a
# new, unrelated token. Also matches an Italian decimal comma ("0,91") so
# that localizing the separator is normalized away below instead of being
# counted as two unrelated small numbers ("0" and "91").
_NUMBER_RE = re.compile(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:[.,]\d+)?")

# First-run finding (2026-10-16): a first pass flagged "Sezione 1"/"Sezione
# 2"/"Sezione 3" (the report's own section numbering, not a data figure) as
# "unknown" numbers - a false positive of the naive regex above, not a real
# hallucination. Bare single-digit numbers are overwhelmingly this kind of
# structural noise (section/list numbering) rather than a financial figure,
# so they're excluded from the fidelity check entirely rather than compared.
_STRUCTURAL_NUMBER_MAX_DIGITS = 1


def _normalize_decimal(token: str) -> str:
    """"0,91" and "0.91" must compare equal - the model localizing the
    decimal separator is a (now explicitly disallowed, see _build_prompt)
    formatting choice, not a different number. Only the LAST separator is
    treated as decimal; "48.230" has none to normalize."""
    if "," in token and "." not in token:
        return token.replace(",", ".")
    return token


def extract_numbers(text: str) -> set[str]:
    raw = _NUMBER_RE.findall(text)
    return {
        _normalize_decimal(token)
        for token in raw
        if not (token.isdigit() and len(token) <= _STRUCTURAL_NUMBER_MAX_DIGITS)
    }


def expected_numbers(risk_scores: list, forecast: dict) -> set[str]:
    """The exact number tokens actually given to the model, built the same
    way the real prompt is (via agent_service's own formatting functions)
    so this can never drift from what production sends."""
    formatted = (
        agent_service._format_risk_scores(risk_scores)
        + "\n"
        + agent_service._format_forecast(forecast)
    )
    return extract_numbers(formatted)


def name_coverage(report: str, risk_scores: list) -> float:
    top_names = [r["customerName"] for r in risk_scores[: agent_service.TOP_RISK_CONTRACTS]]
    if not top_names:
        return 1.0
    found = sum(1 for name in top_names if name in report)
    return found / len(top_names)


def number_fidelity(report: str, risk_scores: list, forecast: dict) -> tuple[float, set[str]]:
    report_numbers = extract_numbers(report)
    if not report_numbers:
        return 1.0, set()
    allowed = expected_numbers(risk_scores, forecast)
    unknown = report_numbers - allowed
    return 1 - (len(unknown) / len(report_numbers)), unknown


def run_scenario(scenario: dict, model: str) -> dict | None:
    settings.OLLAMA_MODEL = model
    prompt = agent_service._build_prompt(
        scenario["risk_scores"], scenario["forecast"], settings.REPORT_LANGUAGE
    )
    try:
        report = agent_service._call_ollama(prompt)
    except agent_service._OLLAMA_ERRORS as exc:
        print(f"  [{scenario['name']} / {model}] Ollama call failed: {exc}")
        return None

    coverage = name_coverage(report, scenario["risk_scores"])
    fidelity, unknown_numbers = number_fidelity(report, scenario["risk_scores"], scenario["forecast"])
    return {
        "scenario": scenario["name"],
        "model": model,
        "report": report,
        "name_coverage": coverage,
        "number_fidelity": fidelity,
        "unknown_numbers": unknown_numbers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--models",
        default=settings.OLLAMA_MODEL,
        help="Comma-separated Ollama model names to compare (must already be pulled).",
    )
    args = parser.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    results = []
    for model in models:
        for scenario in SCENARIOS:
            print(f"--- {scenario['name']} / {model} ---")
            result = run_scenario(scenario, model)
            if result is None:
                continue
            results.append(result)
            print(result["report"])
            unknown_note = f"  unknown={sorted(result['unknown_numbers'])}" if result["unknown_numbers"] else ""
            print(
                f"name_coverage={result['name_coverage']:.0%}  "
                f"number_fidelity={result['number_fidelity']:.0%}{unknown_note}"
            )
            print()

    if not results:
        print("No scenario produced a result - is Ollama running?")
        return

    print("=== Summary ===")
    for model in models:
        model_results = [r for r in results if r["model"] == model]
        if not model_results:
            print(f"{model}: no successful runs")
            continue
        avg_coverage = sum(r["name_coverage"] for r in model_results) / len(model_results)
        avg_fidelity = sum(r["number_fidelity"] for r in model_results) / len(model_results)
        print(f"{model}: avg name_coverage={avg_coverage:.0%}  avg number_fidelity={avg_fidelity:.0%}")


if __name__ == "__main__":
    main()
