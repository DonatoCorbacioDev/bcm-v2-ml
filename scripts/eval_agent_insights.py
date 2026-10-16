# Copyright (c) 2025 Donato Corbacio. All rights reserved.
# Licensed under the terms of the LICENSE file at the repository root.

"""
Evaluate /agent/insights report quality against a handful of hand-crafted
scenarios, without a database or the FastAPI app running.

Reuses the REAL prompt-building/placeholder/Ollama-calling code from
app.services.agent (_format_risk_scores/_format_forecast/_build_prompt/
_call_ollama/_resolve_placeholders) instead of reimplementing it, so this
eval exercises exactly what production runs.

Since 2026-10-16, agent.py no longer asks the model to copy numbers
verbatim - it never shows the model a number at all. Risk scores and
amounts are replaced with opaque placeholder tokens ("[R1]", "[F2L]", ...)
that this script's run_scenario resolves back to real figures the same way
generate_insights does, deterministically, after the model responds. That
architecture change is what actually fixed number fidelity (verified
2026-10-16: 76% -> 100% on the original scenarios); it is not something
this eval can "fail" on anymore by construction, so there is no
number_fidelity metric below - checking it would just be re-testing
Python's own str.replace().

Two things are still worth checking automatically, because the model's
BEHAVIOR (not the number substitution) can still go wrong:

- name_coverage: of the HIGH-risk customer names fed into the prompt, what
  fraction appear verbatim in the generated report (same exact-substring
  idea as clause_risk.py's _is_grounded, applied to names). Low coverage
  usually means the model summarized vaguely instead of naming contracts.
- unresolved_placeholders: any "[...]"-shaped token still present in the
  FINAL (already-resolved) report. This should always be empty - if it
  isn't, the model wrote a token in a form _resolve_placeholders couldn't
  exact-match (wrong case, a stray space, or an invented token that was
  never given to it), and it needs a human look, not an automatic fix.

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
    {
        # Checks the "... and N more HIGH-risk contract(s)" overflow note
        # (app/services/agent.py's _format_risk_scores, added 2026-10-16):
        # 7 HIGH contracts, only the top 5 are named individually.
        "name": "overflow-high-risk",
        "risk_scores": [
            # round(): production's risk_scoring.py always rounds to 4
            # decimals before this dict is built - an unrounded float here
            # would produce artifacts like 0.9299999999999999 that are a bug
            # in this test fixture, not a real risk-scoring or model issue.
            {"customerName": f"Cliente {i}", "contractId": 300 + i, "riskScore": round(0.95 - i * 0.02, 2),
             "level": "HIGH", "anomalies": ["EXPIRED"]}
            for i in range(7)
        ],
        "forecast": {
            "historical": [{"month": "2026-06", "amount": 30000.0}],
            "forecast": [{"month": "2026-07", "amount": 31000.0, "lower": 28000.0, "upper": 34000.0}],
        },
    },
]

# Any leftover token matching OUR OWN naming scheme ([R1], [H2], [F3],
# [F3L], [F3U] - see _format_risk_scores/_format_forecast) still present in
# the FINAL (post-resolution) report. A clean run has none.
#
# Deliberately NOT "any [...]-shaped text": a first run showed llama3.2
# inventing its own bracket notation for contract references, e.g.
# "[Gialli Trasporti #201]" - copying the STYLE it was taught (brackets =
# something referenceable) onto names we never asked it to bracket. That's
# the model being creative with formatting, not a resolution failure - a
# broader regex would have flagged it as one, the same false-positive shape
# as the earlier "Sezione 1/2/3" issue.
# Case-insensitive on purpose: _resolve_placeholders itself is exact-match
# and would NOT fix a case-mangled token like "[r1]" (same reasoning as
# _normalize_decimal elsewhere - detect the real failure, don't silently
# paper over it) - this check needs to catch that near-miss as unresolved,
# even though production's resolution step wouldn't have caught it either.
_BRACKET_TOKEN_RE = re.compile(r"\[(?:R\d+|H\d+|F\d+L?U?)\]", re.IGNORECASE)


def unresolved_placeholders(resolved_report: str) -> list[str]:
    return _BRACKET_TOKEN_RE.findall(resolved_report)


def name_coverage(report: str, risk_scores: list) -> float:
    """Checks only HIGH-level names by design, not every name sent to the
    prompt: the report's job (per MODEL_CARD.md's "Intended use") is
    surfacing what to look at first, so correctly omitting a LOW-risk name
    is not an error - silently dropping a HIGH one is. Falls back to all
    names when there is no HIGH contract in this batch, since there's
    nothing more important to single out."""
    top = risk_scores[: agent_service.TOP_RISK_CONTRACTS]
    high_names = [r["customerName"] for r in top if r["level"] == "HIGH"]
    target_names = high_names or [r["customerName"] for r in top]
    if not target_names:
        return 1.0
    found = sum(1 for name in target_names if name in report)
    return found / len(target_names)


def run_scenario(scenario: dict, model: str) -> dict | None:
    settings.OLLAMA_MODEL = model

    # Same 3-step flow as generate_insights: format (collects placeholders),
    # build the prompt from the already-formatted text, resolve after the
    # model responds.
    placeholders: dict[str, str] = {}
    formatted_risk_scores = agent_service._format_risk_scores(scenario["risk_scores"], placeholders)
    formatted_forecast = agent_service._format_forecast(scenario["forecast"], placeholders)
    prompt = agent_service._build_prompt(formatted_risk_scores, formatted_forecast, settings.REPORT_LANGUAGE)

    try:
        raw_report = agent_service._call_ollama(prompt)
    except agent_service._OLLAMA_ERRORS as exc:
        print(f"  [{scenario['name']} / {model}] Ollama call failed: {exc}")
        return None

    report = agent_service._resolve_placeholders(raw_report, placeholders)
    coverage = name_coverage(report, scenario["risk_scores"])
    leftover = unresolved_placeholders(report)
    return {
        "scenario": scenario["name"],
        "model": model,
        "report": report,
        "name_coverage": coverage,
        "unresolved": leftover,
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
            unresolved_note = f"  UNRESOLVED={result['unresolved']}" if result["unresolved"] else ""
            print(f"name_coverage={result['name_coverage']:.0%}{unresolved_note}")
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
        total_unresolved = sum(len(r["unresolved"]) for r in model_results)
        print(f"{model}: avg name_coverage={avg_coverage:.0%}  unresolved_placeholders={total_unresolved}")


if __name__ == "__main__":
    main()
