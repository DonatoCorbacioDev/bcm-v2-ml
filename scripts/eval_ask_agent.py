# Copyright (c) 2025 Donato Corbacio. All rights reserved.
# Licensed under the terms of the LICENSE file at the repository root.

"""
Evaluate ask_agent's free-text Q&A against a handful of hand-crafted
scenarios, without a database or the FastAPI app running.

Calls the REAL ask_agent (app.services.agent), mocking only
risk_scoring.compute_risk_scores/forecasting.compute_forecast (same style
as tests/test_agent.py's own ask_agent tests) - so this eval exercises the
real tool-calling loop, templating, and placeholder resolution, not a
reimplementation of any of it.

Number fidelity is NOT checked here - since the 2026-10-19 templating fix
(see MODEL_CARD.md section 5), a figure can only ever be exactly right or
visibly unresolved, by construction (same reasoning as
eval_agent_insights.py dropping its own number_fidelity metric). The one
thing still worth checking on that front is unresolved_tokens - any
leftover "[T7]"-shaped token in the final answer, which should never
happen but would mean a token the model echoed didn't exact-match.

The metric THIS script exists for is different: anomaly_misattributions.
A live run on 2026-10-19 found the model correctly quoting two contracts'
exact risk scores (templating worked) while swapping which contract had
which anomalies in its prose - an entity-attribute binding error, not a
number one. Templating cannot fix this by design (see MODEL_CARD.md
section 5) - this script exists to MEASURE how often it happens, not to
prevent it. Scenarios below deliberately give each contract a disjoint
anomaly set, so a sentence naming contract X alongside an anomaly code
that belongs only to a DIFFERENT contract is an unambiguous, cheap-to-detect
sign of misattribution - not a full semantic check, a specific known
failure shape.

A first run with the two metrics above also turned up a THIRD, different
problem: a forced final call (after MAX_TOOL_ITERATIONS) returned
tool-call-shaped JSON as plain text instead of a natural-language answer.
That is now caught in production, not just here -
agent.py's `_looks_like_leaked_tool_call` makes `ask_agent` return it as
`result["error"]` instead of a bogus `answer` (see MODEL_CARD.md section
5). This script relies on that real guard rather than re-detecting the
pattern itself - a run_scenario result with `error` set is counted as
"rejected as a leaked tool call" in the summary, not silently dropped.

Run:
    python scripts/eval_ask_agent.py
    python scripts/eval_ask_agent.py --models llama3.2,llama3.1:8b

Requires Ollama running locally. Does NOT require MySQL/Docker.
"""

import os

os.environ.setdefault("DB_URL", "sqlite:///:memory:")

import argparse  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from unittest.mock import MagicMock, patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.services import agent as agent_service  # noqa: E402

SCENARIOS = [
    {
        "name": "two-contracts",
        "question": (
            "Quali sono i due contratti con il rischio più alto, con il punteggio "
            "esatto di ciascuno e le anomalie specifiche di ciascuno?"
        ),
        "risk_scores": [
            {"customerName": "Rossi Costruzioni Srl", "contractId": 101, "riskScore": 0.91,
             "level": "HIGH", "anomalies": ["EXPIRED"]},
            {"customerName": "Bianchi Logistica Spa", "contractId": 102, "riskScore": 0.72,
             "level": "HIGH", "anomalies": ["EXPIRED", "UNUSUAL_VALUE"]},
        ],
    },
    {
        # Same data as "two-contracts" above, but the EXACT open/comparative
        # phrasing that actually triggered the live misattribution on
        # 2026-10-19 ("Quale contratto ha il rischio piu alto..."), instead
        # of an explicit list-everything instruction - testing whether
        # question phrasing itself affects how often this happens.
        "name": "two-contracts-open-phrasing",
        "question": "Quale contratto ha il rischio piu alto e qual e il suo punteggio esatto?",
        "risk_scores": [
            {"customerName": "Rossi Costruzioni Srl", "contractId": 101, "riskScore": 0.91,
             "level": "HIGH", "anomalies": ["EXPIRED"]},
            {"customerName": "Bianchi Logistica Spa", "contractId": 102, "riskScore": 0.72,
             "level": "HIGH", "anomalies": ["EXPIRED", "UNUSUAL_VALUE"]},
        ],
    },
    {
        "name": "three-contracts-disjoint-anomalies",
        # Each contract has exactly one, different anomaly - maximizes the
        # chance a misattribution shows up as a clean "wrong code" rather
        # than a partially-correct, ambiguous overlap.
        "question": "Elenca tutti i contratti ad alto rischio con il loro punteggio esatto e la relativa anomalia.",
        "risk_scores": [
            {"customerName": "Verdi Impianti Srl", "contractId": 201, "riskScore": 0.88,
             "level": "HIGH", "anomalies": ["EXPIRED"]},
            {"customerName": "Neri Trasporti Spa", "contractId": 202, "riskScore": 0.81,
             "level": "HIGH", "anomalies": ["UNUSUAL_VALUE"]},
            {"customerName": "Gialli Energia Srl", "contractId": 203, "riskScore": 0.75,
             "level": "HIGH", "anomalies": ["NO_END_DATE"]},
        ],
    },
]

_UNRESOLVED_TOKEN_RE = re.compile(r"\[T\d+\]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def unresolved_tokens(answer: str) -> list[str]:
    return _UNRESOLVED_TOKEN_RE.findall(answer)


def anomaly_misattributions(answer: str, risk_scores: list) -> list[str]:
    """Sentence-level heuristic, not a parser: for each sentence that names
    a contract, flags any OTHER contract's anomaly code mentioned in that
    same sentence - the exact shape of the error observed live on
    2026-10-19 ("Rossi Costruzioni ... EXPIRED e UNUSUAL_VALUE" when Rossi
    only has EXPIRED). Only meaningful when scenarios give contracts
    disjoint anomaly sets, as the ones above do.

    KNOWN BLIND SPOT, found the same day this was written: this only
    matches the LITERAL code (e.g. "UNUSUAL_VALUE"). A live run produced
    "Questo contratto presenta due anomalie: 'EXPIRED' e una valenza
    insolita" - "valenza insolita" is an Italian paraphrase of
    UNUSUAL_VALUE, attributed to the wrong contract, and this function
    missed it because the literal code string never appeared. Catching a
    paraphrase would need semantic matching (e.g. an LLM-as-judge comparing
    the claimed anomaly against the real one), not a better regex - a
    different kind of tool, not a bug to patch here. Treat a "0
    misattributions" result as "no LITERAL-code misattribution found", not
    as "the model never misattributes anything" - this script currently
    cannot tell the difference."""
    all_anomalies = {code for item in risk_scores for code in item["anomalies"]}
    sentences = _SENTENCE_SPLIT_RE.split(answer)
    issues = []
    for sentence in sentences:
        for item in risk_scores:
            if item["customerName"] not in sentence:
                continue
            not_its_own = all_anomalies - set(item["anomalies"])
            wrongly_mentioned = [code for code in not_its_own if code in sentence]
            if wrongly_mentioned:
                issues.append(f"{item['customerName']}: sentence also names {wrongly_mentioned} (not its own)")
    return issues


def run_scenario(scenario: dict, model: str) -> dict:
    """Always returns a result dict, never None: an ask_agent error (e.g.
    the leaked-tool-call guard added 2026-10-19, see agent.py's
    _looks_like_leaked_tool_call) is itself a measurable outcome worth
    counting in the summary, not something to silently drop."""
    settings.OLLAMA_MODEL = model
    db = MagicMock()
    with patch("app.services.agent.risk_scoring.compute_risk_scores", return_value=scenario["risk_scores"]), \
         patch("app.services.agent.forecasting.compute_forecast", return_value={"historical": [], "forecast": []}):
        result = agent_service.ask_agent(db, scenario["question"], org_id=1)

    answer = result["answer"] or ""
    return {
        "scenario": scenario["name"],
        "model": model,
        "answer": answer,
        "error": result["error"],
        "unresolved": unresolved_tokens(answer) if not result["error"] else [],
        "misattributions": anomaly_misattributions(answer, scenario["risk_scores"]) if not result["error"] else [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--models",
        default=settings.OLLAMA_MODEL,
        help="Comma-separated Ollama model names to compare (must already be pulled).",
    )
    parser.add_argument(
        "--runs", type=int, default=1,
        help="How many times to repeat each scenario per model, since Ollama sampling is non-deterministic.",
    )
    args = parser.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    results = []
    for model in models:
        for scenario in SCENARIOS:
            for run_i in range(args.runs):
                label = f"{scenario['name']} / {model}" + (f" (run {run_i + 1})" if args.runs > 1 else "")
                print(f"--- {label} ---")
                result = run_scenario(scenario, model)
                results.append(result)
                if result["error"]:
                    print(f"ERROR: {result['error']}")
                else:
                    print(result["answer"])
                    if result["unresolved"]:
                        print(f"UNRESOLVED={result['unresolved']}")
                    if result["misattributions"]:
                        print(f"MISATTRIBUTION={result['misattributions']}")
                print()

    print("=== Summary ===")
    print("(misattribution count only catches literal anomaly codes, not Italian "
          "paraphrases like 'valenza insolita' for UNUSUAL_VALUE - see anomaly_misattributions' "
          "docstring; a low count is not proof the model never misattributes anything)")
    for model in models:
        model_results = [r for r in results if r["model"] == model]
        if not model_results:
            print(f"{model}: no runs")
            continue
        errored = sum(1 for r in model_results if r["error"])
        ok_results = [r for r in model_results if not r["error"]]
        runs_with_misattribution = sum(1 for r in ok_results if r["misattributions"])
        total_unresolved = sum(len(r["unresolved"]) for r in ok_results)
        print(
            f"{model}: {len(ok_results)}/{len(model_results)} runs answered; "
            f"{errored} rejected as a leaked tool call; "
            f"{runs_with_misattribution}/{len(ok_results) or 1} answered runs had an anomaly "
            f"misattribution; unresolved_placeholders={total_unresolved}"
        )


if __name__ == "__main__":
    main()
