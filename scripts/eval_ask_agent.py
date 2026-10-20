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
section 5) - this script exists to MEASURE how often it happens. It did:
on 2026-10-20 the open-phrasing scenario misattributed in 10/16 runs,
which drove the `summary` line in agent.py's _risk_summary (1/17 after). Scenarios below deliberately give each contract a disjoint
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
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

# Each anomaly code as the literal code OR a common Italian paraphrase of it.
# Added 2026-10-20 after the literal-only version missed a real
# misattribution worded as "una valenza insolita" (UNUSUAL_VALUE). Still a
# finite list, not semantic matching - widens the net, doesn't close it.
_ANOMALY_PATTERNS = {
    "EXPIRED": re.compile(r"EXPIRED\b|scadut[oaie]", re.IGNORECASE),
    "EXPIRING_SOON": re.compile(r"EXPIRING_SOON|in scadenza|scade a breve|prossim\w* (alla )?scadenza", re.IGNORECASE),
    "UNUSUAL_VALUE": re.compile(
        r"UNUSUAL_VALUE|(valor|valenz|import)\w* (insolit|anomal|inusual|fuori)", re.IGNORECASE,
    ),
    "NO_END_DATE": re.compile(
        r"NO_END_DATE|senza (una )?data di (fine|scadenza)|data di (fine|scadenza) (mancante|assente)",
        re.IGNORECASE,
    ),
}

# "presenta due anomalie" about a contract that has one - the DOMINANT shape
# in a 21-run batch on 2026-10-20 (5/7 open-phrasing runs), and one the
# per-code check above can't see when the invented second anomaly isn't a
# recognizable code at all ("due anomalie: EXPIRED e possibili altri").
_COUNT_WORDS = {"un": 1, "una": 1, "un'": 1, "due": 2, "tre": 3, "quattro": 4}
_ANOMALY_COUNT_RE = re.compile(r"\b(un'|una|un|due|tre|quattro|\d+)\s*anomali[ae]\b", re.IGNORECASE)


def unresolved_tokens(answer: str) -> list[str]:
    return _UNRESOLVED_TOKEN_RE.findall(answer)


def anomaly_misattributions(answer: str, risk_scores: list) -> list[str]:
    """Sentence-level heuristic, not a parser: for each sentence about a
    contract, flags any OTHER contract's anomaly mentioned in that same
    sentence - the shape of the error observed live on 2026-10-19 ("Rossi
    Costruzioni ... EXPIRED e UNUSUAL_VALUE" when Rossi only has EXPIRED).
    Only meaningful when scenarios give contracts disjoint anomaly sets,
    as the ones above do.

    A sentence is "about" a contract if it names it, or - when it names
    none - if the previous sentence named exactly one (the live case read
    "Rossi Costruzioni ... . Questo contratto presenta due anomalie: ...").
    Anomalies match by literal code or a known Italian paraphrase
    (_ANOMALY_PATTERNS). Both widen the original literal-code, named-only
    check (2026-10-20), but a "0 misattributions" result still means "none
    of these known shapes found", not "the model never misattributes
    anything" - a paraphrase outside the list would still slip through."""
    all_anomalies = {code for item in risk_scores for code in item["anomalies"]}
    issues = []
    subjects: list[dict] = []
    for sentence in _SENTENCE_SPLIT_RE.split(answer):
        named = [item for item in risk_scores if item["customerName"] in sentence]
        if named:
            subjects = named
        if len(subjects) != 1:
            # Several contracts in one sentence ("Rossi e' scaduto, Bianchi ha
            # un valore insolito") can't be split per contract by this
            # heuristic - skip rather than flag a correct sentence.
            continue
        for item in subjects:
            not_its_own = all_anomalies - set(item["anomalies"])
            wrongly_mentioned = sorted(
                code for code in not_its_own
                if code in _ANOMALY_PATTERNS and _ANOMALY_PATTERNS[code].search(sentence)
            )
            if wrongly_mentioned:
                issues.append(f"{item['customerName']}: {sentence.strip()!r} mentions {wrongly_mentioned} (not its own)")
            for match in _ANOMALY_COUNT_RE.finditer(sentence):
                word = match.group(1).lower()
                claimed = int(word) if word.isdigit() else _COUNT_WORDS[word]
                if claimed != len(item["anomalies"]):
                    issues.append(
                        f"{item['customerName']}: {sentence.strip()!r} claims {claimed} anomalies "
                        f"(has {len(item['anomalies'])})"
                    )
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
    print("(misattribution detection is a heuristic - literal codes, a fixed list of Italian "
          "paraphrases and claimed counts; it can miss other wordings and occasionally "
          "false-positive on summary sentences - see anomaly_misattributions' docstring. "
          "Read the answers before trusting the count.)")
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
