# Model Card — BCM ML Service

This document covers the statistical/ML and LLM-backed components exposed by
this service: financial forecasting, contract risk scoring, anomaly
detection, and the two Ollama-backed endpoints (the fixed narrative report
and the free-text Q&A).
For API shapes and how the backend calls these, see [README.md](./README.md).
For where training data comes from, see
[docs/research/dataset_sources.md](./docs/research/dataset_sources.md).

## 1. Financial forecasting (`GET /forecast`)

- **Model:** [Prophet](https://facebook.github.io/prophet/) (additive trend +
  yearly seasonality), fit per-request on an organization's monthly
  `financial_values` totals. No pre-trained artifact — it refits on every call.
- **Fallback:** with fewer than 2 monthly data points, Prophet cannot fit; the
  service returns a flat forecast (last known value ± 10%) instead. If Prophet
  raises during fitting (e.g. degenerate series), the same flat fallback is
  used and the failure is logged, not surfaced as an error.
- **Confidence interval:** 95%, produced by Prophet directly (`interval_width=0.95`).
- **`reliable` flag:** yearly seasonality is only fit with 24+ months of
  history (`MIN_MONTHS_FOR_SEASONALITY` in `forecasting.py`); below that
  Prophet is a trend line only. The response includes `reliable: false`
  when history is under that same 24-month threshold so callers can show a
  caveat instead of presenting the forecast as equally trustworthy in both
  cases.
- **Limitations:** single-variable time series (amount only) — it has no
  visibility into contract-level events (a new contract signed, a large one
  expiring) that would explain a trend break. It also assumes monthly
  granularity; sparse or irregular reporting degrades the fit silently rather
  than raising a warning beyond the `reliable` flag.
- **Two different thresholds, not one** (found while writing the backtest
  below, 2026-10-19): `reliable: false` triggers under **12** months of
  history (`len(monthly) >= 12` in `compute_forecast`), but Prophet's own
  `yearly_seasonality` parameter is only enabled at **24** months
  (`yearly_seasonality=n >= 24` in `_prophet_forecast`). A 15-month history
  reports `reliable: true` to the caller while the model underneath is
  still fit *without* yearly seasonality — the flag is more optimistic
  than the model actually is in the 12–23 month range.
  **Fixed 2026-10-20**: both now read the single
  `MIN_MONTHS_FOR_SEASONALITY = 24` constant, and a unit test asserts they
  flip together. Direction chosen from evidence, not by picking a number —
  see the threshold sweep below: raising `reliable` to 24 (rather than
  lowering seasonality to 12) because a one-off probe (same 10 seeds,
  seasonality forced on) showed fitting yearly seasonality on 12–21 months
  overfits badly (median RMSE 6k–30k, beat naive in 1 of 40 runs), worse
  than the trend-only fit it would replace.

### Backtest — how accurate is the forecast, actually?

Until 2026-10-19 this section asserted Prophet's methodology but never
measured its error against a held-out answer. `scripts/backtest_forecast.py`
does walk-forward validation: hide the last 3 months of a known synthetic
series, fit `_prophet_forecast` on what's left, forecast forward, compare
against the real (hidden) values with the same error metrics reported for
the risk classifier's regression-flavored cousins — MAE, RMSE, MAPE, and
RAE/RSE (relative to an "always predict the mean" baseline) — plus the
same "beat a trivial baseline" honesty check section 2.2 uses for the
classifier, here against a naive "repeat the last known month" forecast.

| Scenario (synthetic, seeded) | Train months | RMSE (Prophet) | RMSE (naive) | Beats naive? |
|---|---|---|---|---|
| Clean trend + yearly seasonality | 27 | 584.79 | 651.52 | **Yes** |
| Same, with heavy noise | 27 | 11,761.33 | 14,437.67 | **Yes** |
| Short history (11 months, below the `reliable` threshold) | 11 | 4,794.97 | 4,151.66 | **No** |

**In plain terms**: with enough history, Prophet is genuinely better than
doing nothing clever (beats the naive baseline on both a clean and a noisy
series). Below the `reliable` threshold, it's not just "less trustworthy"
in the abstract sense the flag already implied — it concretely produced a
*worse* forecast than just repeating last month's number. This is the
first quantitative evidence behind `reliable: false`, not just a
theoretical caveat.

**Threshold sweep** (added 2026-10-20, same script, production
`_prophet_forecast`, clean series, 10 seeds per length) — where does the
threshold actually belong?

| Train months | `reliable` | Median RMSE (Prophet) | Median RMSE (naive) | Beats naive |
|---|---|---|---|---|
| 12 | false | 6,771 | 5,123 | 0/10 |
| 15 | false | 4,632 | 1,074 | 0/10 |
| 18 | false | 3,253 | 3,432 | 8/10 |
| 21 | false | 2,681 | 1,611 | 1/10 |
| 24 | true | 1,424 | 4,840 | 10/10 |

Under the old 12-month flag, the 12–21 rows were all reported
`reliable: true` while beating the naive baseline in only 9 of 40 runs.
The 18-month row is the exception — a trend line happens to suffice
there for this sine phase — which is why a single seed per length would
have been misleading. At 24, with seasonality on, Prophet wins every run.

**Honesty check, same as section 2's**: these are hand-crafted synthetic
series (known trend + yearly sine wave + Gaussian noise, fixed seed), not
real portfolios — the numbers describe how well Prophet recovers a pattern
we know is there, not real-world accuracy on actual client data. RAE/RSE
above 1.0 on all three scenarios reflect that their baseline (the mean of
only 3 held-out months) is itself a small, noisy reference point for a
short horizon - the naive-last-value comparison is the more meaningful of
the two baselines here for that reason, not RAE/RSE in isolation.

## 2. Contract risk scoring (`GET /risk-scores`)

Two scores are computed and merged into the same response — a rule-based
score (always present) and an ML score (present only if a trained model
file exists).

### 2.1 Rule-based score (baseline, always active)

```
expiry_score:  expired = 1.0 | <30d = 0.8 | <90d = 0.5 | <180d = 0.3 | else = 0.1 (no end date = 0.3)
z_score:       (contract_total - org_mean) / org_std, within the contract's organization
risk_score  =  0.6 * expiry_score + 0.4 * min(|z_score| / 3, 1.0)
level:         HIGH >= 0.65 | MEDIUM >= 0.35 | LOW otherwise
anomalies:     EXPIRED, EXPIRING_SOON, NO_END_DATE, UNUSUAL_VALUE (|z| > 2)
```

This is deterministic, has no training data or cold-start problem, and is
the only score guaranteed to be present. The 0.6/0.4 weighting and the
threshold values are hand-set, not fitted — they encode the assumption that
an imminent deadline matters more than an unusual amount, which has not been
validated against real contract outcomes (see Limitations).

### 2.2 ML score (supplementary, opt-in via trained model)

- **Model:** best of 9 candidates — 3 hyperparameter variants each of Logistic
  Regression, Random Forest and XGBoost — selected by `scripts/train_risk_model.py`
  on macro-F1 against a held-out **validation** split (the test split is never
  used for model selection or tuning), then refit on train+validation and
  wrapped in a `CalibratedClassifierCV` (isotonic, 5-fold internal CV) so
  `predict_proba` outputs are probability-calibrated, not just rank-ordered.
  On the current dataset, `RandomForest(n_estimators=200, max_depth=8)` was
  selected. Each candidate is an `imblearn` `Pipeline` (`SMOTE` → `StandardScaler`
  → classifier) — SMOTE oversamples the minority classes (MEDIUM, HIGH) using
  only that fold's own training rows, never validation or test.
- **Train/validation/test split:** grouped by `organization_id` (60/20/20) —
  no organization's contracts span more than one split, since they'd
  otherwise leak information through the shared per-org z-score baseline.
  100 synthetic organizations (not 20) specifically so each split has enough
  organizations (60/20/20) for stable metrics — with only 20 orgs the test
  split had just 4, making test numbers noisy and dependent on which 4
  happened to land there.
- **Features (7):** `days_until_expiry`, `status_code`, `has_end_date`,
  `total_financial_amount`, `num_financial_records`, `financial_std`,
  `financial_zscore` — the same signals the rule-based score uses, plus
  volume/volatility features the rules don't (`num_financial_records`,
  `financial_std`).
- **Output:** `mlScore` (`P(HIGH)` from `predict_proba`) and `mlLevel`
  (`argmax` class), merged into the rule-based result per contract.
- **Availability:** the FastAPI process loads `model/risk_model.joblib` once
  at first request. If the file is missing (model not yet trained) or
  loading/prediction fails, `ml_score`/`ml_level` are simply omitted — the
  endpoint degrades to rule-based-only, it never errors because the ML model
  isn't there.

**Current test-set performance** (from `model/risk_model_metadata.json`,
organization-grouped 60/20/20 split — 60/20/20 organizations out of 100 —
20,000 synthetic samples, seed 42, dataset generated 2026-09-05 with the
sampled-label mechanism described below — test split touched exactly once,
after model selection was final):

| Metric | Value |
|---|---|
| Validation macro F1 (selection criterion) | 0.754 |
| Test macro F1 | 0.720 |
| Test ROC-AUC (macro OvR) | 0.913 |
| Test Brier score (lower = better calibrated) | 0.193 |
| Test log-loss | 0.342 |

**Baselines, evaluated on the same test split** — the ML model must beat
these to justify existing at all:

| Baseline | Macro F1 |
|---|---|
| Majority-class (always predict LOW) | 0.257 |
| Rule-based score (section 2.1, reinterpreted as a classifier) | 0.583 |
| **ML model (this section)** | **0.720** |

The ML model beats the rule-based baseline by a real, meaningful margin
(+0.137 macro F1) — a believable result for "adds value via feature
interactions the rule ignores," unlike the previous 0.957 which meant "can
reproduce the rule that generated its own labels."

| Class | Precision | Recall | F1 |
|---|---|---|---|
| LOW | 0.94 | 0.97 | 0.96 |
| MEDIUM | 0.56 | 0.25 | 0.35 |
| HIGH | 0.78 | 0.95 | 0.86 |

MEDIUM is still the weakest class — by construction it sits in the
transition zone between the two sigmoid label boundaries (see below), the
region where the latent reliability factor has the most influence over the
outcome and the visible features are least informative. SMOTE oversampling
and a larger dataset (100 organizations, 20,000 samples, up from 20/5,000)
lifted its F1 from 0.26 to 0.35 — a real improvement, not a full fix: this
is a structural limit of the label mechanism, not an engineering gap to
close further with more of the same. The previous (2026-09-05, smaller
dataset) version of this document reported 0.957 with MEDIUM recall 0.885 —
that number was measuring the model's ability to reproduce a deterministic
rule built from the same features it was given, not predictive skill (see
"Training data — honesty check" below).

## 3. Anomaly detection (`GET /anomalies`)

- **Model:** `IsolationForest` (scikit-learn), fit fresh per-request on that
  organization's `financial_values` (amount + cyclical month encoding via
  sin/cos, so December and January are treated as adjacent, not 11 months
  apart). `contamination=0.1`, `random_state=42`.
- **Output:** records flagged as outliers (`predict == -1`), sorted by
  `decision_function` score (more negative = more anomalous), labeled
  `HIGH`/`MEDIUM`/`LOW` severity from fixed score cutoffs (-0.2 / -0.05).
- **Minimum data:** returns an empty list below 5 records for the
  organization — Isolation Forest on fewer points is not meaningful.
- **Limitations:** `contamination=0.1` is a fixed assumption (~10% of an
  org's records are anomalous), not derived from the data; an organization
  with a genuinely cleaner or noisier ledger gets the same rate forced on it.
  Unsupervised, so there is no precision/recall to report — "anomalous"
  means "statistically unusual for this org," not "verified as an error or
  fraud."

## 4. Natural-language report (`GET /agent/insights`)

- **Model:** a local [Ollama](https://ollama.com) model (`llama3.2` by
  default, configurable via `OLLAMA_MODEL`) — small enough to run on
  consumer hardware with no external API key or per-request cost, which
  matters for a self-hosted tool. No contract data leaves the server.
- **What it does and does not influence:** the endpoint calls sections 1
  and 2 above (`compute_forecast`, `compute_risk_scores`) first and passes
  their *already-computed* output into the prompt. The model only writes a
  narrative summary of numbers it's given — it does not re-derive risk
  scores or forecasts, and cannot change them. If Ollama is unreachable, the
  endpoint still returns `200` with the raw `riskScores`/`forecast` and
  `report: null` — a report-generation failure never hides the underlying
  data.
- **Quality, honestly:** with the default small model, output is coherent
  and grounded in the numbers it receives (no invented figures observed),
  but noticeably repetitive across list items and occasionally rough in
  Italian phrasing. See
  [docs/demo/agent_insights_example.md](./docs/demo/agent_insights_example.md)
  for a real, unedited captured example — not cherry-picked for quality.
  This is the weakest-tested component of the service: there is no
  automated evaluation of report quality (only that Ollama calls are mocked
  in the test suite, see [README.md](./README.md#testing)), so regressions
  in phrasing/coherence would not be caught by CI.

## 5. Free-text Q&A (`POST /agent/ask`)

- **Model:** the same local Ollama model as section 4, but via tool-calling
  (`/api/chat`) rather than a single-shot prompt: the model can call
  read-only tools (`get_risk_scores`, `get_forecast`, `propose_reminder`)
  to ground its answer in the caller's real data before responding, in up
  to `MAX_TOOL_ITERATIONS` (3) round trips.
- **Number fidelity:** fixed the same way as section 4, but structurally
  different because the model can call tools in any order any number of
  times on an open-ended question — there's no fixed prompt to template
  ahead of time. Every risk score/amount a tool returns is replaced with
  an opaque token (e.g. `[T3]`) before it reaches the model
  (`_templatize_tool_result` in `app/services/agent.py`), resolved back to
  the real value deterministically in the final answer
  (`_resolve_placeholders`) — the model never sees or has to reproduce an
  actual number. Verified live (2026-10-19): asked a real question about
  two contracts, both risk scores came back exactly correct.
- **What this does NOT fix, honestly:** in that same live run, the model
  swapped which of the two contracts had one vs two anomalies in its
  prose — it correctly echoed both contracts' tokens and anomaly codes,
  but attributed them to the wrong contract. This is an entity-attribute
  binding error, not a number-fidelity one: templating guarantees a
  *figure* is never mistranscribed, it says nothing about whether the
  model correctly associates *other* facts (anomaly lists, dates,
  categorical fields) with the right named entity across several items in
  context. No mitigation has been attempted for this — it's a reasoning
  limitation of a small local model holding multiple items in context at
  once, not something a prompt instruction or a deterministic substitution
  step can close the way it did for numbers.
- **No automated eval**: unlike section 4's `/agent/insights`
  (`scripts/eval_agent_insights.py`), `ask_agent` has no equivalent
  harness checking answer quality against fixed scenarios — only that
  Ollama calls are mocked in the test suite. The entity-binding error
  above was found by one manual live run, not caught by anything
  automated; its real frequency is unknown.

## Training data — honesty check

The risk-scoring classifier (2.2) is trained on **synthetic data**. Contract
amounts are drawn from a log-normal distribution whose parameters were
calibrated against real Italian public procurement data (ANAC open data, CC
BY 4.0 — see `scripts/analyze_anac.py`), so the amounts are realistic even
though the labels are not observed outcomes.

Until 2026-09-05, the ground-truth label was generated by the **same kind of
threshold rule** as the rule-based score in section 2.1 (plus 5% uniform
label noise) — a classifier trained on the same 7 features the rule reads
could reproduce that rule almost exactly, which is why the previous version
of this model reported a 0.957 macro-F1: that number measured
rule-reproduction, not predictive skill.

The label mechanism now **samples** the outcome from class probabilities
driven by the visible signal blended with a **latent counterparty
reliability factor that is never exposed to the model** (see
[docs/research/dataset_sources.md](./docs/research/dataset_sources.md) for
the exact mechanism). Because part of what determines the true label is
structurally invisible to the model, the ceiling for what any classifier fit
on these 7 features can achieve is now genuinely bounded below 1.0 — the
current macro-F1 is **0.720**, with the MEDIUM class (the hardest, sitting
in the transition zone) at 0.35 F1 (up from 0.26 on an earlier, smaller
dataset, after adding SMOTE oversampling and more organizations/samples —
still the weakest class by construction, not fully fixable this way).

**In plain terms: this model has still not learned from real contract
outcomes — no synthetic dataset can produce that — but it can no longer get
an artificially high score by memorizing a rule built from its own features.**
The reported 0.720 macro-F1 is a more honest number precisely because it is
lower than 0.957: it reflects how much of a partially-observable outcome can be
recovered from 7 visible signals, which is the actual question a real
deployment would face. Treat the ML score as a second opinion that mostly
agrees with the rule-based score on clear-cut cases (expired, extreme
z-score) and is least reliable exactly where it matters most — the MEDIUM
band — not as independently-validated risk prediction.

## Intended use

- Decision **support** for admins/managers reviewing a contract portfolio —
  surfacing which contracts to look at first, not an automated
  approve/reject/renew decision.
- Internal tool for a single organization's own data, always tenant-scoped
  by `org_id` from the JWT (see [README.md](./README.md) for the
  authentication/proxy flow) — not exposed to end customers directly.

## Out-of-scope use

- Legal or financial advice. Nothing here should be presented to a customer
  or used as grounds for contract termination without human review.
- Any real-world validation of accuracy. Until the model is trained on real
  labeled outcomes (see [docs/research/dataset_sources.md](./docs/research/dataset_sources.md)
  for candidate sources), the 0.720 macro-F1 above describes how well the
  model recovers a synthetic, partially-latent label mechanism, not
  real-world predictive accuracy — do not quote it as the latter.

## When to retrain

`scripts/train_risk_model.py` is not run automatically — there is no
scheduled retraining or drift monitoring. Re-run it (after regenerating or
replacing `data/synthetic_contracts.csv`) whenever the label rule in
`generate_training_data.py` changes, or when real labeled data becomes
available per `dataset_sources.md`.
