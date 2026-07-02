# Model Card — BCM ML Service

This document covers the three statistical/ML components exposed by this
service: financial forecasting, contract risk scoring, and anomaly detection.
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
- **`reliable` flag:** Prophet needs ~12+ months of history to detect real
  yearly seasonality; below that it behaves close to a linear trend line. The
  response includes `reliable: false` when history is under 12 months so
  callers can show a caveat instead of presenting the forecast as equally
  trustworthy in both cases.
- **Limitations:** single-variable time series (amount only) — it has no
  visibility into contract-level events (a new contract signed, a large one
  expiring) that would explain a trend break. It also assumes monthly
  granularity; sparse or irregular reporting degrades the fit silently rather
  than raising a warning beyond the `reliable` flag.

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

- **Model:** `RandomForestClassifier` (scikit-learn, `class_weight="balanced"`),
  chosen automatically by `scripts/train_risk_model.py` because it scored
  highest test macro-F1 among Logistic Regression, Random Forest, and
  XGBoost on the current synthetic dataset. Wrapped in a `Pipeline` with a
  `StandardScaler`.
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
80/20 stratified split, 5,000 synthetic samples, seed 42):

| Metric | Value |
|---|---|
| Test macro F1 | 0.957 |
| 5-fold CV macro F1 | 0.955 ± 0.011 |
| Test ROC-AUC (macro OvR) | 0.966 |

| Class | Precision | Recall | F1 |
|---|---|---|---|
| LOW | 0.971 | 0.985 | 0.978 |
| MEDIUM | 0.972 | 0.885 | 0.926 |
| HIGH | 0.957 | 0.976 | 0.966 |

MEDIUM has the lowest recall (0.885) — it's the class sitting between two
thresholds in the label rule below, so it's the hardest boundary for the
model to reproduce, and the one most sensitive to the 5% label noise.

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

## Training data — honesty check

The risk-scoring classifier (2.2) is trained on **synthetic data whose
ground-truth labels are generated by the same kind of threshold rule as
section 2.1** (see `scripts/generate_training_data.py`), with 5% label noise
added so the model doesn't just memorize the rule boundary. Contract amounts
are drawn from a log-normal distribution whose parameters were calibrated
against real Italian public procurement data (ANAC open data, CC BY 4.0 —
see `scripts/analyze_anac.py`), so the amounts are realistic even though the
labels are not observed outcomes.

**In plain terms: this model has not learned from real contract outcomes —
it has learned to reproduce and generalize a hand-written rule, on
realistically-shaped but synthetic data.** The reported metrics (F1,
ROC-AUC) measure how well it matches that rule, not how well it predicts
real-world contract risk. Treat the ML score as a second opinion that
tends to agree with the rule-based score on cases the rule handles well,
and mostly adds value on feature interactions (e.g. financial volatility)
the rule ignores — not as independently-validated risk prediction.

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
  for candidate sources), the 95.7% F1 above describes agreement with a
  synthetic rule, not real-world predictive accuracy — do not quote it as
  the latter.

## When to retrain

`scripts/train_risk_model.py` is not run automatically — there is no
scheduled retraining or drift monitoring. Re-run it (after regenerating or
replacing `data/synthetic_contracts.csv`) whenever the label rule in
`generate_training_data.py` changes, or when real labeled data becomes
available per `dataset_sources.md`.
