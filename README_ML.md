# BCM ML Service — Methodology

FastAPI service providing three ML endpoints consumed by the BCM backend via a cached proxy.

## Endpoints

| Endpoint | Model | Purpose |
|----------|-------|---------|
| `GET /risk-scores` | XGBoost / RandomForest / LogisticRegression (best by macro-F1) | Per-contract risk classification (LOW / MEDIUM / HIGH) |
| `GET /forecast?months=N` | Facebook Prophet | Monthly financial value forecast with 95% CI |
| `GET /anomalies` | Isolation Forest | Flag financially anomalous records |

---

## Risk Scoring Model

### Features

| Feature | Description |
|---------|-------------|
| `days_until_expiry` | Days to contract end date (365 if open-ended) |
| `status_code` | ACTIVE=0, EXPIRED=1, CANCELLED=2, DRAFT=3 |
| `has_end_date` | Binary flag |
| `total_financial_amount` | Sum of all financial values for the contract |
| `num_financial_records` | Count of financial value rows |
| `financial_std` | Standard deviation of monthly financial values |
| `financial_zscore` | Z-score of total vs. organization mean |

### Training

Three candidate models are trained on the same dataset and evaluated with stratified 5-fold cross-validation:

- `LogisticRegression` (L2, C=1.0)
- `RandomForestClassifier` (n_estimators=200, max_depth=8)
- `XGBClassifier` (n_estimators=200, max_depth=6, learning_rate=0.1)

The model with the highest **macro-F1** on the held-out test set (20% stratified split) is serialized to `model/risk_model.joblib` and loaded at service startup.

Labels are assigned by a rule-based heuristic during synthetic data generation:
- HIGH: contract expired with outstanding values, or z-score > 2.0, or > 80% of budget in final month
- MEDIUM: 30–60 days to expiry, or moderate financial variability
- LOW: otherwise

### Training command

```bash
DB_URL=mysql+pymysql://user:pass@localhost:3307/bcm \
python scripts/train_risk_model.py
```

Prints per-class precision/recall/F1 and the winning model name.

---

## Forecasting (Prophet)

Monthly financial values are aggregated per organization and fed to [Facebook Prophet](https://facebook.github.io/prophet/).

**Configuration:**
- `yearly_seasonality`: enabled only when ≥ 24 monthly data points are available
- `weekly_seasonality` / `daily_seasonality`: disabled (monthly granularity)
- `interval_width`: 0.95 (95% confidence interval)
- `seasonality_mode`: additive

**Reliability flag:** The API response includes `"reliable": bool`.
- `true` when n ≥ 12 months of history — Prophet can detect seasonal patterns
- `false` when n < 12 months — the model approximates a linear regression; treat the forecast as indicative only

**Fallback:** When n < 2 data points, a flat forecast (last known value ± 10% CI) is returned without calling Prophet.

---

## Anomaly Detection (Isolation Forest)

Features: `financial_amount`, `month_sin` (sin of month cycle), `month_cos` (cos of month cycle).

Parameters:
- `contamination = 0.1` (expected ~10% anomalous records in the training set)
- `random_state = 42`
- Minimum records per org: 5 (returns empty list otherwise)

Severity mapping based on Isolation Forest decision score:
- `score < -0.20` → **HIGH**
- `-0.20 ≤ score < -0.05` → **MEDIUM**
- `score ≥ -0.05` → **LOW**

---

## Synthetic Data Calibration

The `scripts/seed_synthetic_data.py` generator is calibrated on **ANAC (Autorità Nazionale Anticorruzione) open data**.

**Data source:** [dati.anticorruzione.it](https://dati.anticorruzione.it/opendata/dataset) — Banca Dati Nazionale Contratti Pubblici  
**License:** CC BY 4.0 / Italian Open Data License 2.0 (IODL 2.0)  
**Usage:** The ANAC dataset is downloaded locally and analyzed by `scripts/analyze_anac.py` to extract distribution parameters. The raw ANAC data is **not** included in this repository.

### Calibrated parameters

**Contract amounts — LogNormal(μ=10.69, σ=1.52)**
Fitted on ~280,000 "forniture e servizi" contracts (importoAggiudicazione, filtered 1k–50M EUR):
- Implied median: ~EUR 44,000
- Implied P90: ~EUR 350,000

**Monthly seasonality indices**

Italian public administration shows well-documented seasonal spending patterns (source: ANAC annual "Relazione" reports, IFEL studies on PA spending):

| Month | Index | Pattern |
|-------|-------|---------|
| Jan | 0.76 | Post-holiday, new budget not yet allocated |
| Feb | 0.84 | |
| Mar | 1.09 | Q1 end, new fiscal year allocations |
| Apr | 1.14 | Spring procurement push |
| May | 1.02 | |
| Jun | 0.94 | |
| Jul | 0.74 | Pre-summer slowdown |
| Aug | 0.46 | Ferragosto |
| Sep | 0.97 | Return from summer |
| Oct | 1.12 | Q4 budget push |
| Nov | 1.33 | Year-end acceleration |
| Dec | 1.35 | Budget exhaustion |

To regenerate calibration parameters from a newer ANAC dataset:
```bash
python scripts/analyze_anac.py --file path/to/anac_contratti.csv
```

---

## Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| fastapi | 0.111.0 | REST API |
| prophet | ≥1.1.5 | Time-series forecasting |
| scikit-learn | 1.5.2 | RandomForest, LogisticRegression, Isolation Forest |
| xgboost | ≥2.0.0 | Gradient boosting classifier |
| sqlalchemy | 2.0.31 | DB access |
| gunicorn | 22.0.0 | Production WSGI server (4 workers) |
