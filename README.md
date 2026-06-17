# bcm-v2-ml

FastAPI ML service for the BCM (Business Contracts Manager) platform. It connects
read-only to the shared MySQL database and exposes statistical/AI endpoints used
by the Spring Boot backend and the Next.js frontend.

## Architecture

BCM is composed of 4 repositories:

| Repo             | Stack                          | Port |
|------------------|---------------------------------|------|
| bcm-v2-backend   | Spring Boot, MySQL, JWT          | 8090 |
| bcm-v2-frontend  | Next.js, TypeScript, TanStack Query | 3000 |
| **bcm-v2-ml**    | FastAPI, SQLAlchemy, pandas/numpy | 8000 |
| bcm-v2-docker    | docker-compose for all services  | -    |

This service is **read-only** on the `contracts` and `financial_values` tables.
It never writes to the database.

- The frontend calls this service directly from the browser (CORS enabled).
- The backend calls `GET /risk-scores` daily (`RiskScoreRefresher`) to generate
  high-risk notifications.

## Endpoints

### `GET /health`
Liveness check used by the Docker healthcheck.

```json
{ "status": "ok" }
```

### `GET /forecast?months=3`
Aggregates `financial_values` by month/year, fits a linear regression and
returns a forecast with a 95% confidence interval. `months` must be between
1 and 24 (default 3).

```json
{
  "historical": [{ "month": "2024-01", "amount": 12000.0 }],
  "forecast": [{ "month": "2024-02", "amount": 12500.0, "lower": 11800.0, "upper": 13200.0 }]
}
```

### `GET /risk-scores`
Rule-based risk score per contract, combining contract expiry and a financial
z-score per organization.

- `expiry_score`: expired = 1.0, < 30 days = 0.8, < 90 days = 0.5, < 180 days =
  0.3, otherwise = 0.1 (no end date = 0.3)
- `risk_score = 0.6 * expiry_score + 0.4 * min(|z| / 3, 1.0)`
- `level`: `HIGH` (>= 0.65), `MEDIUM` (>= 0.35), `LOW` (otherwise)
- `anomalies`: `EXPIRED`, `EXPIRING_SOON`, `NO_END_DATE`, `UNUSUAL_VALUE` (`|z| > 2`)

```json
[
  { "contractId": 1, "customerName": "Acme", "riskScore": 0.82, "level": "HIGH", "anomalies": ["EXPIRED"] }
]
```

### `GET /agent/insights?months=3`
Combines `/risk-scores` and `/forecast` and asks a local [Ollama](https://ollama.com)
model to write a natural-language report (highest-risk contracts, financial
trend, recommended actions). The risk scoring and forecasting logic remain
fully rule-based/statistical — the model only interprets and summarizes the
results, it does not replace them. No data leaves the server.

If Ollama is unreachable, the endpoint still returns `200` with the raw
`riskScores`/`forecast` data, `report: null` and an `error` message.

```json
{
  "riskScores": [ ... ],
  "forecast": { "historical": [ ... ], "forecast": [ ... ] },
  "report": "1) Contratti a rischio piu alto...\n2) Trend finanziario...\n3) Azioni consigliate...",
  "error": null
}
```

## Running locally

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
pip install -r requirements-dev.txt   # for tests

cp .env.example .env           # adjust DB_URL etc.

uvicorn app.main:app --reload
```

The service listens on `http://localhost:8000`.

### Ollama setup (for `/agent/insights`)

```bash
ollama pull llama3.2
ollama serve
```

## Running with Docker

Built and orchestrated from `bcm-v2-docker` via `docker-compose.yml`
(service `ml`, port 8000).

If Ollama runs on the host machine while this service runs in a container, set
`OLLAMA_URL=http://host.docker.internal:11434` for the `ml` service so it can
reach the host's Ollama instance.

## Environment variables

| Variable          | Default                      | Description |
|-------------------|-------------------------------|-------------|
| `DB_URL`          | *(required)*                  | SQLAlchemy connection string, e.g. `mysql+pymysql://user:pass@host:3306/bcm` |
| `CORS_ORIGINS`    | `http://localhost:3000`       | Comma-separated list of allowed browser origins |
| `OLLAMA_URL`      | `http://localhost:11434`      | Base URL of the local Ollama server |
| `OLLAMA_MODEL`    | `llama3.2`                    | Ollama model used for `/agent/insights` |
| `OLLAMA_TIMEOUT`  | `120.0`                        | Timeout (seconds) for the Ollama request (covers cold-start model loading) |
| `REPORT_LANGUAGE` | `italian`                     | Language of the generated report |

## Synthetic data

`scripts/seed_synthetic_data.py` is a standalone offline tool (not part of the
app or test suite) that seeds `contracts` and `financial_values` with a
realistic synthetic dataset, so `/forecast` and `/risk-scores` have enough
data to produce meaningful results in local/dev environments. It requires a
DB user with write access (this service's own `DB_URL` is read-only by
convention) and reuses existing `organizations`/`business_areas`/
`financial_types`. Synthetic rows are tagged with a `SYN-<org>-<seq>` contract
number so the script can be re-run safely (`--reset` clears previous synthetic
data first).

```bash
pip install -r requirements.txt -r scripts/requirements.txt
python scripts/seed_synthetic_data.py --db-url mysql+pymysql://<user>:<password>@localhost:3307/bcm --reset
```

See the script's module docstring for the full option list (`--contracts-per-org`,
`--min-months`/`--max-months`, `--outlier-ratio`, `--seed`) and for running it
from inside the `ml` container when no local Python is available.

## Testing

```bash
pytest --cov=app --cov-report=term-missing
```

Coverage is ~100%. HTTP calls to Ollama are mocked in tests — no running
Ollama instance is required for the test suite or CI.

## License

This project is published as a personal portfolio project - see [LICENSE](./LICENSE) for full details.

**This project itself is NOT open source.** It is licensed under a **Custom Non-Commercial License**: available for review and educational purposes, but commercial use, redistribution, or SaaS offerings are prohibited without written permission.

For commercial licensing inquiries: donatocorbacio92@gmail.com
