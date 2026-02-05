# Mortgage Simulation API

A FastAPI microservice for mortgage and savings simulation. Provides month-by-month financial projections including mortgage amortisation, multi-account savings growth, overpayment modelling, and net worth calculation.

## Features

- **Multi-deal rate modelling** — define multiple fixed-rate deal periods; gaps fall back to the SVR/variable rate
- **Multiple savings accounts** — per-account tracking with individual rates, contributions, and balances
- **Overpayment handling** — flexible overpayment schedules deducted from savings, with automatic reduction if funds are insufficient
- **Rate-boundary recalculation** — monthly payment is recalculated whenever the interest rate changes (at deal boundaries)
- **CSV export** — download simulation results as a spreadsheet
- **Self-documenting** — OpenAPI/Swagger at `/docs`, ReDoc at `/redoc`

## Quick Start

```bash
pip install -r requirements.txt
uvicorn api:app --reload          # http://localhost:8000
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET /` | Health check |
| `POST /simulate` | Run simulation, returns JSON |
| `POST /simulate/csv` | Run simulation, returns CSV file |
| `GET /simulate/sample` | Sample request payload |
| `GET /overpayment-schedule/create` | Generate overpayment schedule strings |

### Request Body (`POST /simulate`)

```json
{
  "mortgage": {
    "amount": 200000,
    "term_years": 25,
    "variable_rate": 6.0,
    "deals": [
      { "start_month": 0, "end_month": 24, "rate": 1.65 },
      { "start_month": 24, "end_month": 60, "rate": 3.5 }
    ]
  },
  "savings": {
    "accounts": [
      { "name": "ISA", "rate": 4.3, "monthly_contribution": 2500, "initial_balance": 170000 },
      { "name": "SIPP", "rate": 5.0, "monthly_contribution": 500, "initial_balance": 50000 }
    ]
  },
  "simulation": {
    "typical_payment": 878,
    "asset_value": 360000,
    "show_years_after_payoff": 5,
    "overpayments": "18:20000,36:15000"
  }
}
```

**Key fields:**

- `mortgage.deals` — array of `{start_month, end_month, rate}` periods. If omitted, falls back to legacy `fixed_rate`/`fixed_term_months`.
- `savings.accounts` — array of named accounts. If omitted, falls back to legacy single-account `rate`/`monthly_contribution`/`initial_balance`.
- `simulation.overpayments` — comma-separated `month:amount` pairs.

### Response

Returns `monthly_data` (per-month breakdown), `summary_statistics`, `chart_data` (pre-processed arrays for charting), and `warnings`.

## Data Models

### Mortgage Parameters
- `amount` — loan amount (required, > 0)
- `term_years` — mortgage term in years (required, 0–40)
- `variable_rate` — SVR after deal periods (default 6.0%)
- `deals` — list of `Deal` objects: `{start_month, end_month, rate}`
- `fixed_rate` / `fixed_term_months` — legacy fields, used if `deals` is not provided

### Savings Parameters
- `accounts` — list of `SavingsAccount`: `{name, rate, monthly_contribution, initial_balance}`
- Legacy single-account fields (`rate`, `monthly_contribution`, `initial_balance`) still accepted

### Simulation Parameters
- `typical_payment` — base monthly payment; difference vs actual goes to savings
- `asset_value` — property value for net worth calculation
- `show_years_after_payoff` — extra years to project after mortgage hits zero
- `overpayments` — schedule string, e.g. `"18:20000,36:15000"`

## Testing

```bash
pytest tests/                      # All tests
pytest tests/test_mortgage.py      # Unit tests
pytest tests/test_integration.py   # Integration tests
```

## Development

```bash
uvicorn api:app --reload --port 8000
ruff check .          # Lint
ruff format .         # Format
```

### Project Structure

```
├── api.py              # FastAPI app, Pydantic models, endpoints
├── main.py             # Core simulation logic
├── requirements.txt    # Dependencies
├── tests/
│   ├── test_mortgage.py
│   └── test_integration.py
└── api/
    └── requirements.txt  # Vercel serverless dependencies
```

## Deployment

Deployed on **Vercel** as a serverless function via `api/requirements.txt` + `vercel.json`.

## License

Copyright (c) 2024 Romain Bossut. All Rights Reserved.
