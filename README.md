# Mortgage Simulation API

A FastAPI-based microservice for comprehensive mortgage and savings simulation. This API provides detailed financial modeling capabilities for mortgage payments, savings growth, and net worth calculations over time.

## Features

- **Comprehensive Mortgage Simulation**: Fixed-rate periods followed by variable rates
- **Savings Growth Modeling**: Interest compounding and regular contributions
- **Overpayment Handling**: Flexible overpayment schedules with savings deduction
- **Payment Constraints**: Maximum payment limits with feasibility checks
- **Net Worth Calculation**: Including property value appreciation
- **Data Export**: CSV export for further analysis
- **Self-Documenting**: Automatic OpenAPI/Swagger documentation
- **Real-time Validation**: Pydantic-based request/response validation

## Installation

### Prerequisites

- Python 3.8+
- pip or poetry for dependency management

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd finances
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the API:
```bash
uvicorn api:app --reload
```

The API will be available at `http://localhost:8000`

## API Documentation

### Interactive Documentation

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Core Endpoints

#### 1. Health Check
```
GET /
```
Simple health check endpoint.

#### 2. Run Simulation
```
POST /simulate
```
Performs a complete mortgage and savings simulation.

**Request Body Structure:**
```json
{
  "mortgage": {
    "amount": 187000.0,
    "term_years": 21.0,
    "fixed_rate": 1.65,
    "fixed_term_months": 12,
    "variable_rate": 6.0,
    "max_payment_after_fixed": null
  },
  "savings": {
    "rate": 4.30,
    "monthly_contribution": 2500.0,
    "initial_balance": 170000.0
  },
  "simulation": {
    "typical_payment": 878.0,
    "asset_value": 360000.0,
    "show_years_after_payoff": 5,
    "overpayments": "18:20000,24:15000"
  }
}
```

**Response Structure:**
```json
{
  "monthly_data": [
    {
      "month": 1,
      "year": 0.083,
      "principal_start": 187000.0,
      "principal_end": 186234.56,
      "monthly_payment": 878.0,
      "overpayment": 0.0,
      "total_payment": 878.0,
      "interest_paid": 123.45,
      "principal_repaid": 754.55,
      "savings_balance_end": 172589.23,
      "savings_interest": 89.23,
      "net_worth": 346823.67,
      "annual_mortgage_rate": 1.65,
      "monthly_interest_rate": 0.001375,
      "annual_savings_rate": 4.30,
      "monthly_savings_rate": 0.003583,
      "payment_difference": 0.0
    }
  ],
  "summary_statistics": {
    "final_mortgage_balance": 0.0,
    "final_savings_balance": 425678.90,
    "final_net_worth": 785678.90,
    "min_savings_balance": 125000.0,
    "min_savings_month": 36,
    "mortgage_paid_off_month": 245,
    "fixed_term_end_balance": 185234.56
  },
  "chart_data": {
    "years": [0.083, 0.167, ...],
    "mortgage_balance": [186234.56, 185456.78, ...],
    "savings_balance": [172589.23, 175234.67, ...],
    "net_worth": [346823.67, 349790.45, ...],
    "monthly_payments": [878.0, 878.0, ...],
    "interest_paid": [123.45, 122.89, ...],
    "principal_paid": [754.55, 755.11, ...],
    "monthly_savings_data": [2500.0, 2500.0, ...],
    "interest_received": [89.23, 91.45, ...]
  },
  "warnings": [
    "Note: Mortgage fully paid off at month 245 (Year 20.4)"
  ]
}
```

#### 3. Export CSV
```
POST /simulate/csv
```
Runs the same simulation but returns results as a downloadable CSV file.

#### 4. Get Sample Request
```
GET /simulate/sample
```
Returns a properly formatted sample request with realistic default values.

#### 5. Create Overpayment Schedule
```
GET /overpayment-schedule/create
```
Utility endpoint to generate overpayment schedules based on different patterns.

**Query Parameters:**
- `term_months`: Total mortgage term in months
- `schedule_type`: Type of schedule (`none`, `fixed`, `lump_sum`, `yearly_bonus`, `custom`)
- `monthly_amount`: For `fixed` type
- `bonus_month`: For `yearly_bonus` type (1-12)
- `bonus_amount`: For `yearly_bonus` type
- `lump_sums`: For `lump_sum`/`custom` types in format `month:amount,month:amount`

## Data Models

### Core Parameters

#### Mortgage Parameters
- **amount**: Initial mortgage amount in pounds (required, > 0)
- **term_years**: Mortgage term in years (required, 0-40)
- **fixed_rate**: Fixed interest rate as percentage (required, 0-15)
- **fixed_term_months**: Fixed rate term in months (required, ≥ 0)
- **variable_rate**: Variable rate after fixed term (default: 6.0, 0-15)
- **max_payment_after_fixed**: Maximum monthly payment after fixed period (optional)

#### Savings Parameters
- **rate**: Annual savings interest rate as percentage (default: 4.30, 0-15)
- **monthly_contribution**: Monthly savings contribution in pounds (default: 2500.0, ≥ 0)
- **initial_balance**: Initial savings balance in pounds (default: 170000.0, ≥ 0)

#### Simulation Parameters
- **typical_payment**: Typical monthly payment - difference goes to savings (default: 878.0, ≥ 0)
- **asset_value**: Property value in pounds (default: 360000.0, ≥ 0)
- **show_years_after_payoff**: Years to show after mortgage payoff (default: 5, 0-20)
- **overpayments**: Overpayment string in format `month:amount,month:amount` (optional)

### Response Data

#### Monthly Data
Each month includes:
- Principal start/end amounts
- Payment breakdown (base payment, overpayment, interest, principal)
- Savings balance and interest earned
- Net worth calculation
- Interest rates applied
- Payment differences

#### Summary Statistics
- Final balances and net worth
- Minimum savings balance and timing
- Mortgage payoff information
- Fixed term end balance

#### Chart Data
Pre-processed arrays suitable for frontend charting:
- Time series data (years, balances, payments)
- Payment breakdown data
- Interest comparison data

## Usage Examples

### Basic Simulation

```python
import requests

# Basic simulation request
data = {
    "mortgage": {
        "amount": 200000,
        "term_years": 25,
        "fixed_rate": 2.5,
        "fixed_term_months": 24,
        "variable_rate": 5.0
    },
    "savings": {
        "rate": 4.0,
        "monthly_contribution": 2000,
        "initial_balance": 150000
    }
}

response = requests.post("http://localhost:8000/simulate", json=data)
result = response.json()

print(f"Final net worth: £{result['summary_statistics']['final_net_worth']:,.2f}")
```

### With Overpayments

```python
# Simulation with overpayments
data = {
    "mortgage": {
        "amount": 200000,
        "term_years": 25,
        "fixed_rate": 2.5,
        "fixed_term_months": 24,
        "variable_rate": 5.0
    },
    "savings": {
        "rate": 4.0,
        "monthly_contribution": 2000,
        "initial_balance": 150000
    },
    "simulation": {
        "overpayments": "24:20000,36:15000,48:10000"
    }
}

response = requests.post("http://localhost:8000/simulate", json=data)
```

### Generate Overpayment Schedule

```python
# Create yearly bonus overpayment schedule
params = {
    "term_months": 300,
    "schedule_type": "yearly_bonus",
    "bonus_month": 12,  # December
    "bonus_amount": 5000
}

response = requests.get("http://localhost:8000/overpayment-schedule/create", params=params)
schedule = response.json()
print(f"Overpayment string: {schedule['overpayment_string']}")
```

## Advanced Features

### Interest Calculation
- **Daily Compounding**: Mortgage interest uses daily compounding for accuracy
- **Monthly Compounding**: Savings interest uses monthly compounding
- **Payment Timing**: Interest calculated based on month-specific day counts

### Payment Constraints
- **Maximum Payment Limits**: Prevents unrealistic payment amounts
- **Interest Coverage**: Ensures payments always cover monthly interest
- **Feasibility Checks**: Validates amortization possibility under constraints

### Overpayment Handling
- **Savings Deduction**: Overpayments automatically deducted from savings
- **Insufficient Funds**: Automatic reduction if savings insufficient
- **Payment Recalculation**: Monthly payment adjusted after large overpayments

### Data Validation
- **Input Validation**: Comprehensive validation using Pydantic
- **Business Logic Validation**: Fixed term < total term, positive amounts, etc.
- **Warning System**: Non-fatal issues reported in warnings array

## Error Handling

The API provides detailed error messages for common issues:

- **400 Bad Request**: Invalid input parameters or simulation errors
- **422 Unprocessable Entity**: Request validation failures
- **500 Internal Server Error**: Unexpected server errors

Example error response:
```json
{
  "detail": "Simulation error: Fixed term (300 months) must be less than total term (240 months)"
}
```

## Testing

Run the test suite:
```bash
pytest tests/
```

Tests cover:
- Core calculation functions
- Simulation scenarios
- Edge cases and constraints
- Integration testing

## Development

### Running in Development
```bash
uvicorn api:app --reload --port 8000
```

### Code Quality
The project uses Ruff for linting and formatting:
```bash
ruff check .
ruff format .
```

### Project Structure
```
├── api.py              # FastAPI application
├── main.py             # Core simulation logic
├── requirements.txt    # Dependencies
├── pyproject.toml     # Project configuration
├── tests/             # Test suite
│   ├── test_mortgage.py
│   └── test_integration.py
└── data/              # Generated output files
```

## Deployment

### Docker (Recommended)

Create a `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t mortgage-api .
docker run -p 8000:8000 mortgage-api
```

### Production Considerations

- Use a production ASGI server (e.g., Gunicorn with Uvicorn workers)
- Configure proper logging
- Set up monitoring and health checks
- Use environment variables for configuration
- Implement rate limiting if needed

## License

MIT License - see LICENSE file for details. 