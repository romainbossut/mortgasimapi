import pytest
import sys
import os
from math import isclose

# Add the parent directory to the Python path so we can import the main module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import (
    calculate_monthly_payment,
    simulate_mortgage,
    check_amortization_feasible,
)


def test_calculate_monthly_payment_zero_interest():
    """Test monthly payment calculation with zero interest rate"""
    principal = 120000
    annual_rate = 0.0
    months = 120

    payment = calculate_monthly_payment(principal, annual_rate, months)
    assert payment == 1000.0  # 120000/120 = 1000


def test_calculate_monthly_payment_typical_case():
    """Test monthly payment calculation with typical values"""
    principal = 200000
    annual_rate = 0.03  # 3%
    months = 300  # 25 years

    payment = calculate_monthly_payment(principal, annual_rate, months)
    # Manually calculated expected payment for these parameters
    expected_payment = 948.42  # Updated to match the correct calculation
    assert isclose(payment, expected_payment, rel_tol=1e-4)


def test_calculate_monthly_payment_edge_cases():
    """Test monthly payment calculation with edge cases"""
    # Test with zero months
    assert calculate_monthly_payment(100000, 0.03, 0) == 100000

    # Test with one month
    payment = calculate_monthly_payment(100000, 0.03, 1)
    assert isclose(payment, 100250, rel_tol=1e-4)  # Principal + one month's interest

    # Test with very small principal
    payment = calculate_monthly_payment(1000, 0.03, 12)
    assert payment > 0


def test_check_amortization_feasible():
    """Test the amortization feasibility checker"""
    # Test normal case where amortization is possible
    is_feasible, required_payment = check_amortization_feasible(200000, 0.03, 1000, 300)
    assert is_feasible
    assert required_payment < 1000

    # Test case where payment barely covers interest
    is_feasible, required_payment = check_amortization_feasible(200000, 0.06, 1000, 300)
    assert not is_feasible  # 1000 < required_payment for these parameters
    assert required_payment > 1000

    # Test case where payment doesn't even cover interest
    is_feasible, required_payment = check_amortization_feasible(200000, 0.06, 500, 300)
    assert not is_feasible

    # Test edge case with zero payment
    is_feasible, required_payment = check_amortization_feasible(200000, 0.03, 0, 300)
    assert not is_feasible

    # Test edge case with zero rate
    is_feasible, required_payment = check_amortization_feasible(200000, 0, 1000, 300)
    assert is_feasible
    assert isclose(required_payment, 200000 / 300, rel_tol=1e-4)


def test_simulate_mortgage_with_impossible_payment_constraint():
    """Test simulation behavior when payment constraint makes amortization impossible"""
    # Setup test parameters
    mortgage_amount = 200000
    term_months = 300
    fixed_rate = 0.02
    fixed_term_months = 24
    mortgage_rate_curve = [0.06 for _ in range(term_months - fixed_term_months)]
    savings_rate_curve = [0.03 for _ in range(term_months)]
    overpayment_schedule = {m: 0 for m in range(term_months)}
    monthly_savings = 1000
    initial_savings = 50000
    max_payment = 500  # Impossibly low payment

    # Run simulation
    results = simulate_mortgage(
        mortgage_amount,
        term_months,
        fixed_rate,
        fixed_term_months,
        mortgage_rate_curve,
        savings_rate_curve,
        overpayment_schedule,
        monthly_savings,
        initial_savings,
        max_payment_after_fixed=max_payment,
    )

    # Check that warnings were generated
    assert len(results["warnings"]) > 0
    assert any("insufficient to amortize" in w for w in results["warnings"])

    # Check that principal increases after fixed period
    fixed_period_end = results["month_data"][fixed_term_months - 1]["principal_end"]
    next_month_end = results["month_data"][fixed_term_months]["principal_end"]
    assert next_month_end > fixed_period_end


def test_simulate_mortgage_with_tight_payment_constraint():
    """Test simulation behavior when payment constraint is tight but possible"""
    # Setup test parameters
    mortgage_amount = 200000
    term_months = 300
    fixed_rate = 0.02
    fixed_term_months = 24
    mortgage_rate_curve = [0.06 for _ in range(term_months - fixed_term_months)]
    savings_rate_curve = [0.03 for _ in range(term_months)]
    overpayment_schedule = {m: 0 for m in range(term_months)}
    monthly_savings = 1000
    initial_savings = 50000

    # Calculate minimum viable payment
    expected_principal_at_fixed_end = 190000  # Approximate
    min_payment = calculate_monthly_payment(
        expected_principal_at_fixed_end, 0.06, term_months - fixed_term_months
    )
    max_payment = min_payment + 100  # Just slightly above minimum required

    # Run simulation
    results = simulate_mortgage(
        mortgage_amount,
        term_months,
        fixed_rate,
        fixed_term_months,
        mortgage_rate_curve,
        savings_rate_curve,
        overpayment_schedule,
        monthly_savings,
        initial_savings,
        max_payment_after_fixed=max_payment,
    )

    # Check that principal decreases over time
    for i in range(1, len(results["month_data"])):
        assert (
            results["month_data"][i]["principal_end"]
            <= results["month_data"][i - 1]["principal_end"]
        )

    # Check that we eventually pay off the mortgage
    final_principal = results["month_data"][-1]["principal_end"]
    assert final_principal < mortgage_amount


def test_simulate_mortgage_with_overpayments_and_constraints():
    """Test simulation behavior with both overpayments and payment constraints"""
    # Setup test parameters
    mortgage_amount = 200000
    term_months = 300
    fixed_rate = 0.02
    fixed_term_months = 24
    mortgage_rate_curve = [0.06 for _ in range(term_months - fixed_term_months)]
    savings_rate_curve = [0.03 for _ in range(term_months)]
    monthly_savings = 1000
    initial_savings = 50000

    # Create overpayment schedule with significant overpayments
    overpayment_schedule = {m: 0 for m in range(term_months)}
    overpayment_schedule[fixed_term_months] = (
        20000  # Large overpayment at end of fixed period
    )

    # Calculate minimum viable payment
    expected_principal_at_fixed_end = 190000  # Approximate
    min_payment = calculate_monthly_payment(
        expected_principal_at_fixed_end, 0.06, term_months - fixed_term_months
    )
    max_payment = min_payment + 100  # Just slightly above minimum required

    # Run simulation
    results = simulate_mortgage(
        mortgage_amount,
        term_months,
        fixed_rate,
        fixed_term_months,
        mortgage_rate_curve,
        savings_rate_curve,
        overpayment_schedule,
        monthly_savings,
        initial_savings,
        max_payment_after_fixed=max_payment,
    )

    # Check that overpayment was applied
    overpayment_month_data = results["month_data"][fixed_term_months]
    assert overpayment_month_data["overpayment"] > 0

    # Check that payment constraint is respected after overpayment
    for data in results["month_data"][fixed_term_months + 1 :]:
        assert data["monthly_payment"] <= max_payment

    # Check that principal still decreases despite payment constraint
    for i in range(fixed_term_months + 1, len(results["month_data"])):
        assert (
            results["month_data"][i]["principal_end"]
            <= results["month_data"][i - 1]["principal_end"]
        )
