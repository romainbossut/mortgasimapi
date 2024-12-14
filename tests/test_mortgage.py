import pytest
import sys
import os
from math import isclose

# Add the parent directory to the Python path so we can import the main module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import calculate_monthly_payment, simulate_mortgage

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

def test_full_mortgage_simulation():
    """Test the full mortgage simulation with a simple case"""
    # Setup test parameters
    mortgage_amount = 100000
    term_months = 120  # 10 years
    fixed_rate = 0.03
    fixed_term_months = 24  # 2 years
    mortgage_rate_curve = [0.04 for _ in range(term_months - fixed_term_months)]
    savings_rate_curve = [0.02 for _ in range(term_months)]
    overpayment_schedule = {m: 0 for m in range(term_months)}  # No overpayments
    monthly_savings = 500
    initial_savings = 10000
    
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
        initial_savings
    )
    
    # Check basic expectations
    assert len(results['month_data']) <= term_months
    assert results['month_data'][0]['principal_start'] == mortgage_amount
    assert results['month_data'][0]['savings_balance_end'] > initial_savings  # Should increase with first month's interest
    
    # Check that principal decreases over time
    for i in range(1, len(results['month_data'])):
        assert results['month_data'][i]['principal_start'] < results['month_data'][i-1]['principal_start']

def test_overpayment_impact():
    """Test that overpayments correctly reduce the mortgage term"""
    # Setup base case
    base_params = {
        'mortgage_amount': 100000,
        'term_months': 120,
        'fixed_rate': 0.03,
        'fixed_term_months': 24,
        'mortgage_rate_curve': [0.04 for _ in range(96)],  # 120-24
        'savings_rate_curve': [0.02 for _ in range(120)],
        'monthly_savings_contribution': 500,
        'initial_savings': 50000,
        'typical_payment': 0,
        'asset_value': 0
    }
    
    # Case 1: No overpayments
    no_overpayment = {m: 0 for m in range(120)}
    results_no_overpayment = simulate_mortgage(
        **base_params,
        overpayment_schedule=no_overpayment
    )
    
    # Case 2: With significant overpayment
    with_overpayment = {m: 0 for m in range(120)}
    with_overpayment[24] = 20000  # Add £20k overpayment after fixed period
    results_with_overpayment = simulate_mortgage(
        **base_params,
        overpayment_schedule=with_overpayment
    )
    
    # Check that both simulations ran
    assert len(results_no_overpayment['month_data']) > 0
    assert len(results_with_overpayment['month_data']) > 0
    
    # Check that the overpayment version has a lower final principal
    final_principal_no_overpayment = results_no_overpayment['month_data'][-1]['principal_end']
    final_principal_with_overpayment = results_with_overpayment['month_data'][-1]['principal_end']
    assert final_principal_with_overpayment < final_principal_no_overpayment
    
    # Check that the overpayment was actually applied
    overpayment_month_data = results_with_overpayment['month_data'][24]
    assert overpayment_month_data['overpayment'] == 20000

def test_savings_growth():
    """Test that savings grow correctly with interest and contributions"""
    mortgage_amount = 100000
    term_months = 120
    fixed_rate = 0.03
    fixed_term_months = 24
    mortgage_rate_curve = [0.04 for _ in range(term_months - fixed_term_months)]
    savings_rate_curve = [0.05 for _ in range(term_months)]  # 5% savings rate
    overpayment_schedule = {m: 0 for m in range(term_months)}
    monthly_savings_contribution = 1000
    initial_savings = 20000
    
    results = simulate_mortgage(
        mortgage_amount,
        term_months,
        fixed_rate,
        fixed_term_months,
        mortgage_rate_curve,
        savings_rate_curve,
        overpayment_schedule,
        monthly_savings_contribution,
        initial_savings
    )
    
    # Check that savings grow each month
    for i in range(1, len(results['month_data'])):
        prev_savings = results['month_data'][i-1]['savings_balance_end']
        curr_savings = results['month_data'][i]['savings_balance_end']
        # Savings should grow by at least the monthly contribution
        assert curr_savings > prev_savings + monthly_savings_contribution * 0.99  # Allow for small floating point differences

def test_monthly_payment_components():
    """Test that interest + principal repayment equals total monthly payment"""
    # Setup test parameters
    mortgage_amount = 200000
    term_months = 300  # 25 years
    fixed_rate = 0.03
    fixed_term_months = 24
    mortgage_rate_curve = [0.04 for _ in range(term_months - fixed_term_months)]
    savings_rate_curve = [0.02 for _ in range(term_months)]
    overpayment_schedule = {m: 0 for m in range(term_months)}
    monthly_savings_contribution = 500
    initial_savings = 10000
    
    results = simulate_mortgage(
        mortgage_amount,
        term_months,
        fixed_rate,
        fixed_term_months,
        mortgage_rate_curve,
        savings_rate_curve,
        overpayment_schedule,
        monthly_savings_contribution,
        initial_savings
    )
    
    # Check each month's payment components
    for month_data in results['month_data']:
        # Get the components
        monthly_payment = month_data['monthly_payment']
        interest_paid = month_data['interest_paid']
        principal_repaid = month_data['principal_repaid']
        overpayment = month_data['overpayment']
        
        # For regular payment (excluding overpayment)
        assert abs(monthly_payment - (interest_paid + principal_repaid)) < 0.01, \
            f"Month {month_data['month']}: Payment £{monthly_payment:.2f} != Interest £{interest_paid:.2f} + Principal £{principal_repaid:.2f}"
        
        # Check that principal reduction matches
        principal_reduction = month_data['principal_start'] - month_data['principal_end']
        assert abs(principal_reduction - principal_repaid - overpayment) < 0.01, \
            f"Month {month_data['month']}: Principal reduction £{principal_reduction:.2f} != Principal repaid £{principal_repaid:.2f} + Overpayment £{overpayment:.2f}"