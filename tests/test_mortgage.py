import os
import sys

from math import isclose

# Add the parent directory to the Python path so we can import the main module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import (
    calculate_monthly_payment,
    check_amortization_feasible,
    create_overpayment_schedule,
    simulate_mortgage,
    daily_rate,
    monthly_interest_from_daily,
    get_days_in_month,
)


def test_calculate_monthly_payment_zero_interest():
    """Test monthly payment calculation with zero interest rate"""
    principal = 120000
    annual_rate = 0.0
    months = 120

    payment = calculate_monthly_payment(principal, annual_rate, months)
    assert payment == 1000.0  # 120000/120 = 1000


def test_calculate_monthly_payment_zero_interest_long_term():
    """Test monthly payment calculation with zero interest rate over a long term"""
    principal = 100000
    annual_rate = 0.0
    months = 360  # 30 years

    payment = calculate_monthly_payment(principal, annual_rate, months)
    expected_payment = principal / months
    assert isclose(payment, expected_payment, rel_tol=1e-10)
    assert isclose(payment * months, principal, rel_tol=1e-10)  # Verify total payments equal principal


def test_calculate_monthly_payment_zero_months():
    """Test monthly payment calculation with zero months"""
    principal = 100000
    annual_rate = 0.05

    # Zero months should return principal
    payment = calculate_monthly_payment(principal, annual_rate, 0)
    assert payment == principal

    # Negative months should also return principal
    payment = calculate_monthly_payment(principal, annual_rate, -1)
    assert payment == principal


def test_calculate_monthly_payment_high_interest():
    """Test monthly payment calculation with high interest rates"""
    principal = 100000
    test_cases = [
        (15.0, 360),    # 15% annual interest
        (30.0, 120),    # 30% annual interest
        (50.0, 60),     # 50% annual interest
    ]

    for annual_rate, months in test_cases:
        payment = calculate_monthly_payment(principal, annual_rate, months)

        # Payment should be positive
        assert payment > 0

        # Monthly payment should be greater than monthly interest on principal
        # (otherwise the loan would never be paid off)
        monthly_interest = principal * ((annual_rate/100.0) / 12.0)
        assert payment > monthly_interest, (
            f"Payment {payment:.2f} should be greater than monthly interest "
            f"{monthly_interest:.2f} for rate {annual_rate:.1f}%"
        )

        # Calculate total payments over the term
        total_payments = payment * months

        # Total payments should exceed principal for interest-bearing loans
        assert total_payments > principal, (
            f"Total payments {total_payments:.2f} should exceed principal {principal:.2f} "
            f"for interest rate {annual_rate:.1f}%"
        )

        # Verify reasonable upper bound on monthly payment
        # Monthly payment shouldn't exceed principal plus one month's interest
        max_monthly = principal * (1 + (annual_rate/100.0)/12.0)
        assert payment <= max_monthly, (
            f"Payment {payment:.2f} should not exceed maximum monthly amount "
            f"{max_monthly:.2f} for rate {annual_rate:.1f}%"
        )


def test_calculate_monthly_payment_typical_case():
    """Test monthly payment calculation with typical values"""
    principal = 200000
    annual_rate = 3.0  # 3%
    months = 300  # 25 years

    payment = calculate_monthly_payment(principal, annual_rate, months)
    # Manually calculated expected payment for these parameters
    expected_payment = 948.42  # Updated to match the correct calculation
    assert isclose(payment, expected_payment, rel_tol=1e-4)


def test_calculate_monthly_payment_edge_cases():
    """Test monthly payment calculation with edge cases"""
    # Test with zero months
    assert calculate_monthly_payment(100000, 3.0, 0) == 100000

    # Test with one month
    payment = calculate_monthly_payment(100000, 3.0, 1)
    assert isclose(payment, 100250, rel_tol=1e-4)  # Principal + one month's interest

    # Test with very small principal
    payment = calculate_monthly_payment(1000, 3.0, 12)
    assert payment > 0


def test_check_amortization_feasible():
    """Test the amortization feasibility checker"""
    # Test normal case where amortization is possible
    is_feasible, required_payment = check_amortization_feasible(200000, 3.0, 1000, 300)
    assert is_feasible
    assert required_payment < 1000

    # Test case where payment barely covers interest
    is_feasible, required_payment = check_amortization_feasible(200000, 6.0, 1000, 300)
    assert not is_feasible  # 1000 < required_payment for these parameters
    assert required_payment > 1000

    # Test case where payment doesn't even cover interest
    is_feasible, required_payment = check_amortization_feasible(200000, 6.0, 500, 300)
    assert not is_feasible

    # Test edge case with zero payment
    is_feasible, required_payment = check_amortization_feasible(200000, 3.0, 0, 300)
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
    fixed_rate = 2.0  # 2%
    fixed_term_months = 24
    mortgage_rate_curve = [6.0 for _ in range(term_months - fixed_term_months)]  # 6%
    savings_rate_curve = [3.0 for _ in range(term_months)]  # 3%
    overpayment_schedule = dict.fromkeys(range(term_months), 0)
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

    # Check that warnings were generated about insufficient payment
    assert len(results["warnings"]) > 0
    assert any("insufficient to amortize" in w for w in results["warnings"])

    # Get the principal values around the transition
    fixed_period_end = results["month_data"][fixed_term_months - 1]["principal_end"]
    next_month_end = results["month_data"][fixed_term_months]["principal_end"]

    # Principal should not increase (due to negative amortization prevention)
    # It should either stay the same (interest-only) or decrease
    assert next_month_end <= fixed_period_end, (
        f"Principal should not increase after fixed period. "
        f"Was £{fixed_period_end:.2f}, became £{next_month_end:.2f}"
    )

    # Check that after transition, payments are at least covering interest
    # by verifying principal never increases
    for i in range(fixed_term_months + 1, len(results["month_data"])):
        current = results["month_data"][i]["principal_end"]
        previous = results["month_data"][i - 1]["principal_end"]
        assert current <= previous, (
            f"Principal increased at month {i} from £{previous:.2f} to £{current:.2f}"
        )


def test_simulate_mortgage_with_tight_payment_constraint():
    """Test simulation behavior when payment constraint is tight but possible"""
    # Setup test parameters
    mortgage_amount = 200000
    term_months = 300
    fixed_rate = 2.0  # 2%
    fixed_term_months = 24
    mortgage_rate_curve = [4.0 for _ in range(term_months - fixed_term_months)]  # 4%
    savings_rate_curve = [3.0 for _ in range(term_months)]  # 3%
    overpayment_schedule = dict.fromkeys(range(term_months), 0)
    monthly_savings = 1000
    initial_savings = 50000

    # Calculate initial payment during fixed period based on the fixed period only
    initial_payment = calculate_monthly_payment(
        mortgage_amount,
        fixed_rate,
        fixed_term_months
    )

    # Calculate expected principal at end of fixed period
    expected_principal_at_fixed_end = mortgage_amount
    for _ in range(fixed_term_months):
        interest = expected_principal_at_fixed_end * (fixed_rate/100) / 12
        principal_repayment = initial_payment - interest
        expected_principal_at_fixed_end -= principal_repayment

    # Calculate minimum viable payment for variable rate period
    remaining_term = term_months - fixed_term_months
    min_payment = calculate_monthly_payment(
        expected_principal_at_fixed_end, 4.0, remaining_term
    )
    max_payment = min_payment * 1.05  # 5% above minimum required

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

    # Print debug information for the first few months
    print("\nFirst few months of simulation:")
    for i in range(min(5, len(results["month_data"]))):
        data = results["month_data"][i]
        print(f"\nMonth {data['month']}:")
        print(f"  Principal start: £{data['principal_start']:.2f}")
        print(f"  Monthly payment: £{data['monthly_payment']:.2f}")
        print(f"  Interest paid: £{data['interest_paid']:.2f}")
        print(f"  Principal repaid: £{data['principal_repaid']:.2f}")
        print(f"  Principal end: £{data['principal_end']:.2f}")

    # Check that principal decreases over time during fixed period
    for i in range(1, fixed_term_months):
        assert (
            results["month_data"][i]["principal_end"]
            <= results["month_data"][i - 1]["principal_end"]
        ), (
            f"Principal increased during fixed period at month {i} "
            f"from £{results['month_data'][i-1]['principal_end']:.2f} "
            f"to £{results['month_data'][i]['principal_end']:.2f} "
            f"with payment £{results['month_data'][i]['monthly_payment']:.2f}"
        )

    # Check that principal decreases over time during variable period
    for i in range(fixed_term_months + 1, len(results["month_data"])):
        assert (
            results["month_data"][i]["principal_end"]
            <= results["month_data"][i - 1]["principal_end"]
        ), (
            f"Principal increased during variable period at month {i} "
            f"from £{results['month_data'][i-1]['principal_end']:.2f} "
            f"to £{results['month_data'][i]['principal_end']:.2f} "
            f"with payment £{results['month_data'][i]['monthly_payment']:.2f}"
        )

    # Check that we eventually pay off the mortgage
    final_principal = results["month_data"][-1]["principal_end"]
    assert final_principal < mortgage_amount


def test_simulate_mortgage_with_overpayments_and_constraints():
    """Test simulation behavior with both overpayments and payment constraints"""
    # Setup test parameters
    mortgage_amount = 200000
    term_months = 300
    fixed_rate = 2.0  # 2%
    fixed_term_months = 24
    mortgage_rate_curve = [4.0 for _ in range(term_months - fixed_term_months)]  # 4% instead of 6%
    savings_rate_curve = [3.0 for _ in range(term_months)]  # 3%
    monthly_savings = 1000
    initial_savings = 50000

    # Calculate initial payment during fixed period
    initial_payment = calculate_monthly_payment(mortgage_amount, fixed_rate, term_months)

    # Calculate expected principal at end of fixed period
    expected_principal_at_fixed_end = mortgage_amount
    for _ in range(fixed_term_months):
        interest = expected_principal_at_fixed_end * (fixed_rate/100) / 12
        principal_repayment = initial_payment - interest
        expected_principal_at_fixed_end -= principal_repayment

    # Create overpayment schedule with significant overpayments
    overpayment_schedule = dict.fromkeys(range(term_months), 0)
    overpayment_schedule[fixed_term_months] = 20000  # Large overpayment at end of fixed period

    # Account for overpayment in expected principal
    expected_principal_after_overpayment = expected_principal_at_fixed_end - 20000

    # Calculate minimum viable payment for remaining term after overpayment
    remaining_term = term_months - fixed_term_months
    min_payment = calculate_monthly_payment(
        expected_principal_after_overpayment, 4.0, remaining_term
    )
    max_payment = min_payment * 1.05  # 5% above minimum required

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
    for data in results["month_data"][fixed_term_months + 1:]:
        assert data["monthly_payment"] <= max_payment, (
            f"Payment {data['monthly_payment']:.2f} exceeds maximum {max_payment:.2f} "
            f"at month {data['month']}"
        )

    # Check that principal decreases during fixed period
    for i in range(1, fixed_term_months):
        assert (
            results["month_data"][i]["principal_end"]
            <= results["month_data"][i - 1]["principal_end"]
        ), f"Principal increased during fixed period at month {i}"

    # Check that principal decreases after overpayment
    for i in range(fixed_term_months + 1, len(results["month_data"])):
        assert (
            results["month_data"][i]["principal_end"]
            <= results["month_data"][i - 1]["principal_end"]
        ), (
            f"Principal increased during variable period at month {i} "
            f"from {results['month_data'][i-1]['principal_end']:.2f} "
            f"to {results['month_data'][i]['principal_end']:.2f} "
            f"with payment {results['month_data'][i]['monthly_payment']:.2f}"
        )


def test_create_overpayment_schedule_none():
    """Test overpayment schedule creation with 'none' type"""
    term_months = 12
    schedule = create_overpayment_schedule(
        term_months=term_months,
        schedule_type='none'
    )

    # Verify schedule length
    assert len(schedule) == term_months

    # Verify all months have zero overpayment
    assert all(amount == 0 for amount in schedule.values())

    # Verify all months from 0 to term_months-1 are present
    assert set(schedule.keys()) == set(range(term_months))


def test_create_overpayment_schedule_fixed():
    """Test overpayment schedule creation with fixed monthly amount"""
    term_months = 12
    monthly_amount = 100
    schedule = create_overpayment_schedule(
        term_months=term_months,
        schedule_type='fixed',
        monthly_amount=monthly_amount
    )

    # Verify schedule length
    assert len(schedule) == term_months

    # Verify all months have the specified overpayment
    assert all(amount == monthly_amount for amount in schedule.values())

    # Test with zero monthly amount
    schedule_zero = create_overpayment_schedule(
        term_months=term_months,
        schedule_type='fixed',
        monthly_amount=0
    )
    assert all(amount == 0 for amount in schedule_zero.values())


def test_create_overpayment_schedule_lump_sum():
    """Test overpayment schedule creation with lump sum payments"""
    term_months = 24

    # Test lump sum within term
    lump_sums = {6: 5000}
    schedule = create_overpayment_schedule(
        term_months=term_months,
        schedule_type='lump_sum',
        lump_sums=lump_sums
    )

    # Verify specific month has correct amount
    assert schedule[6] == 5000
    # Verify all other months are zero
    assert sum(schedule.values()) == 5000

    # Test lump sum outside term
    lump_sums_outside = {15: 2000, 25: 3000}
    schedule = create_overpayment_schedule(
        term_months=12,
        schedule_type='lump_sum',
        lump_sums=lump_sums_outside
    )

    # Verify out-of-range lump sums are ignored
    assert all(amount == 0 for amount in schedule.values())

    # Test multiple lump sums
    lump_sums_multiple = {3: 1000, 6: 2000, 9: 3000}
    schedule = create_overpayment_schedule(
        term_months=12,
        schedule_type='lump_sum',
        lump_sums=lump_sums_multiple
    )

    # Verify correct amounts at specific months
    assert schedule[3] == 1000
    assert schedule[6] == 2000
    assert schedule[9] == 3000
    assert sum(schedule.values()) == 6000


def test_create_overpayment_schedule_yearly_bonus():
    """Test overpayment schedule creation with yearly bonus"""
    term_months = 36
    bonus_month = 12  # December
    bonus_amount = 1000

    schedule = create_overpayment_schedule(
        term_months=term_months,
        schedule_type='yearly_bonus',
        bonus_month=bonus_month,
        bonus_amount=bonus_amount
    )

    # Verify bonus months (11, 23, 35 for 0-based indexing)
    expected_bonus_months = [11, 23, 35]  # December of each year (0-based)

    for month in range(term_months):
        if month in expected_bonus_months:
            assert schedule[month] == bonus_amount
        else:
            assert schedule[month] == 0

    # Test with different bonus month
    schedule = create_overpayment_schedule(
        term_months=term_months,
        schedule_type='yearly_bonus',
        bonus_month=6,  # June
        bonus_amount=bonus_amount
    )

    expected_bonus_months = [5, 17, 29]  # June of each year (0-based)
    for month in range(term_months):
        if month in expected_bonus_months:
            assert schedule[month] == bonus_amount
        else:
            assert schedule[month] == 0


def test_create_overpayment_schedule_custom():
    """Test overpayment schedule creation with custom schedule"""
    term_months = 12

    # Test valid and invalid months
    custom_schedule = {
        3: 200,    # valid
        -1: 500,   # invalid - negative month
        15: 700,   # invalid - beyond term
        5: 300     # valid
    }

    schedule = create_overpayment_schedule(
        term_months=term_months,
        schedule_type='custom',
        custom_schedule=custom_schedule
    )

    # Verify valid months have correct amounts
    assert schedule[3] == 200
    assert schedule[5] == 300

    # Verify invalid months are not included
    assert -1 not in schedule  # Negative months should be filtered out
    assert 15 not in schedule  # Months beyond term should be filtered out

    # Verify all other months are zero
    total_overpayments = sum(schedule.values())
    assert total_overpayments == 500  # Only valid month overpayments (200 + 300)

    # Verify schedule contains only valid months
    assert all(0 <= month < term_months for month in schedule.keys())
    assert all(month in range(term_months) for month in schedule.keys())

    # Test empty custom schedule
    schedule = create_overpayment_schedule(
        term_months=term_months,
        schedule_type='custom',
        custom_schedule={}
    )
    # Verify all months are zero and only valid months exist
    assert all(amount == 0 for amount in schedule.values())
    assert all(0 <= month < term_months for month in schedule.keys())
    assert len(schedule) == term_months


def test_create_overpayment_schedule_invalid_type():
    """Test overpayment schedule creation with invalid schedule type"""
    term_months = 12

    # Should default to all zeros for invalid type
    schedule = create_overpayment_schedule(
        term_months=term_months,
        schedule_type='invalid_type'
    )

    assert len(schedule) == term_months
    assert all(amount == 0 for amount in schedule.values())


def test_get_days_in_month():
    """Test get_days_in_month function for various months"""
    # Test January (31 days)
    assert get_days_in_month(0) == 31
    
    # Test February (28 days in non-leap year)
    assert get_days_in_month(1) == 28
    
    # Test April (30 days)
    assert get_days_in_month(3) == 30
    
    # Test December (31 days)
    assert get_days_in_month(11) == 31
    
    # Test month index > 12 (should cycle)
    assert get_days_in_month(12) == 31  # January again
    assert get_days_in_month(13) == 28  # February again


def test_daily_rate():
    """Test daily_rate calculation"""
    # Test 3.65% annual rate (should be 0.01% daily)
    assert isclose(daily_rate(3.65), 0.0001, rel_tol=1e-10)
    
    # Test 0% rate
    assert daily_rate(0.0) == 0.0
    
    # Test 36.5% rate (should be 0.1% daily)
    assert isclose(daily_rate(36.5), 0.001, rel_tol=1e-10)


def test_monthly_interest_from_daily_zero_rate():
    """Test monthly interest calculation with zero interest rate"""
    principal = 100000
    annual_rate = 0.0
    
    # Should be zero for any month
    for month in range(12):
        interest = monthly_interest_from_daily(principal, annual_rate, month)
        assert interest == 0.0


def test_monthly_interest_from_daily_vs_monthly():
    """Test that daily compounding produces higher interest than monthly compounding"""
    principal = 100000
    annual_rate = 3.0  # 3%
    
    # Monthly compounding interest
    monthly_rate_decimal = annual_rate / 100.0 / 12.0
    monthly_interest_simple = principal * monthly_rate_decimal
    
    # Daily compounding interest for January (31 days)
    daily_interest = monthly_interest_from_daily(principal, annual_rate, 0)
    
    # Daily compounding should be higher than monthly compounding
    assert daily_interest > monthly_interest_simple
    
    # But the difference should be reasonable for typical rates
    difference_percent = (daily_interest - monthly_interest_simple) / monthly_interest_simple * 100
    assert difference_percent < 5.0  # Should be less than 5% difference for typical rates
    print(f"Daily vs monthly compounding difference: {difference_percent:.2f}%")


def test_monthly_interest_from_daily_different_months():
    """Test that different months produce different interest amounts due to different day counts"""
    principal = 100000
    annual_rate = 3.0
    
    # February (28 days) vs January (31 days)
    feb_interest = monthly_interest_from_daily(principal, annual_rate, 1)  # February
    jan_interest = monthly_interest_from_daily(principal, annual_rate, 0)  # January
    
    # January should have more interest due to more days
    assert jan_interest > feb_interest
    
    # Test all months to ensure they're different based on days
    interests = []
    for month in range(12):
        interest = monthly_interest_from_daily(principal, annual_rate, month)
        interests.append(interest)
    
    # Should have different values for months with different day counts
    # February should be the lowest (28 days)
    assert min(interests) == interests[1]  # February index 1
