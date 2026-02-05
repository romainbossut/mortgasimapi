#!/usr/bin/env python3
"""
Mortgage Simulation Core Logic

Copyright (c) 2024 Romain Bossut. All Rights Reserved.
This software is proprietary and confidential. Unauthorized copying, distribution, 
or use of this software is strictly prohibited.
"""

import csv
import json
import os
from collections.abc import Callable
from datetime import datetime
from typing import Any
import calendar


def calculate_monthly_payment(
    principal: float, annual_interest_rate: float, months: int
) -> float:
    """Calculate the monthly mortgage payment for a given principal, annual interest rate, and term in months.

    Uses the standard amortization formula:
      P = (r * PV) / (1 - (1+r)^(-n))
    
    Where:
      P = monthly payment
      r = monthly interest rate (annual_interest_rate/12)
      PV = present value (principal)
      n = number of months
      
    Note: annual_interest_rate should be in percentage (e.g., 1.65 for 1.65%)
    """
    if months <= 0:
        return principal  # If no term left, just return what's due.

    # Convert percentage to decimal
    annual_rate_decimal = annual_interest_rate / 100.0
    monthly_rate = annual_rate_decimal / 12.0

    if monthly_rate == 0:
        # No interest scenario
        return principal / months

    # Standard amortization formula: P = r * PV / (1 - (1+r)^(-n))
    payment = monthly_rate * principal / (1 - (1 + monthly_rate) ** (-months))
    return payment


def create_overpayment_schedule(
    term_months: int, schedule_type: str, **kwargs: Any
) -> dict[int, float]:
    """Create an overpayment schedule based on different patterns.

    Parameters:
    - term_months: total number of months
    - schedule_type: type of schedule ('none', 'fixed', 'lump_sum', 'yearly_bonus', 'custom')
    - kwargs: additional arguments depending on schedule type:
        - fixed: monthly_amount
        - lump_sum: {month: amount, ...}
        - yearly_bonus: bonus_month (1-12), bonus_amount
        - custom: custom_schedule (dict of month: amount)

    Returns:
    - Dictionary mapping month number to overpayment amount
    """
    schedule: dict[int, float] = dict.fromkeys(range(term_months), 0)

    if schedule_type == "none":
        return schedule

    if schedule_type == "fixed":
        monthly_amount = kwargs.get("monthly_amount", 0)
        for m in range(term_months):
            schedule[m] = monthly_amount

    elif schedule_type == "lump_sum":
        lump_sums = kwargs.get("lump_sums", {})
        for month, amount in lump_sums.items():
            if 0 <= month < term_months:  # Ensure month is valid
                schedule[month] = amount

    elif schedule_type == "yearly_bonus":
        bonus_month = kwargs.get("bonus_month", 12)  # Default to December
        bonus_amount = kwargs.get("bonus_amount", 0)
        for year in range(term_months // 12 + 1):
            month = year * 12 + (bonus_month - 1)  # -1 because months are 0-based
            if month < term_months:
                schedule[month] = bonus_amount

    elif schedule_type == "custom":
        custom_schedule = kwargs.get("custom_schedule", {})
        for month, amount in custom_schedule.items():
            if 0 <= month < term_months:  # Only include valid months
                schedule[month] = amount

    return schedule


def check_amortization_feasible(
    principal: float, rate: float, max_payment: float, term_months: int
) -> tuple[bool, float]:
    """Check if a mortgage can be amortized within given constraints.

    Parameters:
    - principal: Remaining principal
    - rate: Annual interest rate in percentage (e.g., 1.6 for 1.6%)
    - max_payment: Maximum allowed monthly payment
    - term_months: Remaining term in months

    Returns:
    - (bool, float): (Is amortization possible, Required monthly payment)
    """
    if max_payment <= 0:
        return False, float("inf")

    # Convert percentage to decimal
    rate_decimal = rate / 100.0
    monthly_rate = rate_decimal / 12.0
    monthly_interest = principal * monthly_rate

    # Calculate required payment for the term
    if monthly_rate == 0:
        required_payment = principal / term_months
    else:
        required_payment = (
            monthly_rate * principal / (1 - (1 + monthly_rate) ** (-term_months))
        )

    # Check if payment can cover interest and amortize within term
    is_feasible = max_payment >= required_payment and max_payment > monthly_interest

    return is_feasible, required_payment


def calculate_fixed_period_principal(
    principal: float,
    fixed_rate: float,
    fixed_term_months: int,
    initial_payment: float
) -> float:
    """Calculate the expected principal at the end of the fixed rate period.
    fixed_rate is assumed to be an annual percentage (e.g., 2.0 means 2%).
    """
    expected_principal = principal

    for month_idx in range(fixed_term_months):
        interest = monthly_interest_from_daily(expected_principal, fixed_rate, month_idx)
        principal_repayment = initial_payment - interest
        expected_principal -= principal_repayment

    return expected_principal


def handle_overpayment(
    overpayment: float,
    savings_balance: float,
    month: int
) -> tuple[float, float, str | None]:
    """Handle overpayment logic and constraints.
    
    Returns:
    - Tuple of (adjusted_overpayment, new_savings_balance, warning_message)
    """
    warning_message = None
    adjusted_overpayment = overpayment

    if overpayment > savings_balance:
        warning_message = (
            f"Month {month}: Overpayment reduced from £{overpayment:,.2f} "
            f"to £{savings_balance:,.2f} due to insufficient savings."
        )
        adjusted_overpayment = savings_balance

    new_savings_balance = savings_balance - adjusted_overpayment
    return adjusted_overpayment, new_savings_balance, warning_message


def handle_payment_excess(
    total_payment: float,
    principal: float,
    interest_for_month: float,
    savings_balance: float,
    month: int
) -> tuple[float, float, str | None]:
    """Handle case where total payment exceeds remaining principal + interest.
    
    Returns:
    - Tuple of (adjusted_payment, new_savings_balance, warning_message)
    """
    warning_message = None
    adjusted_payment = total_payment

    if total_payment > principal + interest_for_month:
        excess = total_payment - (principal + interest_for_month)
        adjusted_payment = principal + interest_for_month
        new_savings_balance = savings_balance + excess
        warning_message = (
            f"Month {month}: £{excess:,.2f} of overpayment returned to savings "
            f"as it exceeded remaining principal + interest."
        )
        return adjusted_payment, new_savings_balance, warning_message

    return adjusted_payment, savings_balance, None


def get_days_in_month(month_index: int) -> int:
    """Get the number of days in a month. 
    
    For simplicity, we assume January for month 0, February for month 1, etc.
    We use a non-leap year (2023) for consistency.
    
    Args:
        month_index: 0-based month index (0-11 for Jan-Dec)
    
    Returns:
        Number of days in the month
    """
    # Convert to 1-based month for calendar module
    month = (month_index % 12) + 1
    # Use 2023 as a consistent non-leap year
    return calendar.monthrange(2023, month)[1]


def daily_rate(annual_rate: float) -> float:
    """Convert annual rate in percentage to daily rate in decimal."""
    return annual_rate / 100.0 / 365.0


def monthly_interest_from_daily(principal: float, annual_rate: float, month_index: int) -> float:
    """Calculate monthly interest using daily compounding.
    
    Args:
        principal: Principal amount
        annual_rate: Annual interest rate in percentage (e.g., 3.0 for 3%)
        month_index: 0-based month index to determine days in month
    
    Returns:
        Interest for the month
    """
    if annual_rate == 0:
        return 0.0
    
    daily_rate_decimal = daily_rate(annual_rate)
    days_in_month = get_days_in_month(month_index)
    
    # Calculate compound interest: P * (1 + r)^n - P
    return principal * ((1 + daily_rate_decimal) ** days_in_month - 1)


def monthly_rate(annual_rate: float) -> float:
    """Convert annual rate in percentage to monthly rate in decimal.
    
    This function is kept for backwards compatibility with savings calculations
    and other parts of the code that don't need daily compounding.
    """
    return annual_rate / 100.0 / 12.0


def record_month_data(
    month: int,
    principal: float,
    principal_repaid: float,
    annual_mortgage_rate: float,
    monthly_mortgage_rate: float,
    base_payment: float,
    overpayment: float,
    interest_for_month: float,
    annual_savings_rate: float,
    monthly_savings_rate: float,
    savings_balance: float
) -> dict[str, int | float]:
    """Create a dictionary with all the data for a given month.
    """
    return {
        "month": month + 1,
        "principal_start": principal + principal_repaid,
        "annual_mortgage_rate": annual_mortgage_rate,
        "monthly_interest_rate": monthly_mortgage_rate,
        "monthly_payment": base_payment,
        "overpayment": overpayment,
        "interest_paid": interest_for_month,
        "principal_repaid": principal_repaid,
        "principal_end": principal,
        "annual_savings_rate": annual_savings_rate,
        "monthly_savings_rate": monthly_savings_rate,
        "savings_balance_end": savings_balance
    }


def validate_rate_curves(
    term_months: int,
    fixed_term_months: int,
    mortgage_rate_curve: list[float] | Callable[[int], float],
    savings_rate_curve: list[float] | Callable[[int], float]
) -> list[str]:
    """Validate rate curves to ensure they have correct lengths and valid values.
    
    Returns:
    - List of warning messages if any issues found
    """
    warnings = []

    # Check if fixed term is longer than total term
    if fixed_term_months >= term_months:
        warnings.append(
            f"Fixed term ({fixed_term_months} months) must be less than total term ({term_months} months)"
        )

    # Check mortgage rate curve
    if not callable(mortgage_rate_curve):
        expected_length = term_months - fixed_term_months
        if len(mortgage_rate_curve) < expected_length:
            warnings.append(
                f"Mortgage rate curve too short: has {len(mortgage_rate_curve)} entries, "
                f"needs {expected_length} for term of {term_months} months with "
                f"{fixed_term_months} months fixed"
            )
        if any(rate < 0 for rate in mortgage_rate_curve):
            warnings.append("Mortgage rate curve contains negative rates")

    # Check savings rate curve
    if not callable(savings_rate_curve):
        if len(savings_rate_curve) < term_months:
            warnings.append(
                f"Savings rate curve too short: has {len(savings_rate_curve)} entries, "
                f"needs {term_months} for term of {term_months} months"
            )
        if any(rate < 0 for rate in savings_rate_curve):
            warnings.append("Savings rate curve contains negative rates")

    return warnings


def get_rate_for_month(
    rate_curve: list[float] | Callable[[int], float],
    month: int,
    default_rate: float,
    curve_name: str
) -> float:
    """Safely get rate for a given month from either a list or callable rate curve.
    Returns default_rate if index is out of range.
    """
    try:
        if callable(rate_curve):
            return rate_curve(month)
        return rate_curve[month]
    except (IndexError, ValueError) as e:
        print(f"Warning: Error getting {curve_name} for month {month}, using {default_rate}%: {str(e)}")
        return default_rate


def ensure_payment_covers_interest(
    current_payment: float,
    principal: float,
    annual_rate: float,
    month: int
) -> tuple[float, str | None]:
    """Ensure monthly payment covers interest to prevent negative amortization.
    
    Returns:
    - Tuple of (adjusted_payment, warning_message)
    """
    if principal <= 0:
        return current_payment, None

    # Use daily compounding for interest calculation
    interest_for_month = monthly_interest_from_daily(principal, annual_rate, month)

    if current_payment < interest_for_month:
        warning = (
            f"Month {month}: Payment of £{current_payment:.2f} cannot cover "
            f"monthly interest of £{interest_for_month:.2f} at {annual_rate:.2f}%. "
            f"Increased to cover interest."
        )
        return interest_for_month, warning

    return current_payment, None


def simulate_mortgage(
    mortgage_amount: float,
    term_months: int,
    fixed_rate: float,
    fixed_term_months: int,
    mortgage_rate_curve: list[float] | Callable[[int], float],
    savings_rate_curve: list[float] | Callable[[int], float],
    overpayment_schedule: dict[int, float],
    monthly_savings_contribution: float,
    initial_savings: float = 0.0,
    typical_payment: float = 0.0,
    asset_value: float = 0.0,
    verbose: bool = False,
    rate_curve: list[float] | None = None,
) -> dict[str, Any]:
    """Simulate a mortgage with rates defined by a full rate curve or fixed+variable periods.

    Parameters:
    - mortgage_amount: float, initial principal
    - term_months: int, total mortgage duration in months
    - fixed_rate: float, annual interest rate for the fixed term in percentage (legacy)
    - fixed_term_months: int, number of months for which the fixed_rate applies (legacy)
    - mortgage_rate_curve: array-like or callable giving the annual interest rate after the fixed term in percentage (legacy)
    - savings_rate_curve: array-like or callable giving the annual interest rate for savings in percentage
    - overpayment_schedule: dict mapping month_index (0-based) to overpayment amount
    - monthly_savings_contribution: float, amount contributed to savings each month
    - initial_savings: float, starting balance in savings account
    - typical_payment: float, if monthly payment is below this, difference is added to savings
    - asset_value: float, value of the property (assumed constant)
    - rate_curve: optional full rate curve (one rate per month for the entire term).
      If provided, fixed_rate/fixed_term_months/mortgage_rate_curve are ignored.

    Returns:
    dict with simulation results including month_data, warnings, etc.
    """
    # Validate inputs
    results = {'month_data': [], 'warnings': []}

    # Build full rate curve
    if rate_curve is not None:
        # New multi-deal path: rate_curve covers entire term
        full_rate_curve = list(rate_curve)
        if len(full_rate_curve) < term_months:
            # Pad with the last rate if too short
            last_rate = full_rate_curve[-1] if full_rate_curve else 0.0
            full_rate_curve.extend([last_rate] * (term_months - len(full_rate_curve)))
    else:
        # Legacy path: build from fixed_rate + mortgage_rate_curve
        # Validate rate curves
        validation_warnings = validate_rate_curves(
            term_months, fixed_term_months, mortgage_rate_curve, savings_rate_curve
        )
        if validation_warnings:
            results['warnings'].extend(validation_warnings)
            if any("must be less than" in w for w in validation_warnings):
                return results

        full_rate_curve = [fixed_rate] * fixed_term_months
        for i in range(term_months - fixed_term_months):
            rate = get_rate_for_month(mortgage_rate_curve, i, fixed_rate, "mortgage rate")
            full_rate_curve.append(rate)

    # Initialize state variables
    principal = mortgage_amount
    total_months = term_months

    # Determine initial monthly payment
    initial_rate = full_rate_curve[0] if full_rate_curve else 0.0
    current_monthly_payment = calculate_monthly_payment(principal, initial_rate, term_months)

    # Savings account state
    savings_balance = initial_savings

    # Main simulation loop
    for m in range(total_months):
        annual_mortgage_rate = full_rate_curve[m]

        # Recalculate payment when rate changes
        if m > 0 and full_rate_curve[m] != full_rate_curve[m - 1]:
            months_remaining = term_months - m
            if verbose:
                print(f"\nDEBUG: Rate change at month {m}")
                print(f"DEBUG: Principal: £{principal:.2f}")
                print(f"DEBUG: Rate changing from {full_rate_curve[m-1]:.2f}% to {annual_mortgage_rate:.2f}%")
                print(f"DEBUG: Months remaining: {months_remaining}")

            current_monthly_payment = calculate_monthly_payment(
                principal, annual_mortgage_rate, months_remaining
            )
            if verbose:
                print(f"DEBUG: New payment: £{current_monthly_payment:.2f}")

        # Ensure payment covers interest
        current_monthly_payment, warning = ensure_payment_covers_interest(
            current_monthly_payment, principal, annual_mortgage_rate, m
        )
        if warning:
            results["warnings"].append(warning)

        # Get savings rate safely
        annual_savings_rate = get_rate_for_month(
            savings_rate_curve,
            m,
            0.0,  # Use 0% as fallback for savings
            "savings rate"
        )
        monthly_savings_rate = monthly_rate(annual_savings_rate)

        # Handle overpayment
        overpayment = overpayment_schedule.get(m, 0.0)
        overpayment, savings_balance, warning = handle_overpayment(
            overpayment, savings_balance, m
        )
        if warning:
            results["warnings"].append(warning)

        # Calculate mortgage payments using daily compounding  
        monthly_mortgage_rate = monthly_rate(annual_mortgage_rate)  # Keep for compatibility with data recording
        interest_for_month = monthly_interest_from_daily(principal, annual_mortgage_rate, m)

        base_payment = current_monthly_payment if principal > 0 else 0
        # Calculate the difference, allowing it to be negative
        payment_difference = (typical_payment - base_payment) if typical_payment > 0 else 0

        # Handle total payment and any excess
        total_payment = base_payment + overpayment
        total_payment, savings_balance, warning = handle_payment_excess(
            total_payment, principal, interest_for_month, savings_balance, m
        )
        if warning:
            results["warnings"].append(warning)

        # Calculate principal portion and update mortgage state
        principal_repaid = total_payment - interest_for_month
        new_principal = principal - principal_repaid

        if verbose and (m == 0 or full_rate_curve[m] != full_rate_curve[m - 1] if m > 0 else False):
            print(f"\nDEBUG: Month {m} payment breakdown:")
            print(f"  Current payment: £{current_monthly_payment:.2f}")
            print(f"  Base payment: £{base_payment:.2f}")
            print(f"  Interest: £{interest_for_month:.2f}")
            print(f"  Principal repaid: £{principal_repaid:.2f}")
            print(f"  Principal: £{principal:.2f} -> £{new_principal:.2f}")

        principal = max(new_principal, 0.0)
        # Warning moved down to after data recording

        # Update savings balance with contributions first
        savings_balance = (
            savings_balance
            + monthly_savings_contribution
            + payment_difference  # Add the payment difference to savings
        )

        # Calculate savings interest on the updated balance
        savings_interest = savings_balance * monthly_savings_rate

        # Add the calculated interest to the final end-of-month balance
        savings_balance_end = savings_balance + savings_interest

        # Record month data
        month_data = {
            "month": m + 1,  # 1-based month indexing
            "year": (m + 1) / 12,  # Year as decimal (e.g., 1.5 for month 18)
            "principal_start": principal + principal_repaid,
            "principal_end": principal,
            "monthly_payment": base_payment,
            "overpayment": overpayment,
            "total_payment": total_payment,
            "interest_paid": interest_for_month,
            "principal_repaid": principal_repaid,
            "savings_balance_end": savings_balance_end, # Use final balance including interest
            "savings_interest": savings_interest,   # Use calculated interest
            "net_worth": savings_balance_end - principal + asset_value, # Use final balance
            "annual_mortgage_rate": annual_mortgage_rate,
            "monthly_interest_rate": monthly_mortgage_rate,
            "annual_savings_rate": annual_savings_rate,
            "monthly_savings_rate": monthly_savings_rate,
            "payment_difference": payment_difference  # Add this to track the difference
        }
        results["month_data"].append(month_data)

        # Update savings balance for the next iteration (it now includes interest)
        savings_balance = savings_balance_end

        # Note when mortgage is paid off but continue simulation
        if principal == 0 and not results.get("mortgage_paid_off_month"):
            results["mortgage_paid_off_month"] = m + 1  # 1-based month indexing
            results["warnings"].append(
                f"Note: Mortgage fully paid off at month {m + 1} (Year {(m + 1)/12:.1f})"
            )

        # Recalculate monthly payment if overpayment > 1000 and mortgage not paid off
        if overpayment > 1000 and principal > 0:
            months_left = term_months - (m + 1)
            new_payment = calculate_monthly_payment(
                principal, annual_mortgage_rate, months_left
            )
            if new_payment < current_monthly_payment:
                # Estimate next-month interest portion using daily compounding
                next_month_interest = monthly_interest_from_daily(principal, annual_mortgage_rate, m + 1)
                # Principal portion is new_payment minus interest
                next_month_principal = new_payment - next_month_interest

                current_monthly_payment = new_payment
                results["warnings"].append(
                    f"Info: Reduced monthly payment to £{new_payment:.2f} at month {m} "
                    f"(interest: £{next_month_interest:.2f}, principal: £{next_month_principal:.2f}) "
                    f"due to overpayment."
                )

        # Savings balance update moved up before month_data recording

    # Find minimum savings balance and month
    min_savings_balance = float('inf')
    min_savings_month = -1
    for i, data in enumerate(results["month_data"]):
        if data["savings_balance_end"] < min_savings_balance:
            min_savings_balance = data["savings_balance_end"]
            min_savings_month = data["month"] # 1-based month

    results["min_savings_balance"] = min_savings_balance
    results["min_savings_month"] = min_savings_month

    # Add final summary to warnings if mortgage was paid off
    if results.get("mortgage_paid_off_month"):
        paid_off_month = results["mortgage_paid_off_month"]
        final_savings = results["month_data"][-1]["savings_balance_end"]
        final_net_worth = results["month_data"][-1]["net_worth"]
        results["warnings"].append(
            f"\nFinal Summary:\n"
            f"Mortgage paid off at month {paid_off_month} (Year {paid_off_month/12:.1f})\n"
            f"Final savings balance: £{final_savings:,.2f}\n"
            f"Final net worth: £{final_net_worth:,.2f}"
        )
    elif principal > 0:
        results["warnings"].append(
            f"\nWarning: Mortgage not fully paid off. Remaining balance: £{principal:,.2f}"
        )

    return results


def save_chart_data_to_json(
    results: dict[str, Any],
    asset_value: float,
    monthly_savings_contribution: float,
    typical_payment: float,
    filename_prefix: str = "chart_data",
) -> str:
    """Save the chart data to a dated JSON file.

    Returns:
    - Path to the saved file
    """
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)

    # Format current date and time
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data/{filename_prefix}_{current_time}.json"

    # Extract data for charts and round financial values to 2 decimal places
    years = [data["month"] / 12 for data in results["month_data"]]
    mortgage_balance = [round(data["principal_end"], 2) for data in results["month_data"]]
    savings_balance = [round(data["savings_balance_end"], 2) for data in results["month_data"]]
    net_worth = [
        round(s - m + asset_value, 2)
        for s, m in zip(savings_balance, mortgage_balance, strict=False) # Use rounded balances
    ]
    monthly_payments = {
        "principal": [round(data["principal_repaid"], 2) for data in results["month_data"]],
        "interest": [round(data["interest_paid"], 2) for data in results["month_data"]],
        "regular_payment": [round(data["monthly_payment"], 2) for data in results["month_data"]],
        "overpayment": [round(data["overpayment"], 2) for data in results["month_data"]]
    }
    monthly_savings = [
        round(monthly_savings_contribution + data["payment_difference"], 2)
        for data in results["month_data"]
    ]

    chart_data = {
        "years": years,
        "mortgage_balance": mortgage_balance,
        "savings_balance": savings_balance,
        "net_worth": net_worth,
        "monthly_payments": monthly_payments,
        "monthly_savings": monthly_savings,
        "payment_difference": [round(data["payment_difference"], 2) for data in results["month_data"]]
    }

    # Save to JSON file
    with open(filename, "w") as f:
        json.dump(chart_data, f, indent=4)

    return filename


def parse_overpayment_string(
    overpayment_str: str | None, term_months: int
) -> dict[int, float]:
    """Parse a string of overpayments in the format "month:amount,month:amount".
    Months should be 1-based (1 to term_months).

    Returns:
    - Dictionary mapping 0-based month index (0 to term_months-1) to overpayment amounts
    """
    schedule = dict.fromkeys(range(term_months), 0)

    if not overpayment_str:
        return schedule

    # First validate the basic format
    if ":" not in overpayment_str:
        print("Error parsing overpayment string: Invalid format")
        print("Expected format: 'month:amount,month:amount' (e.g., '18:20000,19:10000')")
        return schedule

    # Split into pairs and process each one
    pairs = overpayment_str.split(",")
    for pair in pairs:
        if ":" not in pair:
            print(f"Error parsing overpayment string: Invalid pair '{pair}'")
            print("Expected format: 'month:amount' where month and amount are numbers (month is 1-based)")
            continue

        month_str, amount_str = pair.split(":")

        # Try to parse month (expecting 1-based)
        try:
            month_one_based = int(month_str)
        except ValueError:
            print(f"Error parsing overpayment string: Invalid month '{month_str}'")
            print("Month must be a valid integer (1-based)")
            continue

        # Try to parse amount
        try:
            amount = float(amount_str)
        except ValueError:
            print(f"Error parsing overpayment string: Invalid amount '{amount_str}'")
            print("Amount must be a valid number")
            continue

        # Validate values (using 1-based month)
        if month_one_based <= 0:
            print(f"Error parsing overpayment string: Month {month_one_based} must be positive (1-based)")
            continue
        if month_one_based > term_months:
            print(f"Error parsing overpayment string: Month {month_one_based} is outside the mortgage term (1-{term_months})")
            continue
        if amount < 0:
            print(f"Error parsing overpayment string: Overpayment amount cannot be negative: {amount}")
            continue

        # Store using 0-based index
        month_zero_based = month_one_based - 1
        schedule[month_zero_based] = amount

    return schedule


def validate_args(args):
    """Validate command line arguments"""
    if args.mortgage_amount <= 0:
        raise ValueError("Mortgage amount cannot be negative or zero")

    if args.term_years <= 0:
        raise ValueError("Term years cannot be negative or zero")

    if args.fixed_rate < 0:
        raise ValueError("Fixed interest rate cannot be negative")

    if args.fixed_term_months <= 0:
        raise ValueError("Fixed term months cannot be negative or zero")

    if args.variable_rate < 0:
        raise ValueError("Variable interest rate cannot be negative")

    if args.savings_rate < 0:
        raise ValueError("Savings interest rate cannot be negative")

    if args.monthly_savings < 0:
        raise ValueError("Monthly savings contribution cannot be negative")

    if args.initial_savings < 0:
        raise ValueError("Initial savings cannot be negative")

    if args.typical_payment < 0:
        raise ValueError("Typical payment cannot be negative")

    if args.asset_value < 0:
        raise ValueError("Asset value cannot be negative")


def parse_args() -> Any:
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(description="Mortgage and Savings Simulator")

    # Required arguments
    parser.add_argument(
        "--mortgage-amount", type=float, required=True, help="Initial mortgage amount"
    )
    parser.add_argument(
        "--term-years", type=float, required=True, help="Mortgage term in years"
    )
    parser.add_argument(
        "--fixed-rate",
        type=float,
        required=True,
        help="Fixed interest rate in percentage (e.g., 3.0 for 3%%)",
    )
    parser.add_argument(
        "--fixed-term-months", type=int, required=True, help="Fixed rate term in months"
    )

    # Optional arguments with defaults
    parser.add_argument(
        "--variable-rate",
        type=float,
        default=6.0,
        help="Variable interest rate after fixed term in percentage (default: 6.0)",
    )
    parser.add_argument(
        "--savings-rate",
        type=float,
        default=4.0,
        help="Annual savings interest rate in percentage (default: 4.0)",
    )
    parser.add_argument(
        "--monthly-savings",
        type=float,
        default=2500.0,
        help="Monthly savings contribution (default: 2500.0)",
    )
    parser.add_argument(
        "--initial-savings",
        type=float,
        default=170000.0,
        help="Initial savings balance (default: 170000.0)",
    )
    parser.add_argument(
        "--typical-payment",
        type=float,
        default=878.0,
        help="Typical monthly payment - difference goes to savings if actual payment is lower (default: 878.0)",
    )
    parser.add_argument(
        "--asset-value",
        type=float,
        default=360000.0,
        help="Property value (default: 360000.0)",
    )
    parser.add_argument(
        "--max-payment",
        type=float,
        default=float("inf"),
        help="Maximum monthly payment after fixed period (default: no limit)",
    )

    # Overpayment options
    parser.add_argument(
        "--overpayments",
        type=str,
        help='Manual overpayment schedule in format "month:amount,month:amount" (e.g., "18:20000,19:10000")',
    )

    # Control verbosity
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output including debug information."
    )

    args = parser.parse_args()
    validate_args(args)
    return args


def print_debug_info(results: dict[str, Any], fixed_term_months: int = 0) -> None:
    """Print debug information about rate transition periods.
    """
    print("\nPayment components around rate transitions:")
    month_data = results["month_data"]
    for i, data in enumerate(month_data):
        # Print around rate changes
        is_transition = (
            i == 0
            or (i > 0 and data['annual_mortgage_rate'] != month_data[i-1]['annual_mortgage_rate'])
        )

        if is_transition:
            start = max(0, i - 1)
            end = min(len(month_data), i + 3)
            for j in range(start, end):
                d = month_data[j]
                print(f"\nMonth {d['month']}:")
                print(f"  Monthly Payment: £{d['monthly_payment']:.2f}")
                print(f"  Interest Paid: £{d['interest_paid']:.2f}")
                print(f"  Principal Repaid: £{d['principal_repaid']:.2f}")
                print(f"  Annual Rate: {d['annual_mortgage_rate']:.4f}")
                print(f"  Principal Start: £{d['principal_start']:.2f}")
                print(f"  Principal End: £{d['principal_end']:.2f}")


def print_summary(last_month_data: dict[str, float | int | str]) -> None:
    """Print a summary of the last month's data.
    """
    print("\nLast month data:")
    for k, v in last_month_data.items():
        if isinstance(v, float) and k not in [
            "monthly_interest_rate",
            "annual_mortgage_rate",
            "annual_savings_rate",
            "monthly_savings_rate",
        ]:
            print(f"{k}: £{int(v):,}")
        else:
            print(f"{k}: {v}")


def save_results_to_csv(
    results: dict[str, Any],
    asset_value: float,
    filename_prefix: str = "mortgage_data"
) -> str:
    """Save the simulation results to a CSV file with one row per period.
    
    Returns:
    - Path to the saved file
    """
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)

    # Format current date and time
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data/{filename_prefix}_{current_time}.csv"

    # Define CSV headers
    headers = [
        "Month",
        "Year",
        "Principal Start",
        "Principal End",
        "Monthly Payment",
        "Interest Paid",
        "Principal Repaid",
        "Overpayment",
        "Annual Mortgage Rate",
        "Monthly Interest Rate",
        "Savings Balance",
        "Annual Savings Rate",
        "Monthly Savings Rate",
        "Net Worth",
        "Payment Difference"
    ]

    # Write data to CSV
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        for data in results["month_data"]:
            net_worth = data['savings_balance_end'] - data['principal_end'] + asset_value
            writer.writerow([
                data["month"],
                f"{data['month']/12:.2f}",
                f"{data['principal_start']:.2f}",
                f"{data['principal_end']:.2f}",
                f"{data['monthly_payment']:.2f}",
                f"{data['interest_paid']:.2f}",
                f"{data['principal_repaid']:.2f}",
                f"{data['overpayment']:.2f}",
                f"{data['annual_mortgage_rate']:.4f}",
                f"{data['monthly_interest_rate']:.6f}",
                f"{data['savings_balance_end']:.2f}",
                f"{data['annual_savings_rate']:.4f}",
                f"{data['monthly_savings_rate']:.6f}",
                f"{net_worth:.2f}",
                f"{data['payment_difference']:.2f}"
            ])

    return filename


if __name__ == "__main__":
    args = parse_args()

    # Convert term years to months
    term_months = int(args.term_years * 12)

    # Create rate curves
    mortgage_rate_curve = [
        args.variable_rate for _ in range(term_months - args.fixed_term_months)
    ]
    savings_rate_curve = [args.savings_rate for _ in range(term_months)]

    # Initialize overpayment schedule
    if args.overpayments:
        overpayment_schedule = parse_overpayment_string(args.overpayments, term_months)
        if any(amount > 0 for amount in overpayment_schedule.values()):
            print("\nManual Overpayment Schedule:")
            for month, amount in sorted(overpayment_schedule.items()):
                if amount > 0:
                    # Print 1-based month for user clarity
                    print(f"Month {month + 1} (Year {(month + 1)/12:.1f}): £{int(amount):,}")
    else:
        overpayment_schedule = dict.fromkeys(range(term_months), 0)

    # Run simulation for the full term
    results = simulate_mortgage(
        args.mortgage_amount,
        term_months,
        args.fixed_rate,
        args.fixed_term_months,
        mortgage_rate_curve,
        savings_rate_curve,
        overpayment_schedule,
        args.monthly_savings,
        args.initial_savings,
        args.typical_payment,
        args.asset_value,
        args.verbose,
    )

    # Print debug information only if verbose
    if args.verbose:
        print_debug_info(results)

    # Save results to files
    csv_file = save_results_to_csv(results, args.asset_value)
    print(f"CSV data saved to: {csv_file}")

    # Print summary
    print_summary(results["month_data"][-1])
