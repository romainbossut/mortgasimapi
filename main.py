import sys

# Only use Agg backend during tests
if 'pytest' in sys.modules:
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import json
from datetime import datetime
import os
from typing import Dict, List, Union, Tuple, Optional, Any, Callable
import csv


def calculate_monthly_payment(
    principal: float, annual_interest_rate: float, months: int
) -> float:
    """
    Calculate the monthly mortgage payment for a given principal, annual interest rate, and term in months.

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
) -> Dict[int, float]:
    """
    Create an overpayment schedule based on different patterns.

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
    schedule: Dict[int, float] = {m: 0 for m in range(term_months)}

    if schedule_type == "none":
        return schedule

    elif schedule_type == "fixed":
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
) -> Tuple[bool, float]:
    """
    Check if a mortgage can be amortized within given constraints.

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
    """
    Calculate the expected principal at the end of the fixed rate period.
    fixed_rate is assumed to be an annual percentage (e.g., 2.0 means 2%).
    """
    expected_principal = principal
    # Convert annual percentage rate (e.g. 2.0) to monthly decimal rate (e.g. 0.02 / 12)
    monthly_decimal_rate = (fixed_rate / 100.0) / 12.0

    for _ in range(fixed_term_months):
        interest = expected_principal * monthly_decimal_rate
        principal_repayment = initial_payment - interest
        expected_principal -= principal_repayment

    return expected_principal


def handle_overpayment(
    overpayment: float,
    savings_balance: float,
    month: int
) -> Tuple[float, float, Optional[str]]:
    """
    Handle overpayment logic and constraints.
    
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
) -> Tuple[float, float, Optional[str]]:
    """
    Handle case where total payment exceeds remaining principal + interest.
    
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


def monthly_rate(annual_rate: float) -> float:
    """Convert annual rate in percentage to monthly rate in decimal."""
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
) -> Dict[str, Union[int, float]]:
    """
    Create a dictionary with all the data for a given month.
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
    mortgage_rate_curve: Union[List[float], Callable[[int], float]],
    savings_rate_curve: Union[List[float], Callable[[int], float]]
) -> List[str]:
    """
    Validate rate curves to ensure they have correct lengths and valid values.
    
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
    rate_curve: Union[List[float], Callable[[int], float]],
    month: int,
    default_rate: float,
    curve_name: str
) -> float:
    """
    Safely get rate for a given month from either a list or callable rate curve.
    Returns default_rate if index is out of range.
    """
    try:
        if callable(rate_curve):
            return rate_curve(month)
        else:
            return rate_curve[month]
    except (IndexError, ValueError) as e:
        print(f"Warning: Error getting {curve_name} for month {month}, using {default_rate}%: {str(e)}")
        return default_rate


def ensure_payment_covers_interest(
    current_payment: float,
    principal: float,
    annual_rate: float,
    month: int
) -> Tuple[float, Optional[str]]:
    """
    Ensure monthly payment covers interest to prevent negative amortization.
    
    Returns:
    - Tuple of (adjusted_payment, warning_message)
    """
    if principal <= 0:
        return current_payment, None
        
    monthly_rate = annual_rate / 100.0 / 12.0
    interest_for_month = principal * monthly_rate
    
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
    mortgage_rate_curve: Union[List[float], Callable[[int], float]],
    savings_rate_curve: Union[List[float], Callable[[int], float]],
    overpayment_schedule: Dict[int, float],
    monthly_savings_contribution: float,
    initial_savings: float = 0.0,
    typical_payment: float = 0.0,
    asset_value: float = 0.0,
    max_payment_after_fixed: float = float('inf'),
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Simulate a mortgage with a fixed period and then a variable rate according to mortgage_rate_curve.
    
    Parameters:
    - mortgage_amount: float, initial principal
    - term_months: int, total mortgage duration in months
    - fixed_rate: float, annual interest rate for the fixed term in percentage (e.g., 1.65 for 1.65%)
    - fixed_term_months: int, number of months for which the fixed_rate applies
    - mortgage_rate_curve: array-like or callable giving the annual interest rate after the fixed term in percentage
    - savings_rate_curve: array-like or callable giving the annual interest rate for savings in percentage
    - overpayment_schedule: dict mapping month_index (0-based) to overpayment amount
    - monthly_savings_contribution: float, amount contributed to savings each month
    - initial_savings: float, starting balance in savings account
    - typical_payment: float, if monthly payment is below this, difference is added to savings
    - asset_value: float, value of the property (assumed constant)
    - max_payment_after_fixed: float, maximum allowed monthly payment after fixed period
    
    Returns:
    dict with simulation results including:
    - month_data: list of dicts with monthly data where:
        - month: 1-based month number (1 to term_months)
        - year: decimal year (e.g., 1.5 for month 18)
        - principal_start: principal at start of month
        - principal_end: principal at end of month
        - monthly_payment: base monthly payment
        - overpayment: any overpayment made
        - total_payment: monthly_payment + overpayment
        - interest_paid: interest portion of payment
        - principal_repaid: principal portion of payment
        - savings_balance_end: savings balance at end of month
        - savings_interest: interest earned on savings
        - net_worth: savings - mortgage + asset_value
        - annual_mortgage_rate: mortgage rate in percentage
        - monthly_interest_rate: monthly mortgage rate in decimal
        - annual_savings_rate: savings rate in percentage
        - monthly_savings_rate: monthly savings rate in decimal
    - warnings: list of warning messages
    """
    # Validate inputs
    results = {'month_data': [], 'warnings': []}
    
    # Validate rate curves
    validation_warnings = validate_rate_curves(
        term_months, fixed_term_months, mortgage_rate_curve, savings_rate_curve
    )
    if validation_warnings:
        results['warnings'].extend(validation_warnings)
        if any("must be less than" in w for w in validation_warnings):
            return results  # Cannot proceed if fixed term >= total term

    # Initialize state variables
    principal = mortgage_amount
    total_months = term_months

    # Determine initial monthly payment for the fixed period
    initial_payment = calculate_monthly_payment(principal, fixed_rate, term_months)
    current_monthly_payment = initial_payment

    # Calculate expected principal at end of fixed period
    expected_principal = calculate_fixed_period_principal(
        principal, fixed_rate, fixed_term_months, initial_payment
    )

    # Check if variable rate period is feasible with payment constraint
    if max_payment_after_fixed < float('inf'):
        # Get initial variable rate
        initial_variable_rate = get_rate_for_month(
            mortgage_rate_curve,
            0,  # First month after fixed period
            fixed_rate,
            "mortgage rate"
        )
        
        # Check feasibility
        is_feasible, required_payment = check_amortization_feasible(
            expected_principal,
            initial_variable_rate,
            max_payment_after_fixed,
            term_months - fixed_term_months
        )
        
        if not is_feasible:
            results["warnings"].append(
                f"Maximum payment of £{max_payment_after_fixed:,.2f} is insufficient to "
                f"amortize £{expected_principal:,.2f} over {term_months - fixed_term_months} months "
                f"at {initial_variable_rate:.2f}% (requires £{required_payment:,.2f} monthly)"
            )

    # Savings account state
    savings_balance = initial_savings

    # Main simulation loop
    for m in range(total_months):
        # Determine whether we are in fixed period or variable rate period
        if m < fixed_term_months:
            annual_mortgage_rate = fixed_rate
        else:
            # After fixed period, get rate from mortgage_rate_curve
            idx = m - fixed_term_months
            annual_mortgage_rate = get_rate_for_month(
                mortgage_rate_curve,
                idx,
                fixed_rate,  # Use fixed rate as fallback
                "mortgage rate"
            )

            # Recalculate payment for variable rate period
            if m == fixed_term_months:
                months_remaining = term_months - m
                if verbose:
                    print(f"\nDEBUG: Transition to variable rate at month {m}")
                    print(f"DEBUG: Principal at transition: £{principal:.2f}")
                    print(f"DEBUG: Annual rate changing from {fixed_rate:.2f}% to {annual_mortgage_rate:.2f}%")
                    print(f"DEBUG: Months remaining: {months_remaining}")
                
                current_monthly_payment = calculate_monthly_payment(
                    principal, annual_mortgage_rate, months_remaining
                )
                if verbose:
                    print(f"DEBUG: Calculated payment: £{current_monthly_payment:.2f}")
                
                if max_payment_after_fixed < float('inf'):
                    old_payment = current_monthly_payment
                    current_monthly_payment = min(current_monthly_payment, max_payment_after_fixed)
                    if verbose:
                        print(f"DEBUG: Payment constrained from £{old_payment:.2f} to £{current_monthly_payment:.2f}")

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

        # Calculate mortgage payments
        monthly_mortgage_rate = monthly_rate(annual_mortgage_rate)
        interest_for_month = principal * monthly_mortgage_rate

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

        if verbose and m >= fixed_term_months - 1 and m <= fixed_term_months + 1:
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
                # Estimate next-month interest portion
                next_month_interest = principal * (annual_mortgage_rate / 100.0 / 12.0)
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
    results: Dict[str, Any],
    asset_value: float,
    monthly_savings_contribution: float,
    typical_payment: float,
    filename_prefix: str = "chart_data",
) -> str:
    """
    Save the chart data to a dated JSON file.

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
        for s, m in zip(savings_balance, mortgage_balance) # Use rounded balances
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
    overpayment_str: Optional[str], term_months: int
) -> Dict[int, float]:
    """
    Parse a string of overpayments in the format "month:amount,month:amount".
    Months should be 1-based (1 to term_months).

    Returns:
    - Dictionary mapping 0-based month index (0 to term_months-1) to overpayment amounts
    """
    schedule = {m: 0 for m in range(term_months)}

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


def create_charts(
    results: Dict[str, Any],
    asset_value: float,
    monthly_savings_contribution: float,
    typical_payment: float,
) -> None:
    """
    Create charts from simulation results and save them to file.
    Charts are saved to the data directory with a timestamp.
    """
    # Create lists for plotting
    years = [data['month']/12 for data in results['month_data']]
    mortgage_balance = [data['principal_end'] for data in results['month_data']]
    savings_balance = [data['savings_balance_end'] for data in results['month_data']]
    net_worth = [s - m + asset_value for s, m in zip(savings_balance, mortgage_balance)]

    # Find minimum savings and its timing
    min_savings = min(savings_balance)
    min_savings_month = savings_balance.index(min_savings)
    min_savings_year = years[min_savings_month]

    # Create figure with stats text and subplots sharing x axis
    fig = plt.figure(figsize=(12, 16))  # Made taller to accommodate new subplot

    # Add stats text at the top
    stats_text = "Net Worth Summary:\n"
    savings_2y = savings_balance[24] if len(savings_balance) > 24 else None
    savings_3y = savings_balance[36] if len(savings_balance) > 36 else None
    savings_5y = savings_balance[60] if len(savings_balance) > 60 else None
    savings_10y = savings_balance[120] if len(savings_balance) > 120 else None
    net_worth_3y = net_worth[36] if len(net_worth) > 36 else None
    net_worth_5y = net_worth[60] if len(net_worth) > 60 else None
    net_worth_10y = net_worth[120] if len(net_worth) > 120 else None

    if savings_2y is not None:
        stats_text += f"Year 2:  Savings £{int(savings_2y):,}\n"
    if savings_3y is not None:
        stats_text += f"Year 3:  Savings £{int(savings_3y):,}, Net Worth £{int(net_worth_3y):,}\n"
    if savings_5y is not None:
        stats_text += f"Year 5:  Savings £{int(savings_5y):,}, Net Worth £{int(net_worth_5y):,}\n"
    if savings_10y is not None:
        stats_text += f"Year 10: Savings £{int(savings_10y):,}, Net Worth £{int(net_worth_10y):,}"

    fig.text(0.02, 0.98, stats_text, fontsize=10, fontfamily='monospace', va='top')

    # Create subplot grid with 5 rows (stats text, main plot, and three bar charts)
    gs = fig.add_gridspec(5, 1, height_ratios=[0.2, 2, 1, 1, 1])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2], sharex=ax1)
    ax3 = fig.add_subplot(gs[3], sharex=ax1)
    ax4 = fig.add_subplot(gs[4], sharex=ax1)  # New subplot for interest comparison

    # Top subplot - Line chart for balances
    ax1.plot(years, mortgage_balance, label='Mortgage Balance', color='red')
    ax1.plot(years, savings_balance, label='Savings Balance', color='green')
    ax1.plot(years, net_worth, label='Net Worth', color='blue', linestyle='--', linewidth=2)
    ax1.set_ylabel('Amount')
    ax1.set_title('Evolution of Mortgage, Savings and Net Worth Over Time')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    # Add minimum savings annotation with smart positioning
    # Calculate annotation position to avoid overlaps
    x_offset = 0.5  # Default x offset
    y_offset = 20000  # Default y offset
    
    # Adjust if minimum occurs at start
    if min_savings_month == 0:
        x_offset = 1.0  # Move annotation more to the right
    
    # Adjust if minimum occurs at end
    if min_savings_month == len(years) - 1:
        x_offset = -1.0  # Move annotation to the left
    
    # Calculate y position to avoid overlap with axis
    y_pos = min_savings + y_offset
    if y_pos < ax1.get_ylim()[0]:
        y_pos = min_savings + abs(y_offset)  # Move above if too low
    
    ax1.annotate(
        f'Minimum Savings: £{int(min_savings):,}\nYear {min_savings_year:.1f}',
        xy=(min_savings_year, min_savings),
        xytext=(min_savings_year + x_offset, y_pos),
        arrowprops=dict(facecolor='black', shrink=0.05),
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        ha='left' if x_offset > 0 else 'right'
    )

    # Second subplot - Stacked bar chart for monthly payments (regular payment only)
    width = 0.08
    ax2.bar(years, [data['interest_paid'] for data in results['month_data']], width, label="Interest", color="red", alpha=0.6)
    ax2.bar(
        years,
        [data['monthly_payment'] - data['interest_paid'] for data in results['month_data']],
        width,
        bottom=[data['interest_paid'] for data in results['month_data']],
        label="Principal",
        color="blue",
        alpha=0.6,
    )
    ax2.set_ylabel("Monthly Payment")
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.legend()

    # Third subplot - Bar chart for monthly savings
    ax3.bar(
        years,
        [monthly_savings_contribution + data['payment_difference'] for data in results['month_data']],
        width=0.08,
        color="green",
        alpha=0.6,
        label="Monthly Savings",
    )
    ax3.set_ylabel("Monthly Savings")
    ax3.grid(True, linestyle="--", alpha=0.7)
    ax3.legend()

    # Fourth subplot - Stacked bar chart for interest paid vs received
    interest_paid = [-data['interest_paid'] for data in results['month_data']]  # Make interest paid negative
    interest_received = [data['savings_interest'] for data in results['month_data']]
    
    ax4.bar(years, interest_paid, width, label="Interest Paid", color="red", alpha=0.6)
    ax4.bar(years, interest_received, width, label="Interest Received", color="green", alpha=0.6)
    ax4.set_xlabel("Years")
    ax4.set_ylabel("Monthly Interest")
    ax4.grid(True, linestyle="--", alpha=0.7)
    ax4.legend()

    # Format axes
    def format_pounds(x, p):
        return "£{:,}".format(int(x))

    ax1.yaxis.set_major_formatter(plt.FuncFormatter(format_pounds))
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(format_pounds))
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(format_pounds))
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(format_pounds))

    # Set x-axis limits and ticks
    max_year = max(years)
    ax1.set_xlim(-0.5, max_year + 0.5)
    ax1.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax1.xaxis.set_major_formatter(plt.FormatStrFormatter("%d"))

    # Adjust layout to prevent text overlapping
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Make room for the stats text

    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Save the plot
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data/mortgage_charts_{current_time}.png"
    plt.savefig(filename)
    print(f"Charts saved to: {filename}")
    
    # Display the plot if not running tests
    if 'pytest' not in sys.modules:
        plt.show()
    
    plt.close()


def print_debug_info(results: Dict[str, Any], fixed_term_months: int) -> None:
    """
    Print debug information about the payment transition period.
    """
    print("\nPayment components around fixed-to-variable rate transition:")
    transition_start = max(0, fixed_term_months - 2)
    transition_end = min(len(results["month_data"]), fixed_term_months + 3)
    for i in range(transition_start, transition_end):
        data = results["month_data"][i]
        print(f"\nMonth {data['month']}:")
        print(f"  Monthly Payment: £{data['monthly_payment']:.2f}")
        print(f"  Interest Paid: £{data['interest_paid']:.2f}")
        print(f"  Principal Repaid: £{data['principal_repaid']:.2f}")
        print(f"  Annual Rate: {data['annual_mortgage_rate']:.4f}")
        print(f"  Principal Start: £{data['principal_start']:.2f}")
        print(f"  Principal End: £{data['principal_end']:.2f}")


def print_summary(last_month_data: Dict[str, Union[float, int, str]]) -> None:
    """
    Print a summary of the last month's data.
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
    results: Dict[str, Any],
    asset_value: float,
    filename_prefix: str = "mortgage_data"
) -> str:
    """
    Save the simulation results to a CSV file with one row per period.
    
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
        overpayment_schedule = {m: 0 for m in range(term_months)}

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
        args.max_payment,
        args.verbose
    )

    # Print debug information only if verbose
    if args.verbose:
        print_debug_info(results, args.fixed_term_months)

    # Save results to files
    json_file = save_chart_data_to_json(
        results, args.asset_value, args.monthly_savings, args.typical_payment
    )
    print(f"\nChart data saved to: {json_file}")
    
    csv_file = save_results_to_csv(results, args.asset_value)
    print(f"CSV data saved to: {csv_file}")

    # Create and display charts
    create_charts(results, args.asset_value, args.monthly_savings, args.typical_payment)

    # Print summary
    print_summary(results["month_data"][-1])
