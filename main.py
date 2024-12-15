import math
import matplotlib.pyplot as plt
import numpy as np
import itertools
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
    """
    if months <= 0:
        return principal  # If no term left, just return what's due.

    monthly_rate = annual_interest_rate / 12.0
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
    - rate: Annual interest rate
    - max_payment: Maximum allowed monthly payment
    - term_months: Remaining term in months

    Returns:
    - (bool, float): (Is amortization possible, Required monthly payment)
    """
    if max_payment <= 0:
        return False, float("inf")

    monthly_rate = rate / 12.0
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


def check_variable_rate_feasibility(
    expected_principal: float,
    variable_rate: float,
    max_payment_after_fixed: float,
    term_months: int,
    fixed_term_months: int
) -> Tuple[bool, float, str]:
    """
    Check if the variable rate period is feasible with the given payment constraint.
    
    Returns:
    - Tuple of (is_feasible, required_payment, warning_message)
    """
    is_feasible, required_payment = check_amortization_feasible(
        expected_principal,
        variable_rate,
        max_payment_after_fixed,
        term_months - fixed_term_months
    )
    
    warning_message = ""
    if not is_feasible:
        warning_message = (
            f"Warning: Maximum payment of £{max_payment_after_fixed:.2f} is insufficient to amortize "
            f"£{expected_principal:.2f} over {term_months - fixed_term_months} months at {variable_rate*100:.2f}% "
            f"(requires £{required_payment:.2f} monthly)."
        )
    
    return is_feasible, required_payment, warning_message


def calculate_fixed_period_principal(
    principal: float,
    fixed_rate: float,
    fixed_term_months: int,
    initial_payment: float
) -> float:
    """
    Calculate the expected principal at the end of the fixed rate period.
    """
    expected_principal = principal
    for _ in range(fixed_term_months):
        monthly_rate = fixed_rate / 12.0
        interest = expected_principal * monthly_rate
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
            f"Warning: At month {month}, overpayment reduced from £{overpayment:.2f} "
            f"to £{savings_balance:.2f} due to insufficient savings."
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
            f"Note: At month {month}, £{excess:.2f} of overpayment returned to savings "
            f"as it exceeded remaining principal + interest."
        )
        return adjusted_payment, new_savings_balance, warning_message
    
    return adjusted_payment, savings_balance, None


def update_savings_balance(
    savings_balance: float,
    monthly_savings_contribution: float,
    typical_payment: float,
    base_payment: float,
    monthly_savings_rate: float
) -> float:
    """
    Update savings balance with contributions and interest.
    """
    # 1. Add monthly savings contribution
    balance = savings_balance + monthly_savings_contribution
    
    # 2. If monthly payment is less than typical payment, add difference to savings
    if typical_payment > 0 and base_payment < typical_payment:
        payment_difference = typical_payment - base_payment
        balance += payment_difference
        
    # 3. Apply savings interest
    balance = balance * (1 + monthly_savings_rate)
    
    return balance


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
    max_payment_after_fixed: float = float("inf"),
    min_simulation_months: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Simulate a mortgage with a fixed period and then a variable rate according to mortgage_rate_curve.
    If overpayment > 1000 in any given month, recalculate the monthly payment going forward.
    """
    # Initialize state variables
    principal = mortgage_amount
    total_months = term_months
    mortgage_paid_off = False
    results = {"month_data": [], "warnings": []}

    # Calculate initial payment and check feasibility
    initial_payment = calculate_monthly_payment(principal, fixed_rate, term_months)
    expected_principal = calculate_fixed_period_principal(
        principal, fixed_rate, fixed_term_months, initial_payment
    )

    # Check variable rate feasibility
    if max_payment_after_fixed < float("inf"):
        variable_rate = (
            mortgage_rate_curve[0]
            if not callable(mortgage_rate_curve)
            else mortgage_rate_curve(0)
        )
        _, _, warning = check_variable_rate_feasibility(
            expected_principal,
            variable_rate,
            max_payment_after_fixed,
            term_months,
            fixed_term_months
        )
        if warning:
            results["warnings"].append(warning)

    # Initialize payment and savings state
    current_monthly_payment = initial_payment
    savings_balance = initial_savings

    # Helper function to get a monthly interest rate from an annual rate
    def monthly_rate(annual_rate):
        return annual_rate / 12.0

    for m in range(total_months):
        # Stop if we've reached min_simulation_months and mortgage is paid off
        if mortgage_paid_off and min_simulation_months and m >= min_simulation_months:
            break

        # Get rates for this month
        if m < fixed_term_months:
            annual_mortgage_rate = fixed_rate
        else:
            idx = m - fixed_term_months
            annual_mortgage_rate = (
                mortgage_rate_curve[idx]
                if not callable(mortgage_rate_curve)
                else mortgage_rate_curve(idx)
            )

            # Recalculate payment if needed
            if not mortgage_paid_off and (
                m == fixed_term_months
                or (idx > 0 and mortgage_rate_curve[idx] != mortgage_rate_curve[idx - 1])
            ):
                months_remaining = term_months - m
                current_monthly_payment = calculate_monthly_payment(
                    principal, annual_mortgage_rate, months_remaining
                )
                if current_monthly_payment > max_payment_after_fixed and m >= fixed_term_months:
                    current_monthly_payment = max_payment_after_fixed

        # Get savings rate
        annual_savings_rate = (
            savings_rate_curve[m]
            if not callable(savings_rate_curve)
            else savings_rate_curve(m)
        )

        # Handle overpayment
        overpayment = 0 if mortgage_paid_off else overpayment_schedule.get(m, 0.0)
        overpayment, savings_balance, warning = handle_overpayment(
            overpayment, savings_balance, m
        )
        if warning:
            results["warnings"].append(warning)

        # Calculate mortgage payments
        monthly_mortgage_rate = monthly_rate(annual_mortgage_rate)
        interest_for_month = 0 if mortgage_paid_off else principal * monthly_mortgage_rate
        base_payment = 0 if mortgage_paid_off else current_monthly_payment

        # Handle total payment and any excess
        total_payment = base_payment + overpayment
        if not mortgage_paid_off:
            total_payment, savings_balance, warning = handle_payment_excess(
                total_payment, principal, interest_for_month, savings_balance, m
            )
            if warning:
                results["warnings"].append(warning)

        # Calculate principal portion and update mortgage state
        principal_repaid = total_payment - interest_for_month if not mortgage_paid_off else 0
        new_principal = principal - principal_repaid if not mortgage_paid_off else 0

        if not mortgage_paid_off:
            principal = max(new_principal, 0.0)
            if principal == 0:
                mortgage_paid_off = True
                results["warnings"].append(
                    f"Note: Mortgage fully paid off at month {m} (Year {m/12:.1f})"
                )

        # Update savings
        monthly_savings_rate = monthly_rate(annual_savings_rate)
        savings_balance = update_savings_balance(
            savings_balance,
            monthly_savings_contribution,
            typical_payment,
            base_payment,
            monthly_savings_rate
        )

        # Record month data
        results["month_data"].append(
            record_month_data(
                m, principal, principal_repaid, annual_mortgage_rate,
                monthly_mortgage_rate, base_payment, overpayment,
                interest_for_month, annual_savings_rate,
                monthly_savings_rate, savings_balance
            )
        )

        # Recalculate monthly payment if overpayment > 1000 and mortgage not paid off
        if not mortgage_paid_off and overpayment > 1000 and principal > 0:
            months_left = term_months - (m + 1)
            new_payment = calculate_monthly_payment(
                principal, annual_mortgage_rate, months_left
            )
            if new_payment > max_payment_after_fixed and m >= fixed_term_months:
                current_monthly_payment = max_payment_after_fixed
            else:
                current_monthly_payment = new_payment

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

    # Extract data for charts
    chart_data = {
        "years": [data["month"] / 12 for data in results["month_data"]],
        "mortgage_balance": [data["principal_end"] for data in results["month_data"]],
        "savings_balance": [
            data["savings_balance_end"] for data in results["month_data"]
        ],
        "net_worth": [
            data["savings_balance_end"] - data["principal_end"] + asset_value
            for data in results["month_data"]
        ],
        "monthly_payments": {
            "principal": [data["principal_repaid"] for data in results["month_data"]],
            "interest": [data["interest_paid"] for data in results["month_data"]],
            "regular_payment": [data["monthly_payment"] for data in results["month_data"]],
            "overpayment": [data["overpayment"] for data in results["month_data"]]
        },
        "monthly_savings": [
            monthly_savings_contribution
            + (
                max(0, typical_payment - data["monthly_payment"])
                if typical_payment > 0
                else 0
            )
            for data in results["month_data"]
        ],
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

    Returns:
    - Dictionary mapping month numbers to overpayment amounts
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
            print("Expected format: 'month:amount' where month and amount are numbers")
            continue

        month_str, amount_str = pair.split(":")
        
        # Try to parse month
        try:
            month = int(month_str)
        except ValueError:
            print(f"Error parsing overpayment string: Invalid month '{month_str}'")
            print("Month must be a valid integer")
            continue

        # Try to parse amount
        try:
            amount = float(amount_str)
        except ValueError:
            print(f"Error parsing overpayment string: Invalid amount '{amount_str}'")
            print("Amount must be a valid number")
            continue

        # Validate values
        if month < 0:
            print(f"Error parsing overpayment string: Month {month} cannot be negative")
            continue
        if month >= term_months:
            print(f"Error parsing overpayment string: Month {month} is outside the mortgage term (0-{term_months-1})")
            continue
        if amount < 0:
            print(f"Error parsing overpayment string: Overpayment amount cannot be negative: {amount}")
            continue

        schedule[month] = amount

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
        help="Fixed interest rate as decimal (e.g., 0.0165 for 1.65%%)",
    )
    parser.add_argument(
        "--fixed-term-months", type=int, required=True, help="Fixed rate term in months"
    )

    # Optional arguments with defaults
    parser.add_argument(
        "--variable-rate",
        type=float,
        default=0.06,
        help="Variable interest rate after fixed term (default: 0.06)",
    )
    parser.add_argument(
        "--savings-rate",
        type=float,
        default=0.04,
        help="Annual savings interest rate (default: 0.04)",
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
        default=150000.0,
        help="Initial savings balance (default: 150000.0)",
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
    Create and display charts from simulation results.
    """
    # Create lists for plotting
    years = [data["month"] / 12 for data in results["month_data"]]
    mortgage_balance = [data["principal_end"] for data in results["month_data"]]
    savings_balance = [data["savings_balance_end"] for data in results["month_data"]]
    net_worth = [s - m + asset_value for s, m in zip(savings_balance, mortgage_balance)]

    # Calculate net worth at specific years
    def get_net_worth_at_year(target_year):
        target_month = target_year * 12
        if target_month >= len(results["month_data"]):
            return None
        data = results["month_data"][target_month]
        return data["savings_balance_end"] - data["principal_end"] + asset_value

    net_worth_3y = get_net_worth_at_year(3)
    net_worth_5y = get_net_worth_at_year(5)
    net_worth_10y = get_net_worth_at_year(10)

    # Create figure with stats text and three subplots sharing x axis
    fig = plt.figure(figsize=(12, 13))

    # Add stats text at the top
    stats_text = "Net Worth Summary:\n"
    if net_worth_3y is not None:
        stats_text += f"Year 3:  £{int(net_worth_3y):,}\n"
    if net_worth_5y is not None:
        stats_text += f"Year 5:  £{int(net_worth_5y):,}\n"
    if net_worth_10y is not None:
        stats_text += f"Year 10: £{int(net_worth_10y):,}"

    fig.text(0.02, 0.98, stats_text, fontsize=10, fontfamily="monospace", va="top")

    # Create subplot grid
    gs = fig.add_gridspec(4, 1, height_ratios=[0.2, 2, 1, 1])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2], sharex=ax1)
    ax3 = fig.add_subplot(gs[3], sharex=ax1)

    # Split monthly payments into components - only regular payment, no overpayments
    interest_payments = [data["interest_paid"] for data in results["month_data"]]
    principal_payments = [
        data["monthly_payment"] - data["interest_paid"]
        for data in results["month_data"]
    ]

    # Calculate monthly savings including extra from lower payments
    monthly_savings = []
    for data in results["month_data"]:
        base_saving = monthly_savings_contribution
        extra_saving = (
            max(0, typical_payment - data["monthly_payment"])
            if typical_payment > 0
            else 0
        )
        monthly_savings.append(base_saving + extra_saving)

    # Find minimum savings and its timing
    min_savings = min(savings_balance)
    min_savings_month = savings_balance.index(min_savings)
    min_savings_year = years[min_savings_month]

    # Top subplot - Line chart for balances
    ax1.plot(years, mortgage_balance, label="Mortgage Balance", color="red")
    ax1.plot(years, savings_balance, label="Savings Balance", color="green")
    ax1.plot(
        years, net_worth, label="Net Worth", color="blue", linestyle="--", linewidth=2
    )
    ax1.set_ylabel("Amount")
    ax1.set_title("Evolution of Mortgage, Savings and Net Worth Over Time")
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.legend()

    # Add minimum savings annotation
    ax1.annotate(
        f"Minimum Savings: £{int(min_savings):,}\nYear {min_savings_year:.1f}",
        xy=(min_savings_year, min_savings),
        xytext=(min_savings_year + 0.5, min_savings + 20000),
        arrowprops=dict(facecolor="black", shrink=0.05),
        bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5),
        ha="left",
    )

    # Middle subplot - Stacked bar chart for monthly payments (regular payment only)
    width = 0.08
    ax2.bar(years, interest_payments, width, label="Interest", color="red", alpha=0.6)
    ax2.bar(
        years,
        principal_payments,
        width,
        bottom=interest_payments,
        label="Principal",
        color="blue",
        alpha=0.6,
    )
    ax2.set_ylabel("Monthly Payment")
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.legend()

    # Bottom subplot - Bar chart for monthly savings
    ax3.bar(
        years,
        monthly_savings,
        width=0.08,
        color="green",
        alpha=0.6,
        label="Monthly Savings",
    )
    ax3.set_xlabel("Years")
    ax3.set_ylabel("Monthly Savings")
    ax3.grid(True, linestyle="--", alpha=0.7)
    ax3.legend()

    # Format axes
    def format_pounds(x, p):
        return "£{:,}".format(int(x))

    ax1.yaxis.set_major_formatter(plt.FuncFormatter(format_pounds))
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(format_pounds))
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(format_pounds))

    # Set x-axis limits and ticks
    max_year = max(years)
    ax1.set_xlim(-0.5, max_year + 0.5)
    ax1.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax1.xaxis.set_major_formatter(plt.FormatStrFormatter("%d"))

    # Adjust layout to prevent text overlapping
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Make room for the stats text
    plt.show()


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
        "Net Worth"
    ]
    
    # Write data to CSV
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        
        for data in results["month_data"]:
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
                f"{data['savings_balance_end'] - data['principal_end'] + asset_value:.2f}"
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
                    print(f"Month {month} (Year {month/12:.1f}): £{int(amount):,}")
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
        min_simulation_months=term_months,  # Always simulate full term
    )

    # Print debug information
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
