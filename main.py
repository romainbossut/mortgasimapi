import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, differential_evolution
import itertools
import json
from datetime import datetime
import os
from multiprocessing import Pool, cpu_count
from functools import partial

def calculate_monthly_payment(principal, annual_interest_rate, months):
    """
    Calculate the monthly mortgage payment given:
    principal, annual_interest_rate (decimal), and number of months.
    Uses standard mortgage amortization formula.
    """
    if months <= 0:
        return principal  # If no term left, just return what's due.

    monthly_rate = annual_interest_rate / 12.0
    if monthly_rate == 0:
        # No interest scenario
        return principal / months
    
    # Standard amortization formula: P = r * PV / (1 - (1+r)^(-n))
    payment = monthly_rate * principal / (1 - (1 + monthly_rate)**(-months))
    return payment

def create_overpayment_schedule(term_months, schedule_type, **kwargs):
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
    schedule = {m: 0 for m in range(term_months)}
    
    if schedule_type == 'none':
        return schedule
    
    elif schedule_type == 'fixed':
        monthly_amount = kwargs.get('monthly_amount', 0)
        for m in range(term_months):
            schedule[m] = monthly_amount
            
    elif schedule_type == 'lump_sum':
        lump_sums = kwargs.get('lump_sums', {})
        for month, amount in lump_sums.items():
            if month < term_months:
                schedule[month] = amount
                
    elif schedule_type == 'yearly_bonus':
        bonus_month = kwargs.get('bonus_month', 12)  # Default to December
        bonus_amount = kwargs.get('bonus_amount', 0)
        for year in range(term_months // 12 + 1):
            month = year * 12 + (bonus_month - 1)  # -1 because months are 0-based
            if month < term_months:
                schedule[month] = bonus_amount
                
    elif schedule_type == 'custom':
        custom_schedule = kwargs.get('custom_schedule', {})
        for month, amount in custom_schedule.items():
            if month < term_months:
                schedule[month] = amount
    
    return schedule

def simulate_mortgage(mortgage_amount,
                      term_months,
                      fixed_rate,
                      fixed_term_months,
                      mortgage_rate_curve,
                      savings_rate_curve,
                      overpayment_schedule,
                      monthly_savings_contribution,
                      initial_savings=0.0,
                      typical_payment=0.0,
                      asset_value=0.0,
                      max_payment_after_fixed=float('inf')):
    """
    Simulate a mortgage with a fixed period and then a variable rate according to mortgage_rate_curve.
    If overpayment > 1000 in any given month, recalculate the monthly payment going forward.

    Parameters:
    - mortgage_amount: float, initial principal
    - term_months: int, total mortgage duration in months
    - fixed_rate: float (decimal), annual interest rate for the fixed term
    - fixed_term_months: int, number of months for which the fixed_rate applies
    - mortgage_rate_curve: array-like or callable giving the annual interest rate after the fixed term.
    - savings_rate_curve: array-like or callable giving the annual interest rate for savings each month.
    - overpayment_schedule: dict or array that maps month_index (0-based) to an overpayment amount.
    - monthly_savings_contribution: float, amount contributed to savings each month.
    - initial_savings: float, starting balance in savings account.
    - typical_payment: float, if monthly payment is below this, difference is added to savings.
    - asset_value: float, value of the property (assumed constant).
    - max_payment_after_fixed: float, maximum allowed monthly payment after fixed period.

    Returns:
    dict with simulation results
    """

    # Initialize state variables
    principal = mortgage_amount
    current_month = 0
    total_months = term_months

    # Determine initial monthly payment for the fixed period
    initial_payment = calculate_monthly_payment(principal, fixed_rate, term_months)
    current_monthly_payment = initial_payment

    # Savings account state
    savings_balance = initial_savings

    results = {
        'month_data': []
    }

    # Helper function to get a monthly interest rate from an annual rate
    def monthly_rate(annual_rate):
        return annual_rate / 12.0

    for m in range(total_months):
        # Determine whether we are in fixed period or variable rate period
        if m < fixed_term_months:
            # Within the fixed rate period
            annual_mortgage_rate = fixed_rate
        else:
            # After fixed period, get rate from mortgage_rate_curve
            idx = m - fixed_term_months
            if callable(mortgage_rate_curve):
                annual_mortgage_rate = mortgage_rate_curve(idx)
            else:
                annual_mortgage_rate = mortgage_rate_curve[idx]

            # Recalculate payment at the start of variable rate period or if rate changes
            if m == fixed_term_months or (idx > 0 and mortgage_rate_curve[idx] != mortgage_rate_curve[idx-1]):
                months_remaining = term_months - m
                current_monthly_payment = calculate_monthly_payment(principal, annual_mortgage_rate, months_remaining)
                
                # Apply maximum payment constraint if needed
                if current_monthly_payment > max_payment_after_fixed:
                    current_monthly_payment = max_payment_after_fixed
                    # Extend the term if needed
                    monthly_rate_val = monthly_rate(annual_mortgage_rate)
                    if monthly_rate_val > 0:
                        payment_ratio = principal * monthly_rate_val / max_payment_after_fixed
                        if payment_ratio < 1:
                            new_remaining_months = -math.log(1 - payment_ratio) / math.log(1 + monthly_rate_val)
                            new_remaining_months = math.ceil(new_remaining_months)
                            if m + new_remaining_months > total_months:
                                total_months = m + new_remaining_months
                                print(f"Term extended to {total_months/12:.1f} years to meet payment constraint")

        # Get savings rate
        if callable(savings_rate_curve):
            annual_savings_rate = savings_rate_curve(m)
        else:
            annual_savings_rate = savings_rate_curve[m]

        # Get overpayment for this month
        overpayment = overpayment_schedule.get(m, 0.0)
        
        # Ensure overpayment doesn't exceed available savings
        if overpayment > savings_balance:
            overpayment = savings_balance
        
        # Deduct overpayment from savings
        savings_balance -= overpayment

        # Calculate interest for this month on the mortgage
        monthly_mortgage_rate = monthly_rate(annual_mortgage_rate)
        interest_for_month = principal * monthly_mortgage_rate

        # Total payment this month (not including overpayment yet)
        base_payment = current_monthly_payment

        # If total payment needed is more than principal + interest, we cap it at principal + interest
        total_payment = base_payment + overpayment
        if total_payment > principal + interest_for_month:
            # If paying more than due, just pay exactly what's needed
            total_payment = principal + interest_for_month
            # Return excess overpayment to savings if we capped the total payment
            savings_balance += (base_payment + overpayment) - total_payment

        # Principal portion is what's left after interest
        principal_repaid = total_payment - interest_for_month
        new_principal = principal - principal_repaid

        # Update principal
        principal = max(new_principal, 0.0)

        # Calculate monthly savings rate
        monthly_savings_rate = monthly_rate(annual_savings_rate)

        # Update savings before recording the month:
        # 1. Add monthly savings contribution
        savings_balance += monthly_savings_contribution
        # 2. If monthly payment is less than typical payment, add difference to savings
        if typical_payment > 0 and base_payment < typical_payment:
            payment_difference = typical_payment - base_payment
            savings_balance += payment_difference
        # 3. Apply savings interest
        savings_balance = savings_balance * (1 + monthly_savings_rate)

        # Record month data
        results['month_data'].append({
            'month': m + 1,
            'principal_start': principal + principal_repaid,
            'annual_mortgage_rate': annual_mortgage_rate,
            'monthly_interest_rate': monthly_mortgage_rate,
            'monthly_payment': base_payment,
            'overpayment': overpayment,
            'interest_paid': interest_for_month,
            'principal_repaid': principal_repaid,
            'principal_end': principal,
            'annual_savings_rate': annual_savings_rate,
            'monthly_savings_rate': monthly_savings_rate,
            'savings_balance_end': savings_balance
        })

        # If the mortgage is fully paid off, we can stop early
        if principal <= 0:
            break

        # Recalculate monthly payment if overpayment > 1000
        if overpayment > 1000 and principal > 0:
            # Recalculate monthly payment for the remaining term
            months_left = term_months - (m + 1)
            current_monthly_payment = calculate_monthly_payment(principal, annual_mortgage_rate, months_left)

    return results

def run_simulation_for_schedule(args, schedule):
    """Helper function to run a single simulation with a given schedule."""
    mortgage_amount, term_months, fixed_rate, fixed_term_months, \
    mortgage_rate_curve, savings_rate_curve, monthly_savings_contribution, \
    initial_savings, typical_payment, asset_value = args
    
    results = simulate_mortgage(
        mortgage_amount,
        term_months,
        fixed_rate,
        fixed_term_months,
        mortgage_rate_curve,
        savings_rate_curve,
        schedule,
        monthly_savings_contribution,
        initial_savings,
        typical_payment,
        asset_value
    )
    return results, schedule

def evaluate_schedule(x, args):
    """
    Evaluate a specific overpayment schedule.
    
    Parameters:
    - x: Array of overpayment amounts
    - args: Tuple containing all other parameters needed for simulation
    
    Returns:
    - Negative net worth (for minimization) or infinity if constraints are violated
    """
    (mortgage_amount, term_months, fixed_rate, fixed_term_months,
     mortgage_rate_curve, savings_rate_curve, monthly_savings_contribution,
     initial_savings, typical_payment, asset_value, target_month,
     min_savings_buffer, available_months) = args
    
    # Create schedule from x
    schedule = {m: 0 for m in range(term_months)}
    for i, month in enumerate(available_months):
        if i < len(x):
            schedule[month] = x[i]
    
    # Run simulation
    results = simulate_mortgage(
        mortgage_amount,
        term_months,
        fixed_rate,
        fixed_term_months,
        mortgage_rate_curve,
        savings_rate_curve,
        schedule,
        monthly_savings_contribution,
        initial_savings,
        typical_payment,
        asset_value
    )
    
    # Check minimum savings constraint
    min_savings = min(data['savings_balance_end'] for data in results['month_data'])
    if min_savings < min_savings_buffer:
        return float('inf')  # Return infinity for invalid solutions
    
    # Calculate net worth at target month
    month_data = results['month_data']
    if target_month < len(month_data):
        target_data = month_data[target_month]
    else:
        target_data = month_data[-1]
    
    net_worth = target_data['savings_balance_end'] - target_data['principal_end'] + asset_value
    return -net_worth  # Negative because we're minimizing

def optimize_overpayments(mortgage_amount,
                      term_months,
                      fixed_rate,
                      fixed_term_months,
                      mortgage_rate_curve,
                      savings_rate_curve,
                      monthly_savings_contribution,
                      initial_savings,
                      typical_payment,
                      asset_value,
                      target_year=5,
                      min_savings_buffer=50000,
                      max_overpayment_months=4):
    """
    Optimize overpayment schedule using differential evolution to maximize net worth
    at target_year while maintaining minimum savings.
    """
    target_month = target_year * 12
    
    # Calculate maximum safe overpayment
    max_safe_overpayment = initial_savings - min_savings_buffer
    if max_safe_overpayment <= 0:
        print("Warning: No valid overpayment amounts possible with current savings buffer.")
        return {m: 0 for m in range(term_months)}, 0
    
    # Define available months for overpayments
    # Look at the first 5 years after fixed term, or up to target year + 1 year
    available_months = list(range(
        fixed_term_months,
        min(fixed_term_months + 60, term_months, target_month + 12)
    ))[:max_overpayment_months]  # Limit to max_overpayment_months
    
    if not available_months:
        print("Warning: No valid months available for overpayments.")
        return {m: 0 for m in range(term_months)}, 0
    
    # Prepare arguments for evaluation function
    eval_args = (
        mortgage_amount, term_months, fixed_rate, fixed_term_months,
        mortgage_rate_curve, savings_rate_curve, monthly_savings_contribution,
        initial_savings, typical_payment, asset_value, target_month,
        min_savings_buffer, available_months
    )
    
    # Define bounds for each overpayment
    # Allow smaller overpayments (minimum 5000)
    bounds = [(5000, max_safe_overpayment)] * len(available_months)
    
    # Run differential evolution
    print(f"\nOptimizing overpayments using differential evolution...")
    print(f"Searching over {len(available_months)} months with overpayments between £5,000 and £{int(max_safe_overpayment):,}")
    
    # Progress callback
    iterations = 0
    best_score = float('inf')
    def callback(xk, convergence):
        nonlocal iterations, best_score
        iterations += 1
        current_score = evaluate_schedule(xk, eval_args)
        if current_score < best_score:
            best_score = current_score
            print(f"Iteration {iterations}: Best net worth = £{int(-best_score):,}")
        return False
    
    result = differential_evolution(
        evaluate_schedule,
        bounds,
        args=(eval_args,),
        strategy='best1bin',
        maxiter=100,  # Increased iterations
        popsize=20,   # Increased population size
        mutation=(0.5, 1.5),  # Wider mutation range
        recombination=0.9,    # Increased recombination probability
        updating='deferred',
        workers=-1,
        callback=callback,
        disp=False,
        polish=True  # Enable polishing step
    )
    
    if not result.success:
        print("\nWarning: Optimization may not have converged to the best solution.")
    
    # Create final schedule from best solution
    best_schedule = {m: 0 for m in range(term_months)}
    for i, month in enumerate(available_months):
        amount = result.x[i]
        if amount > 5000:  # Only include overpayments above £5,000
            best_schedule[month] = amount
    
    # Calculate final net worth
    results = simulate_mortgage(
        mortgage_amount,
        term_months,
        fixed_rate,
        fixed_term_months,
        mortgage_rate_curve,
        savings_rate_curve,
        best_schedule,
        monthly_savings_contribution,
        initial_savings,
        typical_payment,
        asset_value
    )
    
    if target_month < len(results['month_data']):
        target_data = results['month_data'][target_month]
    else:
        target_data = results['month_data'][-1]
    
    best_net_worth = target_data['savings_balance_end'] - target_data['principal_end'] + asset_value
    
    return best_schedule, best_net_worth

def save_chart_data_to_json(results, asset_value, monthly_savings_contribution, typical_payment, filename_prefix="chart_data"):
    """
    Save the chart data to a dated JSON file.
    
    Parameters:
    - results: Dictionary containing simulation results
    - asset_value: Value of the property
    - monthly_savings_contribution: Monthly savings amount
    - typical_payment: Typical monthly payment
    - filename_prefix: Prefix for the output filename
    
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
        "years": [data["month"]/12 for data in results["month_data"]],
        "mortgage_balance": [data["principal_end"] for data in results["month_data"]],
        "savings_balance": [data["savings_balance_end"] for data in results["month_data"]],
        "net_worth": [data["savings_balance_end"] - data["principal_end"] + asset_value for data in results["month_data"]],
        "monthly_payments": {
            "principal": [data["principal_repaid"] for data in results["month_data"]],
            "interest": [data["interest_paid"] for data in results["month_data"]]
        },
        "monthly_savings": [monthly_savings_contribution + (max(0, typical_payment - data["monthly_payment"]) if typical_payment > 0 else 0) 
                          for data in results["month_data"]]
    }
    
    # Save to JSON file
    with open(filename, "w") as f:
        json.dump(chart_data, f, indent=4)
    
    return filename

def parse_overpayment_string(overpayment_str, term_months):
    """
    Parse a string of overpayments in the format "month:amount,month:amount".
    
    Parameters:
    - overpayment_str: String in format "18:20000,19:10000"
    - term_months: Total number of months in mortgage term
    
    Returns:
    - Dictionary mapping month numbers to overpayment amounts
    """
    schedule = {m: 0 for m in range(term_months)}
    
    if not overpayment_str:
        return schedule
        
    try:
        pairs = overpayment_str.split(',')
        for pair in pairs:
            month, amount = pair.split(':')
            month = int(month)
            amount = float(amount)
            if month < 0 or month >= term_months:
                raise ValueError(f"Month {month} is outside the mortgage term (0-{term_months-1})")
            if amount < 0:
                raise ValueError(f"Overpayment amount cannot be negative: {amount}")
            schedule[month] = amount
    except ValueError as e:
        print(f"Error parsing overpayment string: {e}")
        print("Expected format: 'month:amount,month:amount' (e.g., '18:20000,19:10000')")
        return {m: 0 for m in range(term_months)}
        
    return schedule

def parse_args():
    """Parse command line arguments."""
    import argparse
    parser = argparse.ArgumentParser(description='Mortgage and Savings Simulator')
    
    # Required arguments
    parser.add_argument('--mortgage-amount', type=float, required=True,
                      help='Initial mortgage amount')
    parser.add_argument('--term-years', type=float, required=True,
                      help='Mortgage term in years')
    parser.add_argument('--fixed-rate', type=float, required=True,
                      help='Fixed interest rate as decimal (e.g., 0.0165 for 1.65%%)')
    parser.add_argument('--fixed-term-months', type=int, required=True,
                      help='Fixed rate term in months')
    
    # Optional arguments with defaults
    parser.add_argument('--variable-rate', type=float, default=0.06,
                      help='Variable interest rate after fixed term (default: 0.06)')
    parser.add_argument('--savings-rate', type=float, default=0.04,
                      help='Annual savings interest rate (default: 0.04)')
    parser.add_argument('--monthly-savings', type=float, default=2500.0,
                      help='Monthly savings contribution (default: 2500.0)')
    parser.add_argument('--initial-savings', type=float, default=150000.0,
                      help='Initial savings balance (default: 150000.0)')
    parser.add_argument('--typical-payment', type=float, default=878.0,
                      help='Typical monthly payment - difference goes to savings if actual payment is lower (default: 878.0)')
    parser.add_argument('--asset-value', type=float, default=360000.0,
                      help='Property value (default: 360000.0)')
    parser.add_argument('--max-payment', type=float, default=float('inf'),
                      help='Maximum monthly payment after fixed period (default: no limit)')
    
    # Overpayment options (mutually exclusive)
    overpayment_group = parser.add_mutually_exclusive_group()
    overpayment_group.add_argument('--optimize', action='store_true',
                      help='Enable optimization of overpayments')
    overpayment_group.add_argument('--overpayments', type=str,
                      help='Manual overpayment schedule in format "month:amount,month:amount" (e.g., "18:20000,19:10000")')
    
    # Optimization parameters
    parser.add_argument('--target-year', type=int, default=5,
                      help='Target year for optimization (default: 5)')
    parser.add_argument('--min-savings', type=float, default=30000.0,
                      help='Minimum savings buffer for optimization (default: 30000.0)')
    parser.add_argument('--max-overpayment-months', type=int, default=3,
                      help='Maximum number of months with overpayments (default: 3)')
    
    return parser.parse_args()

def create_charts(results, asset_value, monthly_savings_contribution, typical_payment):
    """
    Create and display charts from simulation results.
    
    Parameters:
    - results: Dictionary containing simulation results
    - asset_value: Value of the property
    - monthly_savings_contribution: Monthly savings amount
    - typical_payment: Typical monthly payment
    """
    # Create lists for plotting
    years = [data['month']/12 for data in results['month_data']]
    mortgage_balance = [data['principal_end'] for data in results['month_data']]
    savings_balance = [data['savings_balance_end'] for data in results['month_data']]
    net_worth = [s - m + asset_value for s, m in zip(savings_balance, mortgage_balance)]
    
    # Calculate net worth at specific years
    def get_net_worth_at_year(target_year):
        target_month = target_year * 12
        if target_month >= len(results['month_data']):
            return None
        data = results['month_data'][target_month]
        return data['savings_balance_end'] - data['principal_end'] + asset_value
    
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
    
    fig.text(0.02, 0.98, stats_text, fontsize=10, fontfamily='monospace', va='top')
    
    # Create subplot grid
    gs = fig.add_gridspec(4, 1, height_ratios=[0.2, 2, 1, 1])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2], sharex=ax1)
    ax3 = fig.add_subplot(gs[3], sharex=ax1)
    
    # Split monthly payments into components - only regular payment, no overpayments
    interest_payments = [data['interest_paid'] for data in results['month_data']]
    principal_payments = [data['monthly_payment'] - data['interest_paid'] for data in results['month_data']]
    
    # Calculate monthly savings including extra from lower payments
    monthly_savings = []
    for data in results['month_data']:
        base_saving = monthly_savings_contribution
        extra_saving = max(0, typical_payment - data['monthly_payment']) if typical_payment > 0 else 0
        monthly_savings.append(base_saving + extra_saving)

    # Find minimum savings and its timing
    min_savings = min(savings_balance)
    min_savings_month = savings_balance.index(min_savings)
    min_savings_year = years[min_savings_month]

    # Top subplot - Line chart for balances
    ax1.plot(years, mortgage_balance, label='Mortgage Balance', color='red')
    ax1.plot(years, savings_balance, label='Savings Balance', color='green')
    ax1.plot(years, net_worth, label='Net Worth', color='blue', linestyle='--', linewidth=2)
    ax1.set_ylabel('Amount')
    ax1.set_title('Evolution of Mortgage, Savings and Net Worth Over Time')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    # Add minimum savings annotation
    ax1.annotate(f'Minimum Savings: £{int(min_savings):,}\nYear {min_savings_year:.1f}',
                xy=(min_savings_year, min_savings),
                xytext=(min_savings_year + 0.5, min_savings + 20000),
                arrowprops=dict(facecolor='black', shrink=0.05),
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                ha='left')
    
    # Middle subplot - Stacked bar chart for monthly payments (regular payment only)
    width = 0.08
    ax2.bar(years, interest_payments, width, label='Interest', color='red', alpha=0.6)
    ax2.bar(years, principal_payments, width, bottom=interest_payments, label='Principal', color='blue', alpha=0.6)
    ax2.set_ylabel('Monthly Payment')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    # Bottom subplot - Bar chart for monthly savings
    ax3.bar(years, monthly_savings, width=0.08, color='green', alpha=0.6, label='Monthly Savings')
    ax3.set_xlabel('Years')
    ax3.set_ylabel('Monthly Savings')
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.legend()
    
    # Format axes
    def format_pounds(x, p):
        return '£{:,}'.format(int(x))
    
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(format_pounds))
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(format_pounds))
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(format_pounds))
    
    # Set x-axis limits and ticks
    max_year = max(years)
    ax1.set_xlim(-0.5, max_year + 0.5)
    ax1.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax1.xaxis.set_major_formatter(plt.FormatStrFormatter('%d'))
    
    # Adjust layout to prevent text overlapping
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Make room for the stats text
    plt.show()

def print_debug_info(results, fixed_term_months):
    """
    Print debug information about the payment transition period.
    
    Parameters:
    - results: Dictionary containing simulation results
    - fixed_term_months: Number of months in fixed rate period
    """
    print("\nPayment components around fixed-to-variable rate transition:")
    transition_start = max(0, fixed_term_months - 2)
    transition_end = min(len(results['month_data']), fixed_term_months + 3)
    for i in range(transition_start, transition_end):
        data = results['month_data'][i]
        print(f"\nMonth {data['month']}:")
        print(f"  Monthly Payment: £{data['monthly_payment']:.2f}")
        print(f"  Interest Paid: £{data['interest_paid']:.2f}")
        print(f"  Principal Repaid: £{data['principal_repaid']:.2f}")
        print(f"  Annual Rate: {data['annual_mortgage_rate']:.4f}")
        print(f"  Principal Start: £{data['principal_start']:.2f}")
        print(f"  Principal End: £{data['principal_end']:.2f}")

def print_summary(last_month_data):
    """
    Print a summary of the last month's data.
    
    Parameters:
    - last_month_data: Dictionary containing the last month's data
    """
    print("\nLast month data:")
    for k, v in last_month_data.items():
        if isinstance(v, float) and k not in ['monthly_interest_rate', 'annual_mortgage_rate', 'annual_savings_rate', 'monthly_savings_rate']:
            print(f"{k}: £{int(v):,}")
        else:
            print(f"{k}: {v}")

if __name__ == "__main__":
    args = parse_args()
    
    # Convert term years to months
    term_months = int(args.term_years * 12)
    
    # Create rate curves
    mortgage_rate_curve = [args.variable_rate for _ in range(term_months - args.fixed_term_months)]
    savings_rate_curve = [args.savings_rate for _ in range(term_months)]
    
    # Initialize overpayment schedule
    if args.optimize:
        print("\nOptimizing overpayments...")
        optimal_schedule, best_net_worth = optimize_overpayments(
            args.mortgage_amount,
            term_months,
            args.fixed_rate,
            args.fixed_term_months,
            mortgage_rate_curve,
            savings_rate_curve,
            args.monthly_savings,
            args.initial_savings,
            args.typical_payment,
            args.asset_value,
            target_year=args.target_year,
            min_savings_buffer=args.min_savings,
            max_overpayment_months=args.max_overpayment_months
        )
        print(f"\nOptimization Results:")
        print(f"Best Net Worth at year {args.target_year}: £{int(best_net_worth):,}")
        print("\nOptimal Overpayment Schedule:")
        for month, amount in sorted(optimal_schedule.items()):
            if amount > 0:
                print(f"Month {month} (Year {month/12:.1f}): £{int(amount):,}")
        overpayment_schedule = optimal_schedule
    elif args.overpayments:
        overpayment_schedule = parse_overpayment_string(args.overpayments, term_months)
        print("\nManual Overpayment Schedule:")
        for month, amount in sorted(overpayment_schedule.items()):
            if amount > 0:
                print(f"Month {month} (Year {month/12:.1f}): £{int(amount):,}")
    else:
        overpayment_schedule = {m: 0 for m in range(term_months)}

    # Run simulation
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
        args.max_payment
    )

    # Print debug information
    print_debug_info(results, args.fixed_term_months)

    # Save chart data to JSON
    json_file = save_chart_data_to_json(
        results,
        args.asset_value,
        args.monthly_savings,
        args.typical_payment
    )
    print(f"\nChart data saved to: {json_file}")

    # Create and display charts
    create_charts(results, args.asset_value, args.monthly_savings, args.typical_payment)

    # Print summary
    print_summary(results['month_data'][-1])
