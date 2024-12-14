import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import itertools

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

            # Check if payment would exceed max after fixed period
            if current_monthly_payment > max_payment_after_fixed:
                # Extend the term to meet the payment constraint
                remaining_principal = principal
                remaining_months = term_months - m
                monthly_rate_val = monthly_rate(annual_mortgage_rate)
                
                # Calculate new payment at max allowed amount
                if monthly_rate_val > 0:
                    # Using the standard mortgage formula backwards to find new term
                    # P = PMT * (1 - (1+r)^-n) / r
                    # Solving for n: n = -log(1 - P*r/PMT) / log(1+r)
                    payment_ratio = remaining_principal * monthly_rate_val / max_payment_after_fixed
                    if payment_ratio < 1:  # Only if it's mathematically possible
                        new_remaining_months = -math.log(1 - payment_ratio) / math.log(1 + monthly_rate_val)
                        new_remaining_months = math.ceil(new_remaining_months)
                        # Extend total months if needed
                        if m + new_remaining_months > total_months:
                            total_months = m + new_remaining_months
                            print(f"Term extended to {total_months/12:.1f} years to meet payment constraint")
                
                current_monthly_payment = max_payment_after_fixed

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
            # Use the most recent annual_mortgage_rate as the baseline for recalculation
            current_monthly_payment = calculate_monthly_payment(principal, annual_mortgage_rate, months_left)

    return results

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
    Optimize overpayment schedule to maximize net worth at target_year while maintaining minimum savings.
    
    Parameters:
    - target_year: year at which to optimize net worth
    - min_savings_buffer: minimum amount to keep in savings
    - max_overpayment_months: maximum number of months where overpayments can be made
    
    Returns:
    - Tuple of (optimal_schedule, best_net_worth)
    """
    target_month = target_year * 12
    best_net_worth = float('-inf')
    best_schedule = None
    
    # Try different combinations of months for overpayments
    # Start after fixed period
    available_months = list(range(fixed_term_months, min(term_months, target_month + 12)))
    
    # Try different amounts in increments
    possible_amounts = [20000, 40000, 60000, 80000, 100000]
    
    # Try different numbers of overpayment months
    for num_months in range(1, max_overpayment_months + 1):
        for months_combination in itertools.combinations(available_months, num_months):
            for amounts in itertools.product(possible_amounts, repeat=num_months):
                # Create schedule
                schedule = {m: 0 for m in range(term_months)}
                for month, amount in zip(months_combination, amounts):
                    schedule[month] = amount
                
                # Run simulation with this schedule
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
                
                # Check if this schedule maintains minimum savings
                min_savings = min(data['savings_balance_end'] for data in results['month_data'])
                if min_savings < min_savings_buffer:
                    continue
                
                # Calculate net worth at target year or last available month if mortgage paid off early
                month_data = results['month_data']
                if target_month < len(month_data):
                    target_data = month_data[target_month]
                else:
                    # If mortgage is paid off early, use the last available month
                    target_data = month_data[-1]
                
                net_worth = target_data['savings_balance_end'] - target_data['principal_end'] + asset_value
                
                if net_worth > best_net_worth:
                    best_net_worth = net_worth
                    best_schedule = schedule.copy()
    
    if best_schedule is None:
        print("Warning: No valid schedule found that meets the constraints. Try adjusting the parameters.")
        # Return a schedule with no overpayments as fallback
        return {m: 0 for m in range(term_months)}, 0
    
    return best_schedule, best_net_worth

# Example usage:
if __name__ == "__main__":
    # Define inputs
    mortgage_amount = 196000.0
    term_months = 22*12  # 22 years
    fixed_rate = 0.0165  # 1.65% per annum fixed
    fixed_term_months = 18  # 1.5 years fixed
    asset_value = 360000.0  # Property value
    max_payment_after_fixed = 500.0  # Maximum monthly payment after fixed period

    # Suppose after fixed term, rates vary slightly
    mortgage_rate_curve = [0.06 for _ in range(term_months - fixed_term_months)]  # 4% thereafter

    # Suppose savings rate is constant at 4%
    savings_rate_curve = [0.04 for _ in range(term_months)]

    # Monthly savings contribution and initial savings
    monthly_savings_contribution = 2500.0
    initial_savings = 150000.0  # Starting with £45,000 in savings
    typical_payment = 878.0   # If monthly payment drops below £1,000, difference goes to savings

    # Run optimization
    target_year = 5  # Optimize for net worth after 5 years
    optimal_schedule, best_net_worth = optimize_overpayments(
        mortgage_amount,
        term_months,
        fixed_rate,
        fixed_term_months,
        mortgage_rate_curve,
        savings_rate_curve,
        monthly_savings_contribution,
        initial_savings,
        typical_payment,
        asset_value,
        target_year=target_year,
        min_savings_buffer=30000,  # Keep at least £30k in savings
        max_overpayment_months=3   # Allow up to 3 overpayment months
    )

    print(f"\nOptimization Results:")
    print(f"Best Net Worth at year {target_year}: £{int(best_net_worth):,}")
    print("\nOptimal Overpayment Schedule:")
    for month, amount in sorted(optimal_schedule.items()):
        if amount > 0:
            print(f"Month {month} (Year {month/12:.1f}): £{int(amount):,}")

    # Run simulation with optimal schedule and visualize results
    results = simulate_mortgage(
        mortgage_amount,
        term_months,
        fixed_rate,
        fixed_term_months,
        mortgage_rate_curve,
        savings_rate_curve,
        optimal_schedule,
        monthly_savings_contribution,
        initial_savings,
        typical_payment,
        asset_value,
        max_payment_after_fixed
    )

    # Create lists for plotting
    years = [data['month']/12 for data in results['month_data']]
    mortgage_balance = [data['principal_end'] for data in results['month_data']]
    savings_balance = [data['savings_balance_end'] for data in results['month_data']]
    net_worth = [s - m + asset_value for s, m in zip(savings_balance, mortgage_balance)]
    
    # Split monthly payments into principal and interest
    principal_payments = [data['monthly_payment'] - data['interest_paid'] for data in results['month_data']]
    interest_payments = [data['interest_paid'] for data in results['month_data']]
    
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

    # Create figure with three subplots sharing x axis
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), height_ratios=[2, 1, 1], sharex=True)
    
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
    
    # Middle subplot - Stacked bar chart for monthly payments
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
    # Format y-axis with pound symbol and comma separator for thousands
    def format_pounds(x, p):
        return '£{:,}'.format(int(x))
    
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(format_pounds))
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(format_pounds))
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(format_pounds))
    
    # Format x-axis to show whole years
    ax1.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax1.xaxis.set_major_formatter(plt.FormatStrFormatter('%d'))
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

    # Print a summary of the last recorded month with formatted output
    last_month_data = results['month_data'][-1]
    print("\nLast month data:")
    for k, v in last_month_data.items():
        if isinstance(v, float) and k not in ['monthly_interest_rate', 'annual_mortgage_rate', 'annual_savings_rate', 'monthly_savings_rate']:
            print(f"{k}: £{int(v):,}")
        else:
            print(f"{k}: {v}")
