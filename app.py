import streamlit as st
import matplotlib.pyplot as plt
from main import simulate_mortgage, save_results_to_csv
import os

def main():
    st.set_page_config(layout="wide") # Set wide mode
    st.title("Interactive Mortgage Calculator")
    st.write("Adjust the parameters below to see how they affect your mortgage and savings.")

    # Create two columns for input parameters
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Mortgage Parameters")
        mortgage_amount = st.number_input(
            "Mortgage Amount (£)",
            min_value=0.0,
            value=187000.0,
            step=1000.0,
            format="%0.2f"
        )
        
        term_years = st.number_input(
            "Term (Years)",
            min_value=1.0,
            max_value=40.0,
            value=21.0,
            step=1.0,
            format="%0.1f"
        )
        
        fixed_rate = st.number_input(
            "Fixed Interest Rate (%)",
            min_value=0.0,
            max_value=15.0,
            value=1.65,
            step=0.1,
            format="%0.2f"
        )
        
        fixed_term_months = st.number_input(
            "Fixed Term (Months)",
            min_value=0,
            max_value=int(term_years * 12),
            value=12,
            step=1
        )
        
        variable_rate = st.number_input(
            "Variable Rate after Fixed Term (%)",
            min_value=0.0,
            max_value=15.0,
            value=6.0,
            step=0.1,
            format="%0.2f"
        )

    with col2:
        st.subheader("Savings Parameters")
        savings_rate = st.number_input(
            "Savings Interest Rate (%)",
            min_value=0.0,
            max_value=15.0,
            value=4.30,
            step=0.1,
            format="%0.2f"
        )
        
        monthly_savings = st.number_input(
            "Monthly Savings Contribution (£)",
            min_value=0.0,
            value=2500.0,
            step=100.0,
            format="%0.2f"
        )
        
        initial_savings = st.number_input(
            "Initial Savings (£)",
            min_value=0.0,
            value=170000.0,
            step=1000.0,
            format="%0.2f"
        )
        
        typical_payment = st.number_input(
            "Typical Monthly Payment (£)",
            min_value=0.0,
            value=878.0,
            step=10.0,
            format="%0.2f"
        )
        
        asset_value = st.number_input(
            "Property Value (£)",
            min_value=0.0,
            value=360000.0,
            step=1000.0,
            format="%0.2f"
        )

    # Advanced options in an expander
    with st.expander("Advanced Options"):
        has_max_payment = st.checkbox("Limit monthly payment after fixed term", value=False)
        if has_max_payment:
            max_payment = st.number_input(
                "Maximum Monthly Payment after Fixed Term (£)",
                min_value=0.0,
                value=1000.0,
                step=100.0,
                format="%0.2f"
            )
        else:
            max_payment = float('inf')
        
        overpayments = st.text_input(
            "Overpayments (format: 'month:amount,month:amount')",
            value="",
            help="Example: '18:20000,19:10000' for £20,000 in month 18 and £10,000 in month 19"
        )

    # Convert parameters
    term_months = int(term_years * 12)
    mortgage_rate_curve = [variable_rate for _ in range(term_months - fixed_term_months)]
    savings_rate_curve = [savings_rate for _ in range(term_months)]

    # Parse overpayments
    overpayment_schedule = {m: 0 for m in range(term_months)}
    if overpayments:
        pairs = overpayments.split(",")
        for pair in pairs:
            if ":" in pair:
                month_str, amount_str = pair.split(":")
                try:
                    month = int(month_str)
                    amount = float(amount_str)
                    if 0 <= month < term_months:
                        overpayment_schedule[month] = amount
                except ValueError:
                    st.warning(f"Invalid overpayment entry: {pair}")

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
        typical_payment,
        asset_value,
        max_payment
    )

    # Display charts
    st.subheader("Charts")
    
    # Create lists for plotting
    years = [data['month']/12 for data in results['month_data']]
    mortgage_balance = [data['principal_end'] for data in results['month_data']]
    savings_balance = [data['savings_balance_end'] for data in results['month_data']]
    net_worth = [s - m + asset_value for s, m in zip(savings_balance, mortgage_balance)]

    # Create figure and subplots
    fig = plt.figure(figsize=(12, 16))
    gs = fig.add_gridspec(5, 1, height_ratios=[0.2, 2, 1, 1, 1])
    
    # Stats text at the top
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

    # Main balance plot
    ax1 = fig.add_subplot(gs[1])
    ax1.plot(years, mortgage_balance, label='Mortgage Balance', color='red')
    ax1.plot(years, savings_balance, label='Savings Balance', color='green')
    ax1.plot(years, net_worth, label='Net Worth', color='blue', linestyle='--', linewidth=2)
    ax1.set_ylabel('Amount')
    ax1.set_title('Evolution of Mortgage, Savings and Net Worth Over Time')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    # Add minimum savings annotation if data available
    if "min_savings_balance" in results and "min_savings_month" in results:
        min_savings = results["min_savings_balance"]
        min_savings_month_idx = results["min_savings_month"] - 1 # Adjust to 0-based index
        if 0 <= min_savings_month_idx < len(years):
            min_savings_year = years[min_savings_month_idx]
            # Smart positioning logic (simplified)
            x_offset = 0.5
            y_offset = (ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.05 # 5% of y-axis range
            if min_savings_month_idx == 0:
                x_offset = 1.0
            if min_savings_month_idx == len(years) - 1:
                x_offset = -1.0
            y_pos = min_savings + y_offset
            if y_pos < ax1.get_ylim()[0]:
                y_pos = min_savings + abs(y_offset)
            
            ax1.annotate(
                f'Min Savings: £{int(min_savings):,}\nYear {min_savings_year:.1f}',
                xy=(min_savings_year, min_savings),
                xytext=(min_savings_year + x_offset, y_pos),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=4),
                bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7),
                ha='left' if x_offset > 0 else 'right'
            )

    # Monthly payments plot
    ax2 = fig.add_subplot(gs[2], sharex=ax1)
    width = 0.08
    monthly_payments = [data['monthly_payment'] for data in results['month_data']]
    interest_paid = [data['interest_paid'] for data in results['month_data']]
    # Calculate principal paid based on actual monthly payment
    principal_paid = [mp - ip for mp, ip in zip(monthly_payments, interest_paid)]
    # Ensure principal paid isn't negative if interest > payment (shouldn't happen with guards)
    principal_paid = [max(0, p) for p in principal_paid]
    
    ax2.bar(years, interest_paid, width, label="Interest", color="red", alpha=0.6)
    ax2.bar(years, principal_paid, width, bottom=interest_paid, label="Principal", color="blue", alpha=0.6)
    ax2.set_ylabel("Monthly Payment")
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.legend()

    # Add annotations for payment changes
    previous_payment = None
    for i, data in enumerate(results["month_data"]):
        current_payment = data['monthly_payment']
        if i > 0 and current_payment != previous_payment:
            year = data['month'] / 12
            # Annotate at the top of the principal bar
            annotation_y = interest_paid[i] + principal_paid[i]
            ax2.annotate(
                f"£{current_payment:,.0f}",
                xy=(year, annotation_y),
                xytext=(year, annotation_y + ax2.get_ylim()[1] * 0.1), # Offset slightly above
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=4),
                bbox=dict(boxstyle='round,pad=0.3', fc='lightblue', alpha=0.7),
                ha='center'
            )
        previous_payment = current_payment

    # Monthly savings plot
    ax3 = fig.add_subplot(gs[3], sharex=ax1)
    monthly_savings_data = [
        monthly_savings + (max(0, typical_payment - data['monthly_payment']) if typical_payment > 0 else 0)
        for data in results['month_data']
    ]
    ax3.bar(years, monthly_savings_data, width=0.08, color="green", alpha=0.6, label="Monthly Savings")
    ax3.set_ylabel("Monthly Savings")
    ax3.grid(True, linestyle="--", alpha=0.7)
    ax3.legend()

    # Interest comparison plot
    ax4 = fig.add_subplot(gs[4], sharex=ax1)
    interest_paid = [-data['interest_paid'] for data in results['month_data']]
    interest_received = [data['savings_interest'] for data in results['month_data']]
    ax4.bar(years, interest_paid, width, label="Interest Paid", color="red", alpha=0.6)
    ax4.bar(years, interest_received, width, label="Interest Received", color="green", alpha=0.6)
    ax4.set_xlabel("Years")
    ax4.set_ylabel("Monthly Interest")
    ax4.grid(True, linestyle="--", alpha=0.7)
    ax4.legend()

    # Format axes
    def format_pounds(x, p):
        return f"£{int(x):,}"

    ax1.yaxis.set_major_formatter(plt.FuncFormatter(format_pounds))
    ax1.yaxis.set_major_locator(plt.MultipleLocator(50000)) # Set Y-axis ticks to 50k intervals
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(format_pounds))
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(format_pounds))
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(format_pounds))

    # Set x-axis limits and ticks
    max_year = max(years)
    ax1.set_xlim(-0.5, max_year + 0.5)
    ax1.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax1.xaxis.set_major_formatter(plt.FormatStrFormatter("%d"))

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    # Display the figure in Streamlit
    st.pyplot(fig)
    plt.close()

    # --- New Mortgage Balance Only Chart ---
    st.subheader("Mortgage Balance Detail")
    fig_mortgage, ax_mortgage = plt.subplots(figsize=(12, 6))
    ax_mortgage.plot(years, mortgage_balance, label='Mortgage Balance', color='red')
    ax_mortgage.set_xlabel("Years")
    ax_mortgage.set_ylabel("Amount (£)")
    ax_mortgage.set_title("Mortgage Balance Over Time")
    ax_mortgage.grid(True, linestyle='--', alpha=0.7)
    ax_mortgage.legend()
    ax_mortgage.yaxis.set_major_formatter(plt.FuncFormatter(format_pounds))
    ax_mortgage.set_xlim(-0.5, max_year + 0.5)
    ax_mortgage.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax_mortgage.xaxis.set_major_formatter(plt.FormatStrFormatter("%d"))
    plt.tight_layout()
    st.pyplot(fig_mortgage)
    plt.close()
    # --- End New Chart ---

    # Display summary statistics
    st.subheader("Summary Statistics")
    last_month = results["month_data"][-1]
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Final Mortgage Balance", f"£{last_month['principal_end']:,.2f}")
        st.metric("Final Savings Balance", f"£{last_month['savings_balance_end']:,.2f}")
        if "min_savings_balance" in results:
            min_savings_month = results["min_savings_month"]
            st.metric("Lowest Savings Balance", f"£{results['min_savings_balance']:,.2f}", delta=f"Month {min_savings_month} (Year {min_savings_month/12:.1f})", delta_color="off")
    with col2:
        net_worth = last_month['savings_balance_end'] - last_month['principal_end'] + asset_value
        st.metric("Final Net Worth", f"£{net_worth:,.2f}")
        if "mortgage_paid_off_month" in results:
            months = results["mortgage_paid_off_month"]
            st.metric("Mortgage Paid Off After", f"{months} months (Year {months/12:.1f})")

    # Generate and offer CSV download
    csv_file_path = save_results_to_csv(results, asset_value)
    with open(csv_file_path, "r") as f:
        csv_data = f.read()
    
    st.download_button(
        label="Download Simulation Data as CSV",
        data=csv_data,
        file_name=os.path.basename(csv_file_path),
        mime="text/csv",
    )

    # Display warnings at the bottom
    if results["warnings"]:
        st.markdown("---")  # Add a separator
        st.subheader("Simulation Notes and Warnings")
        for warning in results["warnings"]:
            if warning.startswith("Note:"):
                st.info(warning)
            elif warning.startswith("Info:"):
                st.info(warning)
            else:
                st.warning(warning)

if __name__ == "__main__":
    main() 