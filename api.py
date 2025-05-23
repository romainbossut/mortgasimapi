#!/usr/bin/env python3
"""
Mortgage Simulation API

Copyright (c) 2024 Romain Bossut. All Rights Reserved.
This software is proprietary and confidential. Unauthorized copying, distribution, 
or use of this software is strictly prohibited.
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import os

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, validator
import uvicorn

from main import (
    simulate_mortgage,
    create_overpayment_schedule,
    save_results_to_csv,
    parse_overpayment_string,
)


app = FastAPI(
    title="Mortgage Simulation API",
    description="A comprehensive mortgage and savings simulation API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware to allow localhost and 127.0.0.1 on any port
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://((localhost|127\.0\.0\.1)(:\d+)?|(www\.)?mortgasim\.com)$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request Models
class MortgageParameters(BaseModel):
    """Mortgage-related input parameters"""
    amount: float = Field(..., gt=0, description="Initial mortgage amount in pounds")
    term_years: float = Field(..., gt=0, le=40, description="Mortgage term in years")
    fixed_rate: float = Field(..., ge=0, le=15, description="Fixed interest rate as percentage (e.g., 1.65)")
    fixed_term_months: int = Field(..., ge=0, description="Fixed rate term in months")
    variable_rate: float = Field(6.0, ge=0, le=15, description="Variable rate after fixed term as percentage")
    max_payment_after_fixed: Optional[float] = Field(None, gt=0, description="Maximum monthly payment after fixed period")


class SavingsParameters(BaseModel):
    """Savings-related input parameters"""
    rate: float = Field(4.30, ge=0, le=15, description="Annual savings interest rate as percentage")
    monthly_contribution: float = Field(2500.0, ge=0, description="Monthly savings contribution in pounds")
    initial_balance: float = Field(170000.0, ge=0, description="Initial savings balance in pounds")


class SimulationParameters(BaseModel):
    """Additional simulation parameters"""
    typical_payment: float = Field(878.0, ge=0, description="Typical monthly payment - difference goes to savings if actual payment is lower")
    asset_value: float = Field(360000.0, ge=0, description="Property value in pounds")
    show_years_after_payoff: int = Field(5, ge=0, le=20, description="Years to show after mortgage is paid off")
    overpayments: Optional[str] = Field(None, description="Overpayments in format 'month:amount,month:amount' (e.g., '18:20000,19:10000')")


class SimulationRequest(BaseModel):
    """Complete simulation request"""
    mortgage: MortgageParameters
    savings: SavingsParameters
    simulation: SimulationParameters = SimulationParameters()
    
    @validator('mortgage')
    def validate_fixed_term(cls, v, values):
        """Ensure fixed term is less than total term"""
        term_months = int(v.term_years * 12)
        if v.fixed_term_months >= term_months:
            raise ValueError(f"Fixed term ({v.fixed_term_months} months) must be less than total term ({term_months} months)")
        return v


# Response Models
class MonthlyData(BaseModel):
    """Data for a single month of the simulation"""
    month: int = Field(..., description="1-based month number")
    year: float = Field(..., description="Year as decimal (e.g., 1.5 for month 18)")
    principal_start: float = Field(..., description="Principal at start of month")
    principal_end: float = Field(..., description="Principal at end of month")
    monthly_payment: float = Field(..., description="Base monthly payment")
    overpayment: float = Field(..., description="Any overpayment made")
    total_payment: float = Field(..., description="Monthly payment + overpayment")
    interest_paid: float = Field(..., description="Interest portion of payment")
    principal_repaid: float = Field(..., description="Principal portion of payment")
    savings_balance_end: float = Field(..., description="Savings balance at end of month")
    savings_interest: float = Field(..., description="Interest earned on savings")
    net_worth: float = Field(..., description="Savings - mortgage + asset value")
    annual_mortgage_rate: float = Field(..., description="Mortgage rate as percentage")
    monthly_interest_rate: float = Field(..., description="Monthly mortgage rate as decimal")
    annual_savings_rate: float = Field(..., description="Savings rate as percentage")
    monthly_savings_rate: float = Field(..., description="Monthly savings rate as decimal")
    payment_difference: float = Field(..., description="Difference between typical payment and actual payment")


class SummaryStatistics(BaseModel):
    """Summary statistics from the simulation"""
    final_mortgage_balance: float = Field(..., description="Final mortgage balance")
    final_savings_balance: float = Field(..., description="Final savings balance")
    final_net_worth: float = Field(..., description="Final net worth")
    min_savings_balance: float = Field(..., description="Lowest savings balance during simulation")
    min_savings_month: int = Field(..., description="Month when savings hit minimum")
    mortgage_paid_off_month: Optional[int] = Field(None, description="Month when mortgage is paid off (if applicable)")
    fixed_term_end_balance: Optional[float] = Field(None, description="Balance at end of fixed term")


class ChartData(BaseModel):
    """Aggregated data suitable for charting"""
    years: List[float] = Field(..., description="Year values for x-axis")
    mortgage_balance: List[float] = Field(..., description="Mortgage balance over time")
    savings_balance: List[float] = Field(..., description="Savings balance over time")
    net_worth: List[float] = Field(..., description="Net worth over time")
    monthly_payments: List[float] = Field(..., description="Monthly payments over time")
    interest_paid: List[float] = Field(..., description="Monthly interest paid")
    principal_paid: List[float] = Field(..., description="Monthly principal paid")
    monthly_savings_data: List[float] = Field(..., description="Monthly savings contributions")
    interest_received: List[float] = Field(..., description="Monthly interest received from savings")


class SimulationResponse(BaseModel):
    """Complete simulation response"""
    monthly_data: List[MonthlyData] = Field(..., description="Month-by-month simulation data")
    summary_statistics: SummaryStatistics = Field(..., description="Summary statistics")
    chart_data: ChartData = Field(..., description="Data formatted for charting")
    warnings: List[str] = Field(..., description="Any warnings or notes from the simulation")


# Utility functions
def create_chart_data(results: Dict[str, Any], request: SimulationRequest, display_limit_month: Optional[int] = None) -> ChartData:
    """Convert simulation results to chart-friendly format"""
    month_data = results['month_data']
    
    # Determine display limit
    if display_limit_month is None:
        display_limit_month = len(month_data)
    
    # Extract base data
    years = [data['year'] for data in month_data[:display_limit_month]]
    mortgage_balance = [data['principal_end'] for data in month_data[:display_limit_month]]
    savings_balance = [data['savings_balance_end'] for data in month_data[:display_limit_month]]
    net_worth = [data['net_worth'] for data in month_data[:display_limit_month]]
    
    # Payment breakdown data
    monthly_payments = [data['monthly_payment'] for data in month_data[:display_limit_month]]
    interest_paid = [data['interest_paid'] for data in month_data[:display_limit_month]]
    principal_paid = [max(0, data['principal_repaid']) for data in month_data[:display_limit_month]]
    
    # Savings data
    monthly_savings_data = [
        request.savings.monthly_contribution + 
        (max(0, request.simulation.typical_payment - data['monthly_payment']) if request.simulation.typical_payment > 0 else 0)
        for data in month_data[:display_limit_month]
    ]
    interest_received = [data['savings_interest'] for data in month_data[:display_limit_month]]
    
    return ChartData(
        years=years,
        mortgage_balance=mortgage_balance,
        savings_balance=savings_balance,
        net_worth=net_worth,
        monthly_payments=monthly_payments,
        interest_paid=interest_paid,
        principal_paid=principal_paid,
        monthly_savings_data=monthly_savings_data,
        interest_received=interest_received
    )


def create_summary_statistics(results: Dict[str, Any], request: SimulationRequest) -> SummaryStatistics:
    """Create summary statistics from simulation results"""
    last_month = results["month_data"][-1]
    
    # Calculate fixed term end balance
    fixed_term_end_balance = None
    if (request.mortgage.fixed_term_months > 0 and 
        request.mortgage.fixed_term_months <= len(results["month_data"])):
        fixed_term_end_balance = results["month_data"][request.mortgage.fixed_term_months - 1]["principal_end"]
    
    return SummaryStatistics(
        final_mortgage_balance=last_month['principal_end'],
        final_savings_balance=last_month['savings_balance_end'],
        final_net_worth=last_month['net_worth'],
        min_savings_balance=results["min_savings_balance"],
        min_savings_month=results["min_savings_month"],
        mortgage_paid_off_month=results.get("mortgage_paid_off_month"),
        fixed_term_end_balance=fixed_term_end_balance
    )


# API Endpoints
@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint"""
    return {"message": "Mortgage Simulation API", "status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/simulate", response_model=SimulationResponse, tags=["Simulation"])
async def simulate_mortgage_endpoint(request: SimulationRequest):
    """
    Run a complete mortgage and savings simulation
    
    This endpoint performs a comprehensive simulation of mortgage payments, savings growth,
    and net worth over time based on the provided parameters.
    
    **Key Features:**
    - Fixed-rate period followed by variable rate
    - Overpayment handling with savings deduction
    - Savings growth with interest compounding
    - Payment constraints and feasibility checks
    - Net worth calculation including property value
    
    **Returns:**
    - Month-by-month detailed data
    - Summary statistics
    - Chart-ready data arrays
    - Warnings and validation messages
    """
    try:
        # Convert parameters
        term_months = int(request.mortgage.term_years * 12)
        
        # Create rate curves
        mortgage_rate_curve = [request.mortgage.variable_rate for _ in range(term_months - request.mortgage.fixed_term_months)]
        savings_rate_curve = [request.savings.rate for _ in range(term_months)]
        
        # Parse overpayments
        overpayment_schedule = {}
        if request.simulation.overpayments:
            overpayment_schedule = parse_overpayment_string(request.simulation.overpayments, term_months)
        else:
            overpayment_schedule = dict.fromkeys(range(term_months), 0)
        
        # Set max payment
        max_payment = request.mortgage.max_payment_after_fixed or float('inf')
        
        # Run simulation
        results = simulate_mortgage(
            mortgage_amount=request.mortgage.amount,
            term_months=term_months,
            fixed_rate=request.mortgage.fixed_rate,
            fixed_term_months=request.mortgage.fixed_term_months,
            mortgage_rate_curve=mortgage_rate_curve,
            savings_rate_curve=savings_rate_curve,
            overpayment_schedule=overpayment_schedule,
            monthly_savings_contribution=request.savings.monthly_contribution,
            initial_savings=request.savings.initial_balance,
            typical_payment=request.simulation.typical_payment,
            asset_value=request.simulation.asset_value,
            max_payment_after_fixed=max_payment
        )
        
        # Convert month data to Pydantic models
        monthly_data = [MonthlyData(**data) for data in results['month_data']]
        
        # Determine display limit for charts
        display_limit_month = term_months
        if "mortgage_paid_off_month" in results:
            payoff_month = results["mortgage_paid_off_month"]
            display_limit_month = min(term_months, payoff_month + (request.simulation.show_years_after_payoff * 12))
        
        # Create response
        response = SimulationResponse(
            monthly_data=monthly_data,
            summary_statistics=create_summary_statistics(results, request),
            chart_data=create_chart_data(results, request, display_limit_month),
            warnings=results['warnings']
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Simulation error: {str(e)}")


@app.post("/simulate/csv", tags=["Export"])
async def export_simulation_csv(request: SimulationRequest):
    """
    Run simulation and return results as CSV file download
    
    This endpoint runs the same simulation as /simulate but returns the results
    as a downloadable CSV file for further analysis in Excel or other tools.
    """
    try:
        # Run simulation (reuse logic from simulate endpoint)
        term_months = int(request.mortgage.term_years * 12)
        mortgage_rate_curve = [request.mortgage.variable_rate for _ in range(term_months - request.mortgage.fixed_term_months)]
        savings_rate_curve = [request.savings.rate for _ in range(term_months)]
        
        overpayment_schedule = {}
        if request.simulation.overpayments:
            overpayment_schedule = parse_overpayment_string(request.simulation.overpayments, term_months)
        else:
            overpayment_schedule = dict.fromkeys(range(term_months), 0)
        
        max_payment = request.mortgage.max_payment_after_fixed or float('inf')
        
        results = simulate_mortgage(
            mortgage_amount=request.mortgage.amount,
            term_months=term_months,
            fixed_rate=request.mortgage.fixed_rate,
            fixed_term_months=request.mortgage.fixed_term_months,
            mortgage_rate_curve=mortgage_rate_curve,
            savings_rate_curve=savings_rate_curve,
            overpayment_schedule=overpayment_schedule,
            monthly_savings_contribution=request.savings.monthly_contribution,
            initial_savings=request.savings.initial_balance,
            typical_payment=request.simulation.typical_payment,
            asset_value=request.simulation.asset_value,
            max_payment_after_fixed=max_payment
        )
        
        # Save to CSV and return file
        csv_file_path = save_results_to_csv(results, request.simulation.asset_value)
        
        return FileResponse(
            path=csv_file_path,
            filename=os.path.basename(csv_file_path),
            media_type='text/csv'
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Export error: {str(e)}")


@app.get("/simulate/sample", response_model=SimulationRequest, tags=["Examples"])
async def get_sample_request():
    """
    Get a sample simulation request with realistic default values
    
    This endpoint returns a properly formatted sample request that can be used
    as a starting point for customization or testing.
    """
    return SimulationRequest(
        mortgage=MortgageParameters(
            amount=187000.0,
            term_years=21.0,
            fixed_rate=1.65,
            fixed_term_months=12,
            variable_rate=6.0
        ),
        savings=SavingsParameters(
            rate=4.30,
            monthly_contribution=2500.0,
            initial_balance=170000.0
        ),
        simulation=SimulationParameters(
            typical_payment=878.0,
            asset_value=360000.0,
            show_years_after_payoff=5
        )
    )


@app.get("/overpayment-schedule/create", tags=["Utilities"])
async def create_overpayment_schedule_endpoint(
    term_months: int = Query(..., description="Total mortgage term in months"),
    schedule_type: str = Query(..., description="Type of schedule: 'none', 'fixed', 'lump_sum', 'yearly_bonus', 'custom'"),
    monthly_amount: Optional[float] = Query(None, description="For 'fixed' type: monthly overpayment amount"),
    bonus_month: Optional[int] = Query(None, description="For 'yearly_bonus' type: month of year (1-12)"),
    bonus_amount: Optional[float] = Query(None, description="For 'yearly_bonus' type: annual bonus amount"),
    lump_sums: Optional[str] = Query(None, description="For 'lump_sum' type: 'month:amount,month:amount' format"),
):
    """
    Generate an overpayment schedule based on different patterns
    
    **Schedule Types:**
    - **none**: No overpayments
    - **fixed**: Fixed amount every month
    - **lump_sum**: Specific amounts in specific months
    - **yearly_bonus**: Same amount every year in specific month
    - **custom**: Custom schedule via lump_sums parameter
    
    Returns a string that can be used in the overpayments field of simulation requests.
    """
    try:
        kwargs = {}
        
        if schedule_type == "fixed" and monthly_amount is not None:
            kwargs["monthly_amount"] = monthly_amount
        elif schedule_type == "yearly_bonus":
            if bonus_month is not None:
                kwargs["bonus_month"] = bonus_month
            if bonus_amount is not None:
                kwargs["bonus_amount"] = bonus_amount
        elif schedule_type in ["lump_sum", "custom"] and lump_sums:
            # Parse lump_sums string into dict
            lump_sum_dict = {}
            pairs = lump_sums.split(",")
            for pair in pairs:
                if ":" in pair:
                    month_str, amount_str = pair.split(":")
                    month = int(month_str)
                    amount = float(amount_str)
                    lump_sum_dict[month] = amount
            
            if schedule_type == "lump_sum":
                kwargs["lump_sums"] = lump_sum_dict
            else:  # custom
                kwargs["custom_schedule"] = lump_sum_dict
        
        schedule = create_overpayment_schedule(term_months, schedule_type, **kwargs)
        
        # Convert to string format for API response
        overpayment_string_parts = []
        for month, amount in schedule.items():
            if amount > 0:
                overpayment_string_parts.append(f"{month}:{amount}")
        
        overpayment_string = ",".join(overpayment_string_parts)
        
        return {
            "schedule_type": schedule_type,
            "term_months": term_months,
            "overpayment_string": overpayment_string,
            "schedule_dict": {k: v for k, v in schedule.items() if v > 0},
            "total_overpayments": sum(schedule.values())
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Schedule creation error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 