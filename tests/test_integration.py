import pytest
import sys
import os
import subprocess
from pathlib import Path

# Configure matplotlib to use non-interactive backend before any other imports
import matplotlib
matplotlib.use('Agg')

# Add the parent directory to the Python path so we can import the main module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_script(args):
    """Helper function to run the main script with given arguments"""
    script_path = Path(__file__).parent.parent / "main.py"
    command = [sys.executable, str(script_path)] + args
    process = subprocess.run(
        command,
        capture_output=True,
        text=True
    )
    return process

def test_basic_command_execution():
    """Test script execution with minimum required arguments"""
    args = [
        "--mortgage-amount", "200000",
        "--term-years", "25",
        "--fixed-rate", "2.99",
        "--fixed-term-months", "24"
    ]
    
    result = run_script(args)
    
    # Check successful execution
    assert result.returncode == 0
    
    # Check expected output elements
    assert "Payment components around fixed-to-variable rate transition" in result.stdout
    assert "Chart data saved to:" in result.stdout
    assert "Last month data:" in result.stdout
    
    # Verify no error messages
    assert not result.stderr

def test_invalid_negative_arguments():
    """Test script behavior with invalid negative arguments"""
    test_cases = [
        {
            "args": ["--mortgage-amount", "-200000"],
            "error_message": "mortgage amount cannot be negative"
        },
        {
            "args": ["--fixed-rate", "-0.0299"],
            "error_message": "interest rate cannot be negative"
        },
        {
            "args": ["--term-years", "-25"],
            "error_message": "term years cannot be negative"
        }
    ]
    
    for case in test_cases:
        args = [
            "--mortgage-amount", "200000",
            "--term-years", "25",
            "--fixed-rate", "0.0299",
            "--fixed-term-months", "24"
        ]
        
        # Replace the relevant argument with negative value
        arg_index = args.index(case["args"][0])
        args[arg_index + 1] = case["args"][1]
        
        result = run_script(args)
        
        # Check error handling
        assert result.returncode != 0
        assert case["error_message"].lower() in result.stderr.lower()

def test_invalid_argument_formats():
    """Test script behavior with invalid argument formats"""
    test_cases = [
        {
            "args": ["--mortgage-amount", "abc"],
            "error_message": "invalid float value: 'abc'"
        },
        {
            "args": ["--fixed-rate", "abc"],
            "error_message": "invalid float value: 'abc'"
        },
        {
            "args": ["--term-years", "abc"],
            "error_message": "invalid float value: 'abc'"
        }
    ]
    
    for case in test_cases:
        args = [
            "--mortgage-amount", "200000",
            "--term-years", "25",
            "--fixed-rate", "0.0299",
            "--fixed-term-months", "24"
        ]
        
        # Replace the relevant argument with invalid format
        arg_index = args.index(case["args"][0])
        args[arg_index + 1] = case["args"][1]
        
        result = run_script(args)
        
        # Check error handling
        assert result.returncode != 0
        assert case["error_message"].lower() in result.stderr.lower()

def test_full_scenario_with_overpayments():
    """Test a full realistic scenario with all arguments including overpayments"""
    args = [
        "--mortgage-amount", "300000",
        "--term-years", "25",
        "--fixed-rate", "2.99",
        "--fixed-term-months", "24",
        "--variable-rate", "5.99",
        "--savings-rate", "3.0",
        "--monthly-savings", "2000",
        "--initial-savings", "100000",
        "--typical-payment", "1500",
        "--asset-value", "350000",
        "--overpayments", "24:20000,36:15000"
    ]
    
    result = run_script(args)
    
    # Check successful execution
    assert result.returncode == 0
    
    # Check expected output elements
    assert "Manual Overpayment Schedule:" in result.stdout
    assert "Month 24 (Year 2.0)" in result.stdout  # First overpayment
    assert "Month 36 (Year 3.0)" in result.stdout  # Second overpayment
    
    # Check for key simulation results
    assert "Payment components around fixed-to-variable rate transition" in result.stdout
    assert "Chart data saved to:" in result.stdout
    
    # Verify the simulation completed successfully
    assert "Last month data:" in result.stdout
    
    # Verify no error messages
    assert not result.stderr

def test_missing_required_arguments():
    """Test script behavior when required arguments are missing"""
    # Test with no arguments
    result = run_script([])
    assert result.returncode != 0
    assert "the following arguments are required" in result.stderr.lower()
    
    # Test with partial arguments
    partial_args = [
        "--mortgage-amount", "200000",
        "--term-years", "25"
    ]
    result = run_script(partial_args)
    assert result.returncode != 0
    assert "the following arguments are required" in result.stderr.lower()

def test_argument_validation():
    """Test validation of numeric arguments"""
    test_cases = [
        {
            "args": ["--mortgage-amount", "0"],
            "error": "mortgage amount cannot be negative or zero"
        },
        {
            "args": ["--term-years", "0"],
            "error": "term years cannot be negative or zero"
        },
        {
            "args": ["--fixed-term-months", "0"],
            "error": "fixed term months cannot be negative or zero"
        },
        {
            "args": ["--monthly-savings", "-100"],
            "error": "monthly savings contribution cannot be negative"
        },
        {
            "args": ["--initial-savings", "-1000"],
            "error": "initial savings cannot be negative"
        }
    ]
    
    base_args = [
        "--mortgage-amount", "200000",
        "--term-years", "25",
        "--fixed-rate", "0.0299",
        "--fixed-term-months", "24"
    ]
    
    for case in test_cases:
        args = list(base_args)  # Create a copy of base_args
        
        # Replace or add the invalid argument
        arg_name = case["args"][0]
        if arg_name in args:
            idx = args.index(arg_name)
            args[idx + 1] = case["args"][1]
        else:
            args.extend(case["args"])
        
        result = run_script(args)
        
        # Check error handling
        assert result.returncode != 0
        assert case["error"].lower() in result.stderr.lower()

def test_zero_interest_after_fixed():
    """Test behavior with 0% variable rate after fixed period"""
    args = [
        "--mortgage-amount", "300000",
        "--term-years", "25",
        "--fixed-rate", "2.99",
        "--fixed-term-months", "24",
        "--variable-rate", "0.0"  # Zero interest after fixed period
    ]
    
    result = run_script(args)
    assert result.returncode == 0
    
    # Extract payment data after fixed period
    fixed_period_end = False
    interest = None
    principal = None
    payment = None
    
    lines = result.stdout.split('\n')
    for i, line in enumerate(lines):
        if "Month 25:" in line:  # First month after fixed period
            fixed_period_end = True
            # Check next few lines for payment components
            for j in range(i, i+10):
                if "Interest Paid:" in lines[j]:
                    interest = float(lines[j].split('£')[1])
                elif "Principal Repaid:" in lines[j]:
                    principal = float(lines[j].split('£')[1])
                elif "Monthly Payment:" in lines[j]:
                    payment = float(lines[j].split('£')[1])
    
    assert fixed_period_end, "Could not find data after fixed period"
    assert interest is not None, "Could not find interest payment data"
    assert principal is not None, "Could not find principal payment data"
    assert payment is not None, "Could not find monthly payment data"
    
    # Now verify the payment components
    assert interest == 0.0, f"Interest payment should be zero after fixed period, got £{interest:.2f}"
    assert principal > 0, "Principal payment should be positive after fixed period"
    assert payment == principal, f"Monthly payment (£{payment:.2f}) should equal principal repayment (£{principal:.2f}) with zero interest"

def test_csv_output():
    """Test CSV output format and content"""
    args = [
        "--mortgage-amount", "300000",
        "--term-years", "25",
        "--fixed-rate", "2.99",
        "--fixed-term-months", "24",
        "--variable-rate", "5.99",
        "--savings-rate", "3.0",
        "--monthly-savings", "2000",
        "--initial-savings", "100000",
        "--asset-value", "350000"
    ]
    
    result = run_script(args)
    assert result.returncode == 0
    
    # Find the CSV file path
    csv_file = None
    for line in result.stdout.split('\n'):
        if "CSV data saved to:" in line:
            csv_file = line.split(": ")[1].strip()
            break
    
    assert csv_file is not None, "Could not find CSV file path in output"
    
    # Read and verify CSV content
    import csv
    with open(csv_file) as f:
        reader = csv.reader(f)
        headers = next(reader)  # Get headers
        first_row = next(reader)  # Get first data row
        
        # Verify headers
        expected_headers = [
            "Month", "Year", "Principal Start", "Principal End",
            "Monthly Payment", "Interest Paid", "Principal Repaid",
            "Overpayment", "Annual Mortgage Rate", "Monthly Interest Rate",
            "Savings Balance", "Annual Savings Rate", "Monthly Savings Rate",
            "Net Worth"
        ]
        assert headers == expected_headers, "CSV headers do not match expected format"
        
        # Verify first row data types and ranges
        assert int(float(first_row[0])) == 1, "First month should be 1"
        assert 0 < float(first_row[1]) < 0.1, "First year should be close to 0"
        assert float(first_row[2]) == 300000, "Initial principal should match mortgage amount"
        assert 0 <= float(first_row[8]) <= 100, "Annual mortgage rate should be between 0 and 100 percent"
        assert float(first_row[7]) == 0, "No overpayment expected in first month"
        
        # Read all rows to verify completeness
        rows = list(reader)
        assert len(rows) == 299, "Expected 300 rows total (25 years * 12 months)"
        
        # Verify last row
        last_row = rows[-1]
        assert float(last_row[3]) == 0, "Principal should be zero at end of term"
        assert float(last_row[1]) == 25.0, "Last year should be 25"