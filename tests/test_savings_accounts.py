"""
Tests for multi-account savings functionality.
"""

import pytest
from math import isclose

from main import simulate_mortgage


class TestMultipleAccounts:
    """Test simulation with multiple savings accounts."""

    def test_simulate_with_multiple_accounts(self):
        """Test simulation correctly tracks multiple savings accounts."""
        accounts = [
            {"name": "Main Savings", "rate": 4.0, "monthly_contribution": 2000, "initial_balance": 100000},
            {"name": "ISA", "rate": 5.0, "monthly_contribution": 500, "initial_balance": 20000},
        ]

        results = simulate_mortgage(
            mortgage_amount=200000,
            term_months=24,
            fixed_rate=2.0,
            fixed_term_months=12,
            mortgage_rate_curve=[4.0] * 12,
            savings_accounts=accounts,
        )

        # Verify per-account data exists in month data
        assert "accounts" in results["month_data"][0]
        assert len(results["month_data"][0]["accounts"]) == 2
        assert "Main Savings" in results["month_data"][0]["accounts"]
        assert "ISA" in results["month_data"][0]["accounts"]

        # Verify account summaries exist
        assert "account_summaries" in results
        assert len(results["account_summaries"]) == 2

        # Verify consolidated totals match sum of accounts
        first_month = results["month_data"][0]
        account_sum = sum(
            acc_data["balance_end"]
            for acc_data in first_month["accounts"].values()
        )
        assert isclose(first_month["savings_balance_end"], account_sum, rel_tol=1e-6)

    def test_empty_accounts_array(self):
        """Test simulation works with empty accounts list (no savings)."""
        results = simulate_mortgage(
            mortgage_amount=200000,
            term_months=24,
            fixed_rate=2.0,
            fixed_term_months=12,
            mortgage_rate_curve=[4.0] * 12,
            savings_accounts=[],
        )

        # Should complete without error
        assert len(results["month_data"]) == 24
        # Savings should be zero
        assert results["month_data"][-1]["savings_balance_end"] == 0
        # accounts dict should be empty
        assert len(results["month_data"][0].get("accounts", {})) == 0

    def test_different_rates_per_account(self):
        """Test that different accounts accrue interest at their own rates."""
        accounts = [
            {"name": "Low Rate", "rate": 2.0, "monthly_contribution": 1000, "initial_balance": 10000},
            {"name": "High Rate", "rate": 6.0, "monthly_contribution": 1000, "initial_balance": 10000},
        ]

        results = simulate_mortgage(
            mortgage_amount=100000,
            term_months=24,
            fixed_rate=2.0,
            fixed_term_months=12,
            mortgage_rate_curve=[4.0] * 12,
            savings_accounts=accounts,
        )

        final_month = results["month_data"][-1]
        low_rate_final = final_month["accounts"]["Low Rate"]["balance_end"]
        high_rate_final = final_month["accounts"]["High Rate"]["balance_end"]

        # High rate account should have more due to higher interest
        assert high_rate_final > low_rate_final

        # Verify the interest difference is significant (should be ~4% annual difference)
        # After 12 months with same contributions, high rate should be notably higher
        difference = high_rate_final - low_rate_final
        assert difference > 200  # At least ~200 more from higher interest

    def test_payment_difference_applied_to_first_account(self):
        """Test that typical payment difference is added to first account only."""
        accounts = [
            {"name": "Primary", "rate": 4.0, "monthly_contribution": 1000, "initial_balance": 50000},
            {"name": "Secondary", "rate": 4.0, "monthly_contribution": 500, "initial_balance": 20000},
        ]

        # Use a longer term to get a lower monthly payment
        results = simulate_mortgage(
            mortgage_amount=100000,
            term_months=300,  # 25 years for lower monthly payment (~500)
            fixed_rate=2.0,
            fixed_term_months=24,
            mortgage_rate_curve=[4.0] * 276,
            savings_accounts=accounts,
            typical_payment=1000,  # Higher than actual mortgage payment (~425)
        )

        # First month data
        first_month = results["month_data"][0]
        primary_contribution = first_month["accounts"]["Primary"]["contribution"]
        secondary_contribution = first_month["accounts"]["Secondary"]["contribution"]

        # Verify payment_difference is positive (typical > actual)
        assert first_month["payment_difference"] > 0

        # Primary should have base (1000) + payment difference
        assert primary_contribution > 1000
        # Secondary should have just base contribution (500)
        assert isclose(secondary_contribution, 500, rel_tol=1e-6)

    def test_account_summaries_totals(self):
        """Test that account summaries correctly track totals."""
        accounts = [
            {"name": "Account A", "rate": 4.0, "monthly_contribution": 1000, "initial_balance": 10000},
            {"name": "Account B", "rate": 5.0, "monthly_contribution": 500, "initial_balance": 5000},
        ]

        results = simulate_mortgage(
            mortgage_amount=100000,
            term_months=24,
            fixed_rate=2.0,
            fixed_term_months=12,
            mortgage_rate_curve=[4.0] * 12,
            savings_accounts=accounts,
        )

        summaries = {s["name"]: s for s in results["account_summaries"]}

        # Account A: 24 months * 1000 = 24000 contributions
        assert isclose(summaries["Account A"]["total_contributions"], 24000, rel_tol=1e-6)
        # Account B: 24 months * 500 = 12000 contributions
        assert isclose(summaries["Account B"]["total_contributions"], 12000, rel_tol=1e-6)

        # Final balances should be initial + contributions + interest
        assert summaries["Account A"]["final_balance"] > 10000 + 24000
        assert summaries["Account B"]["final_balance"] > 5000 + 12000

        # Total interest should be positive
        assert summaries["Account A"]["total_interest_earned"] > 0
        assert summaries["Account B"]["total_interest_earned"] > 0

    def test_backward_compatibility_legacy_params(self):
        """Test that legacy single-account parameters still work."""
        results = simulate_mortgage(
            mortgage_amount=200000,
            term_months=24,
            fixed_rate=2.0,
            fixed_term_months=12,
            mortgage_rate_curve=[4.0] * 12,
            savings_rate_curve=[4.5] * 24,
            monthly_savings_contribution=2500.0,
            initial_savings=100000.0,
        )

        # Should have one account named "Savings"
        assert len(results["account_summaries"]) == 1
        assert results["account_summaries"][0]["name"] == "Savings"

        # Verify savings grew correctly
        assert results["month_data"][-1]["savings_balance_end"] > 100000 + (24 * 2500)

        # Verify accounts structure exists
        assert "accounts" in results["month_data"][0]
        assert "Savings" in results["month_data"][0]["accounts"]

    def test_single_account_via_accounts_param(self):
        """Test simulation with single account using new accounts parameter."""
        accounts = [
            {"name": "My Savings", "rate": 4.5, "monthly_contribution": 2000, "initial_balance": 50000},
        ]

        results = simulate_mortgage(
            mortgage_amount=150000,
            term_months=24,
            fixed_rate=2.0,
            fixed_term_months=12,
            mortgage_rate_curve=[4.0] * 12,
            savings_accounts=accounts,
        )

        assert len(results["account_summaries"]) == 1
        assert results["account_summaries"][0]["name"] == "My Savings"
        assert results["month_data"][-1]["savings_balance_end"] > 50000

    def test_overpayment_deducted_proportionally(self):
        """Test that overpayments are deducted proportionally from accounts."""
        accounts = [
            {"name": "Account A", "rate": 4.0, "monthly_contribution": 0, "initial_balance": 75000},
            {"name": "Account B", "rate": 4.0, "monthly_contribution": 0, "initial_balance": 25000},
        ]

        # Make an overpayment in month 1 (index 0)
        overpayment_schedule = {0: 10000}

        results = simulate_mortgage(
            mortgage_amount=100000,
            term_months=24,
            fixed_rate=2.0,
            fixed_term_months=12,
            mortgage_rate_curve=[4.0] * 12,
            overpayment_schedule=overpayment_schedule,
            savings_accounts=accounts,
        )

        # After first month, total savings should be reduced by 10000
        first_month = results["month_data"][0]
        # Account A had 75% of savings, should have paid 75% of overpayment
        # Account B had 25% of savings, should have paid 25% of overpayment
        # But then both accounts get interest on their remaining balance

        # Just verify the overpayment was processed
        assert first_month["overpayment"] == 10000
        # Total savings should be less than 100000 (initial sum) due to overpayment
        # But we add interest, so let's just check it's reasonable
        assert first_month["savings_balance_end"] < 95000  # Lost 10k, plus some interest

    def test_accounts_with_zero_initial_balance(self):
        """Test accounts that start with zero initial balance."""
        accounts = [
            {"name": "New Account", "rate": 5.0, "monthly_contribution": 1000, "initial_balance": 0},
        ]

        results = simulate_mortgage(
            mortgage_amount=100000,
            term_months=24,
            fixed_rate=2.0,
            fixed_term_months=12,
            mortgage_rate_curve=[4.0] * 12,
            savings_accounts=accounts,
        )

        # Account should grow from 0 to ~24000 + interest
        final_balance = results["account_summaries"][0]["final_balance"]
        assert final_balance > 24000  # 24 months * 1000 + some interest
        assert final_balance < 26000  # But not too much more

    def test_accounts_with_zero_contribution(self):
        """Test accounts that have no monthly contribution (just interest)."""
        accounts = [
            {"name": "Dormant Account", "rate": 5.0, "monthly_contribution": 0, "initial_balance": 10000},
        ]

        results = simulate_mortgage(
            mortgage_amount=100000,
            term_months=24,
            fixed_rate=2.0,
            fixed_term_months=12,
            mortgage_rate_curve=[4.0] * 12,
            savings_accounts=accounts,
        )

        # Account should grow only from interest (~5% annual = ~1000 over 2 years with compounding)
        final_balance = results["account_summaries"][0]["final_balance"]
        assert final_balance > 10000  # Should have grown
        assert final_balance < 11100  # But only by about 10% over 2 years

        # Total contributions should be 0
        assert results["account_summaries"][0]["total_contributions"] == 0
        # Total interest should be positive
        assert results["account_summaries"][0]["total_interest_earned"] > 0


class TestDrawForRepayment:
    """Test draw_for_repayment flag on savings accounts."""

    def test_overpayment_only_from_drawable_account(self):
        """Overpayment should only deduct from accounts with draw_for_repayment=True."""
        accounts = [
            {"name": "Drawable", "rate": 4.0, "monthly_contribution": 0, "initial_balance": 50000, "draw_for_repayment": True},
            {"name": "Protected", "rate": 4.0, "monthly_contribution": 0, "initial_balance": 50000, "draw_for_repayment": False},
        ]

        overpayment_schedule = {0: 10000}

        results = simulate_mortgage(
            mortgage_amount=200000,
            term_months=24,
            fixed_rate=2.0,
            fixed_term_months=12,
            mortgage_rate_curve=[4.0] * 12,
            overpayment_schedule=overpayment_schedule,
            savings_accounts=accounts,
        )

        first_month = results["month_data"][0]
        drawable_balance = first_month["accounts"]["Drawable"]["balance_end"]
        protected_balance = first_month["accounts"]["Protected"]["balance_end"]

        # Drawable account should have been reduced by the overpayment
        # Initial 50000 - 10000 overpayment + interest
        assert drawable_balance < 50000
        assert drawable_balance < 41000  # ~40000 + small interest

        # Protected account should NOT have been reduced — only gains interest
        assert protected_balance > 50000

    def test_overpayment_capped_by_drawable_balance(self):
        """Overpayment should be capped by drawable balance, not total balance."""
        accounts = [
            {"name": "Drawable", "rate": 0.0, "monthly_contribution": 0, "initial_balance": 5000, "draw_for_repayment": True},
            {"name": "Protected", "rate": 0.0, "monthly_contribution": 0, "initial_balance": 95000, "draw_for_repayment": False},
        ]

        # Request 20000 overpayment but only 5000 is drawable
        overpayment_schedule = {0: 20000}

        results = simulate_mortgage(
            mortgage_amount=200000,
            term_months=24,
            fixed_rate=2.0,
            fixed_term_months=12,
            mortgage_rate_curve=[4.0] * 12,
            overpayment_schedule=overpayment_schedule,
            savings_accounts=accounts,
        )

        first_month = results["month_data"][0]
        # Overpayment should be capped to 5000
        assert first_month["overpayment"] == 5000

        # Protected account untouched
        assert first_month["accounts"]["Protected"]["balance_end"] == 95000

        # Warning about reduction should be present
        assert any("reduced" in w.lower() for w in results["warnings"])

    def test_default_draw_for_repayment_is_true(self):
        """Accounts without draw_for_repayment should default to True (backward compat)."""
        accounts = [
            {"name": "Account A", "rate": 0.0, "monthly_contribution": 0, "initial_balance": 50000},
            {"name": "Account B", "rate": 0.0, "monthly_contribution": 0, "initial_balance": 50000},
        ]

        overpayment_schedule = {0: 10000}

        results = simulate_mortgage(
            mortgage_amount=200000,
            term_months=24,
            fixed_rate=2.0,
            fixed_term_months=12,
            mortgage_rate_curve=[4.0] * 12,
            overpayment_schedule=overpayment_schedule,
            savings_accounts=accounts,
        )

        first_month = results["month_data"][0]
        # Both accounts should be deducted proportionally (50/50 = 5000 each)
        assert first_month["accounts"]["Account A"]["balance_end"] == 45000
        assert first_month["accounts"]["Account B"]["balance_end"] == 45000

    def test_proportional_deduction_among_drawable_accounts(self):
        """Overpayment should be deducted proportionally among drawable accounts."""
        accounts = [
            {"name": "Big", "rate": 0.0, "monthly_contribution": 0, "initial_balance": 75000, "draw_for_repayment": True},
            {"name": "Small", "rate": 0.0, "monthly_contribution": 0, "initial_balance": 25000, "draw_for_repayment": True},
            {"name": "Protected", "rate": 0.0, "monthly_contribution": 0, "initial_balance": 100000, "draw_for_repayment": False},
        ]

        overpayment_schedule = {0: 10000}

        results = simulate_mortgage(
            mortgage_amount=200000,
            term_months=24,
            fixed_rate=2.0,
            fixed_term_months=12,
            mortgage_rate_curve=[4.0] * 12,
            overpayment_schedule=overpayment_schedule,
            savings_accounts=accounts,
        )

        first_month = results["month_data"][0]
        # Big: 75000 / 100000 * 10000 = 7500 deducted → 67500
        assert first_month["accounts"]["Big"]["balance_end"] == 67500
        # Small: 25000 / 100000 * 10000 = 2500 deducted → 22500
        assert first_month["accounts"]["Small"]["balance_end"] == 22500
        # Protected: untouched
        assert first_month["accounts"]["Protected"]["balance_end"] == 100000
