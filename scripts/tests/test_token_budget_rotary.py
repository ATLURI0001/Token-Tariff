"""Tests for token_budget_rotary."""
import sys
sys.path.insert(0, "scripts")
from token_budget_rotary import TokenBudgetRotary


def test_token_budget_rotary_bounded():
    r = TokenBudgetRotary(max_size=3)
    r.push(100)
    r.push(200)
    r.push(300)
    r.push(400)  # evicts 100
    assert r.count() == 3
    assert r.total() == 900
