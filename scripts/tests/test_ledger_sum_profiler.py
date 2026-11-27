"""Tests for ledger_sum_profiler."""
from scripts.ledger_sum_profiler import benchmark_ledger_sum


def test_ledger_sum_profiler():
    ops = benchmark_ledger_sum([10, 20, 30, 40], iterations=100)
    assert ops > 0
