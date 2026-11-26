"""Tests for tier_lookup_profiler."""
from scripts.tier_lookup_profiler import benchmark_tier_lookup


def test_tier_lookup_profiler():
    ops = benchmark_tier_lookup({"pro": 0.01, "ent": 0.005}, ["pro", "ent", "free"], iterations=100)
    assert ops > 0
