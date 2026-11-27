"""Tests for OverageCountRotary."""
from scripts.overage_count_rotary import OverageCountRotary


def test_overage_count_rotary():
    r = OverageCountRotary(capacity=2)
    r.record(1)
    r.record(3)
    r.record(5)
    assert r.avg_count() == 4.0
