"""Tests for tariff_burst_circuit_breaker."""
import sys
sys.path.insert(0, "scripts")
from tariff_burst_circuit_breaker import TariffBurstCircuitBreaker


def test_tariff_burst_circuit_breaker():
    cb = TariffBurstCircuitBreaker(threshold=2)
    assert cb.is_open() is False
    cb.record_failure(); cb.record_failure()
    assert cb.is_open() is True
