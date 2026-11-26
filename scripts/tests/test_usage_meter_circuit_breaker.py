"""Tests for UsageMeterCircuitBreaker."""
import pytest
from scripts.usage_meter_circuit_breaker import (
    MeterCircuitOpenError,
    UsageMeterCircuitBreaker,
)


def test_usage_meter_circuit_breaker():
    cb = UsageMeterCircuitBreaker(threshold=2, cooldown_s=5.0)
    with pytest.raises(RuntimeError):
        cb.call(lambda: (_ for _ in ()).throw(RuntimeError("meter down")))
    with pytest.raises(RuntimeError):
        cb.call(lambda: (_ for _ in ()).throw(RuntimeError("meter down")))
    with pytest.raises(MeterCircuitOpenError):
        cb.call(lambda: "ok")
