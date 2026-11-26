"""Tests for PaymentGatewayCircuitBreaker."""
import pytest
from scripts.payment_gateway_circuit_breaker import (
    CircuitOpenError,
    PaymentGatewayCircuitBreaker,
)


def test_circuit_breaker_opens_after_threshold():
    cb = PaymentGatewayCircuitBreaker(failure_threshold=2, cooldown_s=5.0)

    def fail():
        raise ConnectionError("gateway down")

    with pytest.raises(ConnectionError):
        cb.call(fail)
    with pytest.raises(ConnectionError):
        cb.call(fail)
    with pytest.raises(CircuitOpenError):
        cb.call(lambda: "ok")
