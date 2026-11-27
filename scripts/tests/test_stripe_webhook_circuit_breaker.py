"""Tests for StripeWebhookCircuitBreaker."""
import pytest
from scripts.stripe_webhook_circuit_breaker import (
    StripeCircuitOpenError,
    StripeWebhookCircuitBreaker,
)


def test_stripe_webhook_circuit_breaker():
    cb = StripeWebhookCircuitBreaker(threshold=2, cooldown_s=5.0)
    with pytest.raises(RuntimeError):
        cb.call(lambda: (_ for _ in ()).throw(RuntimeError("502")))
    with pytest.raises(RuntimeError):
        cb.call(lambda: (_ for _ in ()).throw(RuntimeError("502")))
    with pytest.raises(StripeCircuitOpenError):
        cb.call(lambda: "ok")
