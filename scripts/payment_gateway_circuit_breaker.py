"""Payment gateway circuit breaker."""
import time


class CircuitOpenError(RuntimeError):
    """Raised when the payment gateway circuit breaker is open."""


class PaymentGatewayCircuitBreaker:
    """Protects payment gateway calls against cascading failures."""

    def __init__(self, failure_threshold: int = 3, cooldown_s: float = 30.0) -> None:
        self.failure_threshold = failure_threshold
        self.cooldown_s = cooldown_s
        self.failures = 0
        self.open_until = 0.0

    def call(self, fn):
        if time.monotonic() < self.open_until:
            raise CircuitOpenError("payment gateway circuit breaker open")
        try:
            res = fn()
        except Exception:
            self.failures += 1
            if self.failures >= self.failure_threshold:
                self.open_until = time.monotonic() + self.cooldown_s
                self.failures = 0
            raise
        self.failures = 0
        return res
