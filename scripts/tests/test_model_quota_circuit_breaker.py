"""Tests for upstream model rate-limit backoff & sliding quota replenishment circuit breaker."""

import pytest
from scripts.model_quota_circuit_breaker import (
    BreakerState,
    CircuitMetrics,
    ModelQuotaCircuitBreaker,
    MultiModelQuotaRegistry,
    RouteDecision,
)


def test_circuit_breaker_initial_state():
    breaker = ModelQuotaCircuitBreaker(
        model_id="gpt-4o",
        fallback_model_id="gpt-4o-mini",
        max_tokens_per_minute=60000,
    )
    assert breaker.state == BreakerState.CLOSED
    assert breaker.token_bucket.capacity_tokens == 60000
    assert breaker.token_bucket.available_tokens == 60000.0


def test_circuit_breaker_normal_traffic_and_quota_consumption():
    t0 = 1000.0
    breaker = ModelQuotaCircuitBreaker(
        model_id="claude-3-5-sonnet",
        fallback_model_id="claude-3-haiku",
        max_tokens_per_minute=60000,
    )

    # 10k tokens consumed
    decision, target, retry_after = breaker.evaluate_request(10000, now=t0)
    assert decision == RouteDecision.ALLOW_PRIMARY
    assert target == "claude-3-5-sonnet"
    assert retry_after is None
    assert breaker.token_bucket.available_tokens == 50000.0


def test_circuit_breaker_sliding_quota_replenishment():
    t0 = 1000.0
    breaker = ModelQuotaCircuitBreaker(
        model_id="gpt-4o",
        max_tokens_per_minute=60000,  # 1000 tokens/sec
    )

    # Consume all 60k tokens
    breaker.evaluate_request(60000, now=t0)
    assert breaker.token_bucket.available_tokens == 0.0

    # Advance 15 seconds -> 15 * 1000 = 15,000 tokens replenished
    decision, target, _ = breaker.evaluate_request(10000, now=t0 + 15.0)
    assert decision == RouteDecision.ALLOW_PRIMARY
    assert breaker.token_bucket.available_tokens == 5000.0


def test_circuit_breaker_rate_limit_trips_to_open_and_fallback():
    t0 = 1000.0
    breaker = ModelQuotaCircuitBreaker(
        model_id="gemini-1.5-pro",
        fallback_model_id="gemini-1.5-flash",
        consecutive_rate_limits_to_open=2,
        base_cooldown_sec=20.0,
        jitter_factor=0.0,
    )

    # First 429
    breaker.record_rate_limit("gemini-1.5-pro", now=t0)
    assert breaker.state == BreakerState.CLOSED

    # Second 429 -> Circuit trips to OPEN
    breaker.record_rate_limit("gemini-1.5-pro", now=t0 + 1.0)
    assert breaker.state == BreakerState.OPEN
    assert breaker._opened_at == t0 + 1.0

    # Next request routes to fallback
    decision, target, retry_after = breaker.evaluate_request(1000, now=t0 + 2.0)
    assert decision == RouteDecision.ROUTE_FALLBACK
    assert target == "gemini-1.5-flash"
    assert retry_after is not None


def test_circuit_breaker_shed_when_no_fallback():
    t0 = 1000.0
    breaker = ModelQuotaCircuitBreaker(
        model_id="deepseek-v3",
        fallback_model_id=None,
        consecutive_rate_limits_to_open=1,
        base_cooldown_sec=10.0,
        jitter_factor=0.0,
    )

    breaker.record_rate_limit("deepseek-v3", now=t0)
    assert breaker.state == BreakerState.OPEN

    decision, target, _ = breaker.evaluate_request(1000, now=t0 + 1.0)
    assert decision == RouteDecision.SHED_REQUEST
    assert target is None


def test_circuit_breaker_half_open_recovery():
    t0 = 1000.0
    breaker = ModelQuotaCircuitBreaker(
        model_id="gpt-4o",
        fallback_model_id="gpt-4o-mini",
        consecutive_rate_limits_to_open=1,
        base_cooldown_sec=10.0,
        half_open_probe_limit=2,
        jitter_factor=0.0,
    )

    breaker.record_rate_limit("gpt-4o", now=t0)
    assert breaker.state == BreakerState.OPEN

    # Advance 11 seconds (cooldown elapsed) -> transitions to HALF_OPEN on request
    decision, target, _ = breaker.evaluate_request(1000, now=t0 + 11.0)
    assert breaker.state == BreakerState.HALF_OPEN
    assert decision == RouteDecision.ALLOW_PRIMARY
    assert target == "gpt-4o"

    # Successful probe 1
    breaker.record_success("gpt-4o")
    assert breaker.state == BreakerState.HALF_OPEN

    # Probe 2
    breaker.evaluate_request(1000, now=t0 + 12.0)
    breaker.record_success("gpt-4o")

    # Reaches probe limit -> recovers to CLOSED
    assert breaker.state == BreakerState.CLOSED


def test_circuit_breaker_half_open_failure_re_trip():
    t0 = 1000.0
    breaker = ModelQuotaCircuitBreaker(
        model_id="gpt-4o",
        fallback_model_id="gpt-4o-mini",
        consecutive_rate_limits_to_open=1,
        base_cooldown_sec=10.0,
        half_open_probe_limit=2,
        jitter_factor=0.0,
    )

    breaker.record_rate_limit("gpt-4o", now=t0)
    # Cooldown elapses
    breaker.evaluate_request(1000, now=t0 + 11.0)
    assert breaker.state == BreakerState.HALF_OPEN

    # Probe encounters another 429
    breaker.record_rate_limit("gpt-4o", now=t0 + 12.0)
    assert breaker.state == BreakerState.OPEN
    # Backoff increased
    assert breaker._consecutive_backoff_steps == 1


def test_multi_model_registry_isolation():
    registry = MultiModelQuotaRegistry()
    b1 = ModelQuotaCircuitBreaker("model-a", fallback_model_id="model-a-small", consecutive_rate_limits_to_open=1)
    b2 = ModelQuotaCircuitBreaker("model-b", fallback_model_id="model-b-small", consecutive_rate_limits_to_open=1)

    registry.register_breaker(b1)
    registry.register_breaker(b2)

    # Trip model-a
    b1.record_rate_limit("model-a")
    assert b1.state == BreakerState.OPEN

    # model-a routes fallback
    decision_a, target_a, _ = registry.route_request("model-a", 1000)
    assert decision_a == RouteDecision.ROUTE_FALLBACK
    assert target_a == "model-a-small"

    # model-b remains CLOSED and unaffected
    decision_b, target_b, _ = registry.route_request("model-b", 1000)
    assert decision_b == RouteDecision.ALLOW_PRIMARY
    assert target_b == "model-b"
