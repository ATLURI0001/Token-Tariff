"""Unit tests for BudgetCircuitBreaker and tiered model load shedding."""
import time
import pytest

from scripts.budget_circuit_breaker import (
    BudgetCircuitBreaker,
    CircuitState,
    BudgetTier,
)


def test_circuit_initial_state_and_spending():
    cb = BudgetCircuitBreaker(tenant_id="tenant-1", budget_limit_usd=10.0)

    assert cb.state == CircuitState.CLOSED
    assert cb.current_tier() == BudgetTier.GREEN
    assert cb.spent_usd == 0.0

    allowed, msg = cb.can_execute(estimated_cost_usd=1.0)
    assert allowed is True

    record = cb.record_spend(tokens=1000, actual_cost_usd=1.5)
    assert record.tokens == 1000
    assert cb.spent_usd == 1.5
    assert len(record.signature) == 64  # SHA256 hex string


def test_budget_tiers_and_model_shedding():
    cb = BudgetCircuitBreaker(tenant_id="tenant-tier", budget_limit_usd=100.0)

    # 1. Under 70% -> GREEN tier: All models permitted
    cb.record_spend(tokens=10000, actual_cost_usd=50.0)
    assert cb.current_tier() == BudgetTier.GREEN
    allowed, _ = cb.is_model_allowed(model_tariff_per_million_usd=15.0)
    assert allowed is True

    # 2. 75% spend -> AMBER tier: High tariff (> $10/MTok) shed
    cb.record_spend(tokens=10000, actual_cost_usd=25.0)
    assert cb.current_tier() == BudgetTier.AMBER
    allowed, reason = cb.is_model_allowed(model_tariff_per_million_usd=15.0)
    assert allowed is False
    assert "AMBER" in reason
    allowed_cheap, _ = cb.is_model_allowed(model_tariff_per_million_usd=5.0)
    assert allowed_cheap is True

    # 3. 95% spend -> RED tier: Moderate tariff (> $2/MTok) shed
    cb.record_spend(tokens=5000, actual_cost_usd=20.0)
    assert cb.current_tier() == BudgetTier.RED
    allowed_mid, reason = cb.is_model_allowed(model_tariff_per_million_usd=5.0)
    assert allowed_mid is False
    assert "RED" in reason
    allowed_budget, _ = cb.is_model_allowed(model_tariff_per_million_usd=0.50)
    assert allowed_budget is True


def test_circuit_trips_to_open_on_budget_breach():
    cb = BudgetCircuitBreaker(tenant_id="tenant-trip", budget_limit_usd=10.0, cooldown_base_seconds=0.1)

    # Spend 10.5 USD -> Exceeds 10.0 USD budget
    cb.record_spend(tokens=5000, actual_cost_usd=10.5)

    assert cb.state == CircuitState.OPEN
    assert cb.current_tier() == BudgetTier.EXHAUSTED

    allowed, reason = cb.can_execute(estimated_cost_usd=0.1)
    assert allowed is False
    assert "Circuit OPEN" in reason


def test_half_open_probe_and_reset():
    cb = BudgetCircuitBreaker(
        tenant_id="tenant-probe",
        budget_limit_usd=10.0,
        cooldown_base_seconds=0.05,
        half_open_probe_limit=2,
    )

    cb.record_spend(tokens=1000, actual_cost_usd=10.5)
    assert cb.state == CircuitState.OPEN

    # Sleep past short cooldown
    time.sleep(0.06)

    # Next check transitions to HALF_OPEN
    allowed, reason = cb.can_execute(estimated_cost_usd=0.1)
    assert allowed is True
    assert "HALF_OPEN" in reason
    assert cb.state == CircuitState.HALF_OPEN

    # Reset budget -> transitions cleanly back to CLOSED
    cb.reset_budget(new_limit_usd=50.0)
    assert cb.state == CircuitState.CLOSED
    assert cb.spent_usd == 0.0
    assert cb.budget_limit_usd == 50.0


def test_invalid_parameters():
    with pytest.raises(ValueError, match="budget_limit_usd must be positive"):
        BudgetCircuitBreaker(tenant_id="bad", budget_limit_usd=-5.0)

    cb = BudgetCircuitBreaker(tenant_id="test", budget_limit_usd=10.0)
    with pytest.raises(ValueError, match="new_limit_usd must be positive"):
        cb.reset_budget(new_limit_usd=0.0)
