"""Tests for the Multi-Tenant Budget Policy Ingress Contract & Schema Evolution Engine."""

import pytest
from scripts.budget_policy_contract import (
    CURRENT_SCHEMA_VERSION,
    BudgetPeriod,
    HardLimitAction,
    SoftLimitAction,
    compute_policy_hash,
    parse_and_validate_budget_policy,
)


def test_valid_v2_budget_policy() -> None:
    raw = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "tenant_id": "tenant-payments-prod",
        "period": "monthly",
        "hard_limit_usd": 5000.0,
        "soft_limit_usd": 4000.0,
        "on_soft_limit": "divert_to_fallback_model",
        "on_hard_limit": "reject",
        "allowed_models": ["gpt-4o", "claude-3-5-sonnet", "gemini-1.5-pro"],
        "fallback_model": "gpt-4o-mini",
        "max_tokens_per_request": 32768,
    }

    policy = parse_and_validate_budget_policy(raw)
    assert policy.schema_version == CURRENT_SCHEMA_VERSION
    assert policy.tenant_id == "tenant-payments-prod"
    assert policy.period == BudgetPeriod.MONTHLY
    assert policy.hard_limit_usd == 5000.0
    assert policy.soft_limit_usd == 4000.0
    assert policy.on_soft_limit == SoftLimitAction.DIVERT_FALLBACK
    assert policy.fallback_model == "gpt-4o-mini"
    assert len(policy.policy_hash) == 64


def test_legacy_v1_schema_evolution() -> None:
    raw_legacy = {
        "schema_version": "1.0.0",
        "tenant_id": "legacy-analytics-team",
        "budget_usd": 1000.0,
        "allowed_models": ["claude-3-haiku"],
    }

    policy = parse_and_validate_budget_policy(raw_legacy)
    # Automatically upgraded to v2.0.0
    assert policy.schema_version == CURRENT_SCHEMA_VERSION
    assert policy.hard_limit_usd == 1000.0
    # Soft limit auto-calculated at 80%
    assert policy.soft_limit_usd == 800.0
    assert policy.on_soft_limit == SoftLimitAction.WARN
    assert policy.on_hard_limit == HardLimitAction.REJECT


def test_budget_policy_validation_errors() -> None:
    # Unsupported version
    with pytest.raises(ValueError, match="Unsupported budget policy schema_version"):
        parse_and_validate_budget_policy({
            "schema_version": "9.9.9",
            "tenant_id": "test",
            "hard_limit_usd": 100,
            "soft_limit_usd": 80,
            "allowed_models": ["gpt-4o"],
        })

    # Invalid tenant_id (uppercase / illegal chars)
    with pytest.raises(ValueError, match="Invalid tenant_id format"):
        parse_and_validate_budget_policy({
            "schema_version": CURRENT_SCHEMA_VERSION,
            "tenant_id": "INVALID_TENANT_ID!",
            "hard_limit_usd": 100,
            "soft_limit_usd": 80,
            "allowed_models": ["gpt-4o"],
        })

    # Soft limit > Hard limit
    with pytest.raises(ValueError, match="soft_limit_usd .* cannot exceed hard_limit_usd"):
        parse_and_validate_budget_policy({
            "schema_version": CURRENT_SCHEMA_VERSION,
            "tenant_id": "tenant-test",
            "hard_limit_usd": 100,
            "soft_limit_usd": 150,
            "allowed_models": ["gpt-4o"],
        })

    # Missing fallback model when action is divert_to_fallback_model
    with pytest.raises(ValueError, match="fallback_model must be specified"):
        parse_and_validate_budget_policy({
            "schema_version": CURRENT_SCHEMA_VERSION,
            "tenant_id": "tenant-test",
            "hard_limit_usd": 100,
            "soft_limit_usd": 80,
            "on_soft_limit": "divert_to_fallback_model",
            "allowed_models": ["gpt-4o"],
        })

    # Empty allowed_models
    with pytest.raises(ValueError, match="allowed_models must be a non-empty list"):
        parse_and_validate_budget_policy({
            "schema_version": CURRENT_SCHEMA_VERSION,
            "tenant_id": "tenant-test",
            "hard_limit_usd": 100,
            "soft_limit_usd": 80,
            "allowed_models": [],
        })


def test_compute_policy_hash_deterministic() -> None:
    p1 = {
        "tenant_id": "tenant-alpha",
        "period": "monthly",
        "hard_limit_usd": 2000.0,
        "soft_limit_usd": 1500.0,
        "on_soft_limit": "warn",
        "on_hard_limit": "reject",
        "allowed_models": ["model-a", "model-b"],
    }

    # Model order permuted
    p2 = {
        "tenant_id": "tenant-alpha",
        "period": "monthly",
        "hard_limit_usd": 2000.0,
        "soft_limit_usd": 1500.0,
        "on_soft_limit": "warn",
        "on_hard_limit": "reject",
        "allowed_models": ["model-b", "model-a"],
    }

    hash1 = compute_policy_hash(p1)
    hash2 = compute_policy_hash(p2)
    assert hash1 == hash2
