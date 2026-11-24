"""Tests for UsageMeterContract and token ingress invariants."""

import pytest
from scripts.usage_meter_contract import (
    UsageContractError,
    UsageContractLimits,
    UsageMeterContract,
)


def test_valid_nominal_record():
    contract = UsageMeterContract()
    raw = {
        "tenant_id": "tenant-alpha_prod",
        "request_id": "req-9872134098-abc",
        "model_name": "openai/gpt-4o",
        "prompt_tokens": 1500,
        "completion_tokens": 500,
        "cached_prompt_tokens": 1000,
        "tariff_tier": "priority",
        "cost_cents": 1.25,
        "metadata": {
            "environment": "production",
            "region": "us-east-1",
            "__proto__": "polluted",
        },
    }

    rec = contract.validate_record(raw)
    assert rec.tenant_id == "tenant-alpha_prod"
    assert rec.request_id == "req-9872134098-abc"
    assert rec.model_name == "openai/gpt-4o"
    assert rec.prompt_tokens == 1500
    assert rec.completion_tokens == 500
    assert rec.cached_prompt_tokens == 1000
    assert rec.total_tokens == 2000
    assert rec.tariff_tier == "priority"
    assert rec.cost_cents == 1.25
    assert "__proto__" not in rec.metadata
    assert rec.metadata["environment"] == "production"


def test_cached_tokens_cannot_exceed_prompt():
    contract = UsageMeterContract()
    raw = {
        "tenant_id": "tenant-a",
        "request_id": "req-12345678",
        "model_name": "anthropic/claude-3-5-sonnet",
        "prompt_tokens": 500,
        "completion_tokens": 200,
        "cached_prompt_tokens": 600,  # Invalid: 600 > 500!
    }

    with pytest.raises(UsageContractError) as exc_info:
        contract.validate_record(raw)
    assert exc_info.value.code == "CACHED_TOKENS_EXCEED_PROMPT"


def test_tenant_id_validation():
    contract = UsageMeterContract()
    invalid_tenants = ["", "ab", "Tenant Capital", "tenant;inject", "tenant/path"]
    for t_id in invalid_tenants:
        raw = {
            "tenant_id": t_id,
            "request_id": "req-12345678",
            "model_name": "meta/llama-3",
            "prompt_tokens": 100,
            "completion_tokens": 50,
        }
        with pytest.raises(UsageContractError) as exc_info:
            contract.validate_record(raw)
        assert exc_info.value.code == "INVALID_TENANT_ID"


def test_negative_and_non_finite_tokens():
    contract = UsageMeterContract()
    # Negative tokens
    with pytest.raises(UsageContractError) as exc:
        contract.validate_record({
            "tenant_id": "tenant-ok",
            "request_id": "req-12345678",
            "model_name": "gpt-4o",
            "prompt_tokens": -5,
            "completion_tokens": 10,
        })
    assert exc.value.code == "NEGATIVE_TOKENS"

    # Non-integer tokens
    with pytest.raises(UsageContractError) as exc:
        contract.validate_record({
            "tenant_id": "tenant-ok",
            "request_id": "req-12345678",
            "model_name": "gpt-4o",
            "prompt_tokens": "invalid-num",
            "completion_tokens": 10,
        })
    assert exc.value.code == "INVALID_INTEGER"


def test_cost_ceiling_and_negative_cost():
    contract = UsageMeterContract(UsageContractLimits(max_cost_cents_per_request=500.0))

    # Negative cost
    with pytest.raises(UsageContractError) as exc:
        contract.validate_record({
            "tenant_id": "tenant-ok",
            "request_id": "req-12345678",
            "model_name": "gpt-4o",
            "prompt_tokens": 10,
            "completion_tokens": 10,
            "cost_cents": -1.0,
        })
    assert exc.value.code == "NEGATIVE_COST"

    # Exceeds cost ceiling
    with pytest.raises(UsageContractError) as exc:
        contract.validate_record({
            "tenant_id": "tenant-ok",
            "request_id": "req-12345678",
            "model_name": "gpt-4o",
            "prompt_tokens": 10,
            "completion_tokens": 10,
            "cost_cents": 501.0,
        })
    assert exc.value.code == "COST_CEILING_BREACH"


def test_batch_validation_success_and_deduplication():
    contract = UsageMeterContract()

    batch = [
        {
            "tenant_id": "tenant-1",
            "request_id": "req-00000001",
            "model_name": "gpt-4o",
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "cached_prompt_tokens": 20,
            "cost_cents": 0.5,
        },
        {
            "tenant_id": "tenant-2",
            "request_id": "req-00000002",
            "model_name": "gpt-4o",
            "prompt_tokens": 200,
            "completion_tokens": 100,
            "cached_prompt_tokens": 50,
            "cost_cents": 1.0,
        },
    ]

    res = contract.validate_batch(batch)
    assert res.valid is True
    assert res.record_count == 2
    assert res.total_prompt_tokens == 300
    assert res.total_completion_tokens == 150
    assert res.total_cached_tokens == 70
    assert res.total_cost_cents == 1.5

    # Duplicate request_id in same batch fails
    duplicate_batch = [batch[0], batch[0]]
    with pytest.raises(UsageContractError) as exc:
        contract.validate_batch(duplicate_batch)
    assert exc.value.code == "DUPLICATE_REQUEST_ID_IN_BATCH"
