"""
Unit tests for Pricing Quantization Sentinel (INC-2026-TARIFF-4421).
"""

import pytest

from scripts.pricing_quantization_sentinel import (
    PricingQuantizationSentinel,
    QuantizationViolationError,
    QuantizedBillingResult,
)


def test_nominal_quantization():
    sentinel = PricingQuantizationSentinel()
    # 250 tokens @ $0.002 / 1k = $0.0005 exactly
    res = sentinel.calculate_billable_cost("gpt-4o", 250, 0.002)

    assert res.model_name == "gpt-4o"
    assert res.token_count == 250
    assert res.raw_cost_usd == 0.0005
    assert res.quantized_cost_usd >= res.raw_cost_usd
    assert res.micro_cents == 50000


def test_ceiling_rounding_prevents_underbilling():
    sentinel = PricingQuantizationSentinel()
    # 333 tokens @ $0.0015 / 1k = $0.0004995 -> ceiled to micro-cents
    res = sentinel.calculate_billable_cost("claude-sonnet", 333, 0.0015)
    assert res.quantized_cost_usd >= res.raw_cost_usd
    assert res.discrepancy_usd >= 0.0


def test_defensive_validation():
    sentinel = PricingQuantizationSentinel()

    with pytest.raises(ValueError):
        sentinel.calculate_billable_cost("", 100, 0.001)

    with pytest.raises(ValueError):
        sentinel.calculate_billable_cost("model", 0, 0.001)

    with pytest.raises(ValueError):
        sentinel.calculate_billable_cost("model", 100, -0.05)


def test_postmortem_export():
    sentinel = PricingQuantizationSentinel()
    pm = sentinel.export_postmortem()
    assert pm["incident_id"] == "INC-2026-TARIFF-4421"
    assert pm["total_incidents"] == 0
