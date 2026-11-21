"""Unit tests for multi-model PriceNormalizer and tiered pricing schedules."""

from __future__ import annotations

import pytest
from scripts.price_normalizer import (
    InvalidTokenCountError,
    ModelProfile,
    PriceNormalizer,
    PricingTier,
    UnknownModelError,
)


def test_price_normalizer_validation() -> None:
    normalizer = PriceNormalizer()

    # Unknown model error
    with pytest.raises(UnknownModelError, match="not registered"):
        normalizer.calculate_cost("unknown-model-xyz", 1000, 500)

    # Negative token count error
    with pytest.raises(InvalidTokenCountError, match="non-negative integer"):
        normalizer.calculate_cost("gpt-4o", -10, 500)

    with pytest.raises(InvalidTokenCountError, match="non-negative integer"):
        normalizer.calculate_cost("gpt-4o", 1000, -5)


def test_price_normalizer_baseline_calculation() -> None:
    normalizer = PriceNormalizer()

    # 10,000 prompt tokens and 2,000 completion tokens on GPT-4o
    # Base: prompt = $2.50 / 1M, completion = $10.00 / 1M
    # Prompt cost: (10,000 / 1,000,000) * 2.50 = 0.025
    # Completion cost: (2,000 / 1,000,000) * 10.00 = 0.020
    # Total = 0.045
    cost = normalizer.calculate_cost("gpt-4o", prompt_tokens=10_000, completion_tokens=2_000)
    assert pytest.approx(cost.prompt_cost, rel=1e-5) == 0.025
    assert pytest.approx(cost.completion_cost, rel=1e-5) == 0.020
    assert pytest.approx(cost.total_cost, rel=1e-5) == 0.045
    assert cost.applied_tier_index == 0


def test_price_normalizer_caching_rates() -> None:
    normalizer = PriceNormalizer()

    # Claude 3.5 Sonnet with prompt caching:
    # 5,000 prompt, 1,000 completion, 20,000 cache read, 10,000 cache write
    cost = normalizer.calculate_cost(
        "claude-3-5-sonnet",
        prompt_tokens=5_000,
        completion_tokens=1_000,
        cache_read_tokens=20_000,
        cache_write_tokens=10_000,
    )
    # prompt: 0.005 * 3.00 = 0.015
    # completion: 0.001 * 15.00 = 0.015
    # cache read: 0.020 * 0.30 = 0.006
    # cache write: 0.010 * 3.75 = 0.0375
    # total = 0.015 + 0.015 + 0.006 + 0.0375 = 0.0735
    assert pytest.approx(cost.prompt_cost, rel=1e-5) == 0.015
    assert pytest.approx(cost.completion_cost, rel=1e-5) == 0.015
    assert pytest.approx(cost.cache_read_cost, rel=1e-5) == 0.006
    assert pytest.approx(cost.cache_write_cost, rel=1e-5) == 0.0375
    assert pytest.approx(cost.total_cost, rel=1e-5) == 0.0735


def test_price_normalizer_tiered_volume_discount() -> None:
    normalizer = PriceNormalizer()

    # Baseline tier (< 10M tokens) on GPT-4o
    base_cost = normalizer.calculate_cost("gpt-4o", 1_000_000, 1_000_000, cumulative_tokens=5_000_000)
    assert base_cost.applied_tier_index == 0
    assert pytest.approx(base_cost.total_cost, rel=1e-5) == 12.50  # 2.50 + 10.00

    # High volume tier (>= 10M tokens) on GPT-4o: prompt $2.25, completion $9.00
    tier_cost = normalizer.calculate_cost("gpt-4o", 1_000_000, 1_000_000, cumulative_tokens=15_000_000)
    assert tier_cost.applied_tier_index == 1
    assert pytest.approx(tier_cost.total_cost, rel=1e-5) == 11.25  # 2.25 + 9.00


def test_price_normalizer_model_comparison_ranking() -> None:
    normalizer = PriceNormalizer()

    # 100k prompt, 20k completion
    comparisons = normalizer.compare_models(prompt_tokens=100_000, completion_tokens=20_000)
    assert len(comparisons) == 3

    # Lowest cost model should be first (gemini-1-5-pro)
    assert comparisons[0]["model_id"] == "gemini-1-5-pro"
    assert comparisons[0]["total_cost"] < comparisons[1]["total_cost"]
    assert comparisons[1]["total_cost"] <= comparisons[2]["total_cost"]


def test_price_normalizer_custom_model_registration() -> None:
    normalizer = PriceNormalizer()
    custom_profile = ModelProfile(
        model_id="mistral-large-2407",
        provider="mistral",
        tiers=[
            PricingTier(min_cumulative_tokens=0, prompt_per_m=2.00, completion_per_m=6.00),
        ],
    )
    normalizer.register_model(custom_profile)

    cost = normalizer.calculate_cost("mistral-large-2407", 500_000, 100_000)
    # (0.5 * 2.00) + (0.1 * 6.00) = 1.00 + 0.60 = 1.60
    assert pytest.approx(cost.total_cost, rel=1e-5) == 1.60
