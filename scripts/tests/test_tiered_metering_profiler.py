import pytest
from scripts.tiered_metering_profiler import (
    RateTier,
    TieredMeteringProfiler,
)


def test_calculate_cost_tiers():
    tiers = [
        RateTier(threshold_tokens=1_000_000, price_per_million=5.0),
        RateTier(threshold_tokens=10_000_000, price_per_million=3.5),
        RateTier(threshold_tokens=100_000_000, price_per_million=2.0),
    ]
    profiler = TieredMeteringProfiler(tiers)

    # 0 tokens -> 0.0
    assert profiler.calculate_cost(0) == 0.0

    # 500k tokens in tier 1 (at $5/M) -> $2.50
    assert profiler.calculate_cost(500_000) == 2.50

    # 5M tokens in tier 2 (at $3.5/M) -> $17.50
    assert profiler.calculate_cost(5_000_000) == 17.50

    # 50M tokens in tier 3 (at $2.0/M) -> $100.00
    assert profiler.calculate_cost(50_000_000) == 100.00


def test_empty_tiers_raises():
    with pytest.raises(ValueError):
        TieredMeteringProfiler([])


def test_benchmark_profiler_quantiles():
    tiers = [
        RateTier(threshold_tokens=1_000_000, price_per_million=5.0),
        RateTier(threshold_tokens=10_000_000, price_per_million=3.0),
    ]
    profiler = TieredMeteringProfiler(tiers)
    volumes = [1000, 50000, 2000000, 15000000]

    res = profiler.run_benchmark(volumes, iterations=30)
    assert res.tier_count == 2
    assert res.batch_size == 4
    assert res.p50_microseconds >= 0.0
    assert res.p90_microseconds >= res.p50_microseconds
    assert res.throughput_ops_sec > 0.0
