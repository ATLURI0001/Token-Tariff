"""
Unit and regression test suite for Tariff Benchmark Profiler.
"""

import pytest
from scripts.tariff_benchmark_profiler import (
    BenchmarkProfileResult,
    CostCalculationResult,
    TariffBenchmarkProfiler,
    TariffSchedule,
    TariffTier,
    TokenWorkload,
)


@pytest.fixture
def standard_schedule() -> TariffSchedule:
    # Input tiers:
    # 0 to 100,000 tokens: $3.00 / M ($3,000,000,000 nanos / M)
    # 100,000+ tokens: $2.00 / M ($2,000,000,000 nanos / M)
    input_tiers = [
        TariffTier(min_tokens=0, max_tokens=100_000, rate_per_m_nanos=3_000_000_000),
        TariffTier(min_tokens=100_000, max_tokens=None, rate_per_m_nanos=2_000_000_000),
    ]

    # Output tiers: flat $15.00 / M ($15,000,000,000 nanos / M)
    output_tiers = [
        TariffTier(min_tokens=0, max_tokens=None, rate_per_m_nanos=15_000_000_000),
    ]

    return TariffSchedule(
        schedule_id="claude-standard-tiered",
        input_tiers=input_tiers,
        output_tiers=output_tiers,
        cache_discount_fraction=0.50,
    )


def test_calculate_tiered_cost_single_tier(standard_schedule):
    profiler = TariffBenchmarkProfiler()

    # 50,000 input tokens (all in Tier 1: $3/M) -> 50,000 * 3,000 = 150,000,000 nanos ($0.15)
    # 10,000 output tokens ($15/M) -> 10,000 * 15,000 = 150,000,000 nanos ($0.15)
    workload = TokenWorkload(input_tokens=50_000, output_tokens=10_000, cached_tokens=0)
    res = profiler.calculate_tiered_cost(standard_schedule, workload)

    assert res.input_cost_nanos == 150_000_000
    assert res.output_cost_nanos == 150_000_000
    assert res.total_cost_nanos == 300_000_000
    assert res.cache_savings_nanos == 0


def test_calculate_tiered_cost_multi_tier_spanning(standard_schedule):
    profiler = TariffBenchmarkProfiler()

    # 150,000 input tokens:
    # First 100k at $3/M = 300,000,000 nanos
    # Remaining 50k at $2/M = 100,000,000 nanos
    # Total input: 400,000,000 nanos
    workload = TokenWorkload(input_tokens=150_000, output_tokens=0, cached_tokens=0)
    res = profiler.calculate_tiered_cost(standard_schedule, workload)

    assert res.input_cost_nanos == 400_000_000
    assert res.total_cost_nanos == 400_000_000


def test_calculate_tiered_cost_cache_discount(standard_schedule):
    profiler = TariffBenchmarkProfiler()

    # 40,000 input tokens total, of which 20,000 are cached (50% discount)
    # Fresh input: 20k * $3/M = 60,000,000 nanos
    # Cached input raw: 20k * $3/M = 60,000,000 nanos, with 50% discount -> 30,000,000 nanos
    # Total input: 60M + 30M = 90M nanos ($0.09)
    # Savings: 30M nanos ($0.03)
    workload = TokenWorkload(input_tokens=40_000, output_tokens=0, cached_tokens=20_000)
    res = profiler.calculate_tiered_cost(standard_schedule, workload)

    assert res.cache_savings_nanos == 30_000_000
    assert res.input_cost_nanos == 90_000_000
    assert res.total_cost_nanos == 90_000_000


def test_input_validation_and_edge_cases(standard_schedule):
    profiler = TariffBenchmarkProfiler()

    with pytest.raises(TypeError):
        profiler.calculate_tiered_cost(None, TokenWorkload(10, 10))

    with pytest.raises(TypeError):
        profiler.calculate_tiered_cost(standard_schedule, None)

    # Zero tokens
    zero_res = profiler.calculate_tiered_cost(standard_schedule, TokenWorkload(0, 0))
    assert zero_res.total_cost_nanos == 0

    # Negative tokens safely clamped to 0
    neg_res = profiler.calculate_tiered_cost(standard_schedule, TokenWorkload(-50, -20))
    assert neg_res.total_cost_nanos == 0


def test_run_workload_benchmark(standard_schedule):
    profiler = TariffBenchmarkProfiler()

    workloads = [
        TokenWorkload(input_tokens=1000 + i * 100, output_tokens=200 + i * 20, cached_tokens=100)
        for i in range(25)
    ]

    res = profiler.run_workload_benchmark("Test-Batch-25", standard_schedule, workloads, iterations=20)

    assert res.workload_name == "Test-Batch-25"
    assert res.workload_count == 25
    assert res.iterations == 20
    assert res.throughput_ops_sec > 0
    assert res.p50_latency_us > 0
    assert res.p99_latency_us >= res.p50_latency_us


def test_generate_matrix_and_markdown(standard_schedule):
    profiler = TariffBenchmarkProfiler()

    results = profiler.generate_benchmark_matrix(standard_schedule, iterations=15)
    assert len(results) == 2

    markdown = profiler.format_markdown_report(results)
    assert "Token Tariff Execution & Multi-Tier Pricing Benchmark Matrix" in markdown
    assert "Batch-100 (Micro)" in markdown
    assert "Batch-1000 (Standard)" in markdown
    assert "ops/s" in markdown
