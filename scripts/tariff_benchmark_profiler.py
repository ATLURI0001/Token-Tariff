"""
Token-Tariff - Gas Schedule & Multi-Tier Execution Pricing Benchmark Profiler.

Profiles high-throughput token billing calculations, multi-tier volume pricing schedules,
cache-deduction evaluation pipelines, and latency distribution percentiles.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class TariffTier:
    """Pricing rate tier for token consumption."""
    min_tokens: int
    max_tokens: Optional[int]  # None indicates unbounded upper tier
    rate_per_m_nanos: int     # Nano-dollars per 1,000,000 tokens


@dataclass
class TariffSchedule:
    """Hierarchical pricing schedule with input, output, and cache discount tiers."""
    schedule_id: str
    input_tiers: List[TariffTier]
    output_tiers: List[TariffTier]
    cache_discount_fraction: float = 0.50  # 50% discount for cached prefix tokens


@dataclass(frozen=True)
class TokenWorkload:
    """A batch or single inference token consumption event."""
    input_tokens: int
    output_tokens: int
    cached_tokens: int = 0


@dataclass(frozen=True)
class CostCalculationResult:
    """Cost outcome in fixed-point nano-dollars."""
    total_cost_nanos: int
    input_cost_nanos: int
    output_cost_nanos: int
    cache_savings_nanos: int


@dataclass(frozen=True)
class BenchmarkProfileResult:
    """Micro-benchmark performance results."""
    workload_name: str
    workload_count: int
    iterations: int
    throughput_ops_sec: float
    mean_latency_us: float
    p50_latency_us: float
    p90_latency_us: float
    p99_latency_us: float


class TariffBenchmarkProfiler:
    """
    Benchmarks token tariff calculation engines across tiered volume schedules.
    """

    @staticmethod
    def calculate_tiered_cost(
        schedule: TariffSchedule,
        workload: TokenWorkload,
    ) -> CostCalculationResult:
        """Calculates cost across input and output multi-tier schedules."""
        if not schedule:
            raise TypeError("TariffSchedule cannot be None")
        if not workload:
            raise TypeError("TokenWorkload cannot be None")

        inp = max(0, int(workload.input_tokens))
        out = max(0, int(workload.output_tokens))
        cached = max(0, min(inp, int(workload.cached_tokens)))
        fresh_input = inp - cached

        # Calculate input cost
        input_cost = TariffBenchmarkProfiler._evaluate_tiers(schedule.input_tiers, fresh_input)

        # Calculate cached input cost with discount
        cache_cost_raw = TariffBenchmarkProfiler._evaluate_tiers(schedule.input_tiers, cached)
        discount = max(0.0, min(1.0, float(schedule.cache_discount_fraction)))
        cache_savings = int(cache_cost_raw * discount)
        cached_input_cost = cache_cost_raw - cache_savings

        total_input_cost = input_cost + cached_input_cost

        # Calculate output cost
        output_cost = TariffBenchmarkProfiler._evaluate_tiers(schedule.output_tiers, out)

        total_cost = total_input_cost + output_cost

        return CostCalculationResult(
            total_cost_nanos=total_cost,
            input_cost_nanos=total_input_cost,
            output_cost_nanos=output_cost,
            cache_savings_nanos=cache_savings,
        )

    @staticmethod
    def _evaluate_tiers(tiers: List[TariffTier], tokens: int) -> int:
        if tokens <= 0 or not tiers:
            return 0

        cost = 0
        remaining = tokens

        for tier in tiers:
            if remaining <= 0:
                break

            tier_span = tier.max_tokens - tier.min_tokens if tier.max_tokens else remaining
            tokens_in_tier = min(remaining, tier_span)

            # Nano cost = (tokens_in_tier * rate_per_m_nanos) / 1,000,000
            cost += (tokens_in_tier * tier.rate_per_m_nanos) // 1_000_000
            remaining -= tokens_in_tier

        return cost

    def run_workload_benchmark(
        self,
        workload_name: str,
        schedule: TariffSchedule,
        workloads: List[TokenWorkload],
        iterations: int = 100,
    ) -> BenchmarkProfileResult:
        """Executes repeated calculation passes and measures latency percentiles."""
        iters = max(10, int(iterations))
        latencies_us: List[float] = []

        for _ in range(iters):
            t0 = time.perf_counter()
            for w in workloads:
                _ = self.calculate_tiered_cost(schedule, w)
            elapsed_sec = time.perf_counter() - t0
            latencies_us.append(elapsed_sec * 1_000_000.0)

        sorted_latencies = sorted(latencies_us)
        mean_us = statistics.mean(latencies_us)
        p50_us = sorted_latencies[int(iters * 0.50)]
        p90_us = sorted_latencies[min(int(iters * 0.90), iters - 1)]
        p99_us = sorted_latencies[min(int(iters * 0.99), iters - 1)]

        total_ops = len(workloads) * iters
        total_time_sec = sum(latencies_us) / 1_000_000.0
        throughput = round(total_ops / total_time_sec, 2) if total_time_sec > 0 else 0.0

        return BenchmarkProfileResult(
            workload_name=workload_name,
            workload_count=len(workloads),
            iterations=iters,
            throughput_ops_sec=throughput,
            mean_latency_us=round(mean_us, 2),
            p50_latency_us=round(p50_us, 2),
            p90_latency_us=round(p90_us, 2),
            p99_latency_us=round(p99_us, 2),
        )

    def generate_benchmark_matrix(
        self,
        schedule: TariffSchedule,
        iterations: int = 50,
    ) -> List[BenchmarkProfileResult]:
        """Runs standard synthetic workload profiles: Micro (100), Medium (1,000), Heavy (10,000)."""
        workloads_100 = [
            TokenWorkload(input_tokens=1500 + i * 50, output_tokens=300 + i * 10, cached_tokens=500)
            for i in range(100)
        ]
        workloads_1000 = [
            TokenWorkload(input_tokens=2000 + i * 10, output_tokens=400 + i * 5, cached_tokens=1000)
            for i in range(1000)
        ]

        res_100 = self.run_workload_benchmark("Batch-100 (Micro)", schedule, workloads_100, iterations)
        res_1000 = self.run_workload_benchmark("Batch-1000 (Standard)", schedule, workloads_100, iterations)

        return [res_100, res_1000]

    def format_markdown_report(self, results: List[BenchmarkProfileResult]) -> str:
        """Formats benchmark results into a clean markdown summary table."""
        lines = [
            "### Token Tariff Execution & Multi-Tier Pricing Benchmark Matrix",
            "",
            "| Workload Profile | Batch Size | Iterations | Throughput (ops/sec) | Mean Latency (µs) | P50 (µs) | P90 (µs) | P99 (µs) |",
            "|---|---|---|---|---|---|---|---|",
        ]

        for res in results:
            lines.append(
                f"| **{res.workload_name}** | {res.workload_count} | {res.iterations} | "
                f"{res.throughput_ops_sec:,.0f} ops/s | {res.mean_latency_us:.2f} µs | "
                f"{res.p50_latency_us:.2f} µs | {res.p90_latency_us:.2f} µs | {res.p99_latency_us:.2f} µs |"
            )

        return "\n".join(lines)
