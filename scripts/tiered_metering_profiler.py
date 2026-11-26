"""Token Rate Tier Lookup & Tiered Metering Benchmark Profiler.

Benchmarks high-throughput volume pricing tier calculations across LLM prompt
and completion token billing cycles using binary search indexing.
"""

from __future__ import annotations

import bisect
import time
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass(frozen=True)
class RateTier:
    threshold_tokens: int  # Cumulative threshold upper bound
    price_per_million: float


@dataclass
class MeteringBenchmarkResult:
    tier_count: int
    batch_size: int
    p50_microseconds: float
    p90_microseconds: float
    p99_microseconds: float
    throughput_ops_sec: float


class TieredMeteringProfiler:
    """Evaluates tiered billing calculations and benchmarks microsecond latency."""

    def __init__(self, tiers: List[RateTier]) -> None:
        if not tiers:
            raise ValueError("Tiers list cannot be empty")
        # Ensure tiers are sorted by threshold
        self.tiers = sorted(tiers, key=lambda t: t.threshold_tokens)
        self.thresholds = [t.threshold_tokens for t in self.tiers]

    def calculate_cost(self, tokens: int) -> float:
        if tokens <= 0:
            return 0.0

        idx = bisect.bisect_left(self.thresholds, tokens)
        if idx >= len(self.tiers):
            tier = self.tiers[-1]
        else:
            tier = self.tiers[idx]

        return (tokens / 1_000_000.0) * tier.price_per_million

    def run_benchmark(self, token_volumes: List[int], iterations: int = 50) -> MeteringBenchmarkResult:
        if not token_volumes:
            raise ValueError("Token volumes cannot be empty")

        latencies_micros: List[float] = []
        total_evaluations = len(token_volumes) * iterations

        start_total = time.perf_counter()
        for _ in range(iterations):
            start = time.perf_counter()
            for v in token_volumes:
                self.calculate_cost(v)
            elapsed_micros = (time.perf_counter() - start) * 1_000_000.0 / len(token_volumes)
            latencies_micros.append(elapsed_micros)

        total_elapsed_sec = time.perf_counter() - start_total
        latencies_micros.sort()

        p50 = latencies_micros[int(len(latencies_micros) * 0.5)]
        p90 = latencies_micros[int(len(latencies_micros) * 0.9)]
        p99 = latencies_micros[int(len(latencies_micros) * 0.99)]

        throughput = total_evaluations / total_elapsed_sec if total_elapsed_sec > 0 else 0.0

        return MeteringBenchmarkResult(
            tier_count=len(self.tiers),
            batch_size=len(token_volumes),
            p50_microseconds=round(p50, 4),
            p90_microseconds=round(p90, 4),
            p99_microseconds=round(p99, 4),
            throughput_ops_sec=round(throughput, 1),
        )
