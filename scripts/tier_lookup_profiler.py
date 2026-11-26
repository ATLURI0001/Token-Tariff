"""Pricing tier lookup throughput profiler."""
import time


def benchmark_tier_lookup(tiers: dict[str, float], keys: list[str], iterations: int = 500) -> float:
    start = time.perf_counter()
    for _ in range(iterations):
        for k in keys:
            _ = tiers.get(k)
    elapsed = max(time.perf_counter() - start, 1e-9)
    return round((iterations * len(keys)) / elapsed, 2)
