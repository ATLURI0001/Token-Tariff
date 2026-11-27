"""Usage ledger summation throughput profiler."""
import time


def benchmark_ledger_sum(entries: list[int], iterations: int = 500) -> float:
    start = time.perf_counter()
    for _ in range(iterations):
        _ = sum(entries)
    elapsed = max(time.perf_counter() - start, 1e-9)
    return round((iterations * len(entries)) / elapsed, 2)
