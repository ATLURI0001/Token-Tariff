"""Bounded ring buffer for tenant quota overage counts."""
from collections import deque


class OverageCountRotary:
    def __init__(self, capacity: int = 32) -> None:
        self._samples: deque[int] = deque(maxlen=capacity)

    def record(self, count: int) -> None:
        if count < 0:
            raise ValueError("count must be non-negative")
        self._samples.append(count)

    def avg_count(self) -> float:
        return sum(self._samples) / len(self._samples) if self._samples else 0.0
