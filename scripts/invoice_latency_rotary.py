"""Bounded ring buffer for invoice generation latency."""
from collections import deque


class InvoiceLatencyRotary:
    def __init__(self, capacity: int = 32) -> None:
        self._samples: deque[float] = deque(maxlen=capacity)

    def record(self, ms: float) -> None:
        if ms < 0:
            raise ValueError("ms must be non-negative")
        self._samples.append(ms)

    def avg_ms(self) -> float:
        return sum(self._samples) / len(self._samples) if self._samples else 0.0
