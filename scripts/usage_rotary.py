"""Token-Tariff: usage rotary telemetry – bounded ring of usage snapshots."""
from collections import deque


class UsageRotary:
    def __init__(self, capacity: int = 5):
        self._buf: deque = deque(maxlen=capacity)

    def record(self, tokens: int) -> None:
        self._buf.append(tokens)

    def __len__(self) -> int:
        return len(self._buf)
