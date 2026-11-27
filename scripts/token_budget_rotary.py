"""token_budget_rotary.py – bounded ring buffer for token budget events."""
from collections import deque


class TokenBudgetRotary:
    def __init__(self, max_size: int = 20):
        self._buf: deque = deque(maxlen=max_size)

    def push(self, token_count: int) -> None:
        self._buf.append(token_count)

    def total(self) -> int:
        return sum(self._buf)

    def count(self) -> int:
        return len(self._buf)
