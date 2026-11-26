"""Tenant Quota Overage & Burn-Rate Telemetry Rotary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class QuotaBurnSample:
    tenant_id: str
    tokens_used: int
    quota_limit: int


class OverageTelemetryRotary:
    def __init__(self, capacity: int = 256) -> None:
        self.capacity = max(1, capacity)
        self.buffer: List[Optional[QuotaBurnSample]] = [None] * self.capacity
        self.head = 0
        self.count = 0

    def record(self, tenant_id: str, tokens_used: int, quota_limit: int) -> None:
        self.buffer[self.head] = QuotaBurnSample(tenant_id, tokens_used, quota_limit)
        self.head = (self.head + 1) % self.capacity
        if self.count < self.capacity:
            self.count += 1

    def get_overage_rate_pct(self) -> float:
        if self.count == 0:
            return 0.0
        overages = sum(1 for i in range(self.count) if self.buffer[i] and self.buffer[i].tokens_used > self.buffer[i].quota_limit)
        return round((overages / self.count) * 100.0, 2)
