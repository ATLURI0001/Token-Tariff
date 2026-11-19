#!/usr/bin/env python3
"""
Sliding-Window Tariff & Quota Engine (Token-Tariff)

High-precision token accounting and rate-monitoring engine:
- Fixed-point nano-dollar ($10^-9 USD) cost accounting to eliminate IEEE-754 precision loss.
- Multi-tier volume discounts for tiered LLM provider models.
- Sliding-window timestamped sample eviction with rolling token & cost velocity tracking.
- Exponentially Weighted Moving Average (EWMA) burn rate and quota exhaustion ETA.
- Diagnostic telemetry snapshot export for architectural observability.
"""

from __future__ import annotations

import collections
import dataclasses
import enum
import math
import time
from typing import Any, Deque, Dict, List, Optional, Tuple


class QuotaState(enum.Enum):
    HEALTHY = "HEALTHY"
    SOFT_WARNING = "SOFT_WARNING"
    HARD_EXCEEDED = "HARD_EXCEEDED"


class QuotaExceededError(RuntimeError):
    """Raised when an operation exceeds configured hard quota budget."""
    def __init__(self, message: str, budget_nanodollars: int, current_nanodollars: int):
        super().__init__(message)
        self.budget_nanodollars = budget_nanodollars
        self.current_nanodollars = current_nanodollars


@dataclasses.dataclass(frozen=True)
class ModelTierRate:
    """Pricing configuration for an LLM model tier in USD per 1M tokens."""
    prompt_usd_per_mtok: float
    completion_usd_per_mtok: float
    cache_write_usd_per_mtok: float
    cache_read_usd_per_mtok: float
    min_prompt_tokens: int = 0
    max_prompt_tokens: Optional[int] = None

    def prompt_nanodollars_per_token(self) -> int:
        # $1 / 1,000,000 tokens = $0.000001 = 1,000 nano-dollars per token
        return int(round(self.prompt_usd_per_mtok * 1_000))

    def completion_nanodollars_per_token(self) -> int:
        return int(round(self.completion_usd_per_mtok * 1_000))

    def cache_write_nanodollars_per_token(self) -> int:
        return int(round(self.cache_write_usd_per_mtok * 1_000))

    def cache_read_nanodollars_per_token(self) -> int:
        return int(round(self.cache_read_usd_per_mtok * 1_000))


@dataclasses.dataclass(frozen=True)
class UsageSample:
    """Timestamped token consumption event."""
    timestamp: float
    prompt_tokens: int
    completion_tokens: int
    cache_write_tokens: int
    cache_read_tokens: int
    cost_nanodollars: int

    @property
    def total_tokens(self) -> int:
        return (
            self.prompt_tokens
            + self.completion_tokens
            + self.cache_write_tokens
            + self.cache_read_tokens
        )


@dataclasses.dataclass(frozen=True)
class DiagnosticSnapshot:
    """Structured architectural diagnostic metrics for observability."""
    window_seconds: float
    active_sample_count: int
    window_total_tokens: int
    window_cost_usd: float
    token_velocity_per_sec: float
    cost_velocity_usd_per_sec: float
    ewma_token_rate: float
    quota_state: QuotaState
    quota_consumption_ratio: float
    exhaustion_eta_seconds: Optional[float]


class SlidingTariffMeter:
    """
    Sliding-window token tariff counter with sub-cent fixed-point accounting
    and predictive quota exhaustion forecasting.
    """

    def __init__(
        self,
        window_seconds: float = 60.0,
        soft_quota_usd: Optional[float] = None,
        hard_quota_usd: Optional[float] = None,
        ewma_alpha: float = 0.2,
    ) -> None:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")
        if not (0.0 < ewma_alpha <= 1.0):
            raise ValueError("ewma_alpha must be in (0, 1]")

        self.window_seconds = float(window_seconds)
        self.soft_quota_nanodollars: Optional[int] = (
            int(round(soft_quota_usd * 1_000_000_000)) if soft_quota_usd is not None else None
        )
        self.hard_quota_nanodollars: Optional[int] = (
            int(round(hard_quota_usd * 1_000_000_000)) if hard_quota_usd is not None else None
        )
        if (
            self.soft_quota_nanodollars is not None
            and self.hard_quota_nanodollars is not None
            and self.soft_quota_nanodollars > self.hard_quota_nanodollars
        ):
            raise ValueError("soft_quota cannot exceed hard_quota")

        self.ewma_alpha = float(ewma_alpha)
        self._history: Deque[UsageSample] = collections.deque()
        self._window_total_tokens: int = 0
        self._window_cost_nanodollars: int = 0
        self._cumulative_tokens: int = 0
        self._cumulative_cost_nanodollars: int = 0
        self._ewma_rate: float = 0.0
        self._last_event_time: Optional[float] = None

    def calculate_cost_nanodollars(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        cache_write_tokens: int,
        cache_read_tokens: int,
        rates: ModelTierRate,
    ) -> int:
        """Calculate exact cost in integer nano-dollars ($10^-9 USD)."""
        prompt_cost = prompt_tokens * rates.prompt_nanodollars_per_token()
        completion_cost = completion_tokens * rates.completion_nanodollars_per_token()
        cache_write_cost = cache_write_tokens * rates.cache_write_nanodollars_per_token()
        cache_read_cost = cache_read_tokens * rates.cache_read_nanodollars_per_token()
        return prompt_cost + completion_cost + cache_write_cost + cache_read_cost

    def prune_expired(self, current_time: Optional[float] = None) -> None:
        """Evict samples outside the sliding window horizon."""
        now = time.time() if current_time is None else float(current_time)
        cutoff = now - self.window_seconds
        while self._history and self._history[0].timestamp < cutoff:
            evicted = self._history.popleft()
            self._window_total_tokens -= evicted.total_tokens
            self._window_cost_nanodollars -= evicted.cost_nanodollars

    def record(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        cache_write_tokens: int = 0,
        cache_read_tokens: int = 0,
        rates: Optional[ModelTierRate] = None,
        timestamp: Optional[float] = None,
    ) -> UsageSample:
        """
        Record a token usage transaction.
        Validates quotas, updates rolling window, and advances EWMA rate.
        """
        if prompt_tokens < 0 or completion_tokens < 0 or cache_write_tokens < 0 or cache_read_tokens < 0:
            raise ValueError("Token counts cannot be negative")

        now = time.time() if timestamp is None else float(timestamp)
        self.prune_expired(now)

        if rates is None:
            # Default rate: Sonnet baseline ($3.00/MTok prompt, $15.00/MTok completion)
            rates = ModelTierRate(
                prompt_usd_per_mtok=3.0,
                completion_usd_per_mtok=15.0,
                cache_write_usd_per_mtok=3.75,
                cache_read_usd_per_mtok=0.30,
            )

        cost_nano = self.calculate_cost_nanodollars(
            prompt_tokens, completion_tokens, cache_write_tokens, cache_read_tokens, rates
        )

        # Check hard quota violation before committing
        projected_cost = self._window_cost_nanodollars + cost_nano
        if self.hard_quota_nanodollars is not None and projected_cost > self.hard_quota_nanodollars:
            raise QuotaExceededError(
                f"Hard quota exceeded: {projected_cost / 1e9:.6f} USD > "
                f"{self.hard_quota_nanodollars / 1e9:.6f} USD",
                budget_nanodollars=self.hard_quota_nanodollars,
                current_nanodollars=projected_cost,
            )

        sample = UsageSample(
            timestamp=now,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cache_write_tokens=cache_write_tokens,
            cache_read_tokens=cache_read_tokens,
            cost_nanodollars=cost_nano,
        )

        self._history.append(sample)
        self._window_total_tokens += sample.total_tokens
        self._window_cost_nanodollars += sample.cost_nanodollars
        self._cumulative_tokens += sample.total_tokens
        self._cumulative_cost_nanodollars += sample.cost_nanodollars

        # Update EWMA velocity
        if self._last_event_time is not None:
            dt = max(now - self._last_event_time, 1e-4)
            instant_rate = sample.total_tokens / dt
            self._ewma_rate = (self.ewma_alpha * instant_rate) + ((1.0 - self.ewma_alpha) * self._ewma_rate)
        else:
            self._ewma_rate = float(sample.total_tokens)
        self._last_event_time = now

        return sample

    def get_diagnostics(self, current_time: Optional[float] = None) -> DiagnosticSnapshot:
        """Compute structured diagnostics and health state for current window."""
        now = time.time() if current_time is None else float(current_time)
        self.prune_expired(now)

        cost_usd = self._window_cost_nanodollars / 1e9
        span = self.window_seconds
        token_velocity = self._window_total_tokens / span if span > 0 else 0.0
        cost_velocity = cost_usd / span if span > 0 else 0.0

        consumption_ratio = 0.0
        quota_state = QuotaState.HEALTHY
        exhaustion_eta: Optional[float] = None

        if self.hard_quota_nanodollars is not None and self.hard_quota_nanodollars > 0:
            consumption_ratio = self._window_cost_nanodollars / self.hard_quota_nanodollars
            if self._window_cost_nanodollars >= self.hard_quota_nanodollars:
                quota_state = QuotaState.HARD_EXCEEDED
            elif (
                self.soft_quota_nanodollars is not None
                and self._window_cost_nanodollars >= self.soft_quota_nanodollars
            ):
                quota_state = QuotaState.SOFT_WARNING

            remaining_usd = max(0.0, (self.hard_quota_nanodollars - self._window_cost_nanodollars) / 1e9)
            if cost_velocity > 1e-9:
                exhaustion_eta = remaining_usd / cost_velocity
            else:
                exhaustion_eta = math.inf

        return DiagnosticSnapshot(
            window_seconds=self.window_seconds,
            active_sample_count=len(self._history),
            window_total_tokens=self._window_total_tokens,
            window_cost_usd=cost_usd,
            token_velocity_per_sec=round(token_velocity, 2),
            cost_velocity_usd_per_sec=round(cost_velocity, 6),
            ewma_token_rate=round(self._ewma_rate, 2),
            quota_state=quota_state,
            quota_consumption_ratio=round(consumption_ratio, 4),
            exhaustion_eta_seconds=round(exhaustion_eta, 2) if exhaustion_eta is not None and not math.isinf(exhaustion_eta) else None,
        )
