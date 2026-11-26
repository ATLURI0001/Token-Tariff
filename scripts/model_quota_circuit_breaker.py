"""Upstream model rate-limit backoff and sliding quota replenishment circuit breaker.

Provides per-model rate limit protection, sliding token quota replenishment,
half-open canary probing, and automated fallback routing to prevent upstream
exhaustion and cascade failures across multi-provider LLM pipelines.
"""

from __future__ import annotations

import enum
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


class BreakerState(enum.Enum):
    CLOSED = "CLOSED"        # Normal execution
    OPEN = "OPEN"            # Quota exhausted / rate-limited; route to fallback
    HALF_OPEN = "HALF_OPEN"  # Canary probing allowed to verify upstream recovery


class RouteDecision(enum.Enum):
    ALLOW_PRIMARY = "ALLOW_PRIMARY"
    ROUTE_FALLBACK = "ROUTE_FALLBACK"
    SHED_REQUEST = "SHED_REQUEST"


@dataclass
class QuotaBucket:
    capacity_tokens: int
    available_tokens: float
    replenish_rate_tokens_per_sec: float
    last_replenish_time: float


@dataclass
class CircuitMetrics:
    total_requests: int = 0
    primary_allowed: int = 0
    fallback_routed: int = 0
    shed_requests: int = 0
    rate_limit_trips: int = 0
    successful_probes: int = 0
    failed_probes: int = 0


class ModelQuotaCircuitBreaker:
    """Production-grade model rate-limit circuit breaker with sliding quota replenishment."""

    def __init__(
        self,
        model_id: str,
        fallback_model_id: Optional[str] = None,
        max_tokens_per_minute: int = 60000,
        requests_per_minute: int = 600,
        consecutive_rate_limits_to_open: int = 2,
        base_cooldown_sec: float = 10.0,
        max_cooldown_sec: float = 300.0,
        half_open_probe_limit: int = 2,
        jitter_factor: float = 0.2,
        initial_time: Optional[float] = None,
    ) -> None:
        if max_tokens_per_minute <= 0:
            raise ValueError("max_tokens_per_minute must be positive")
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")

        self.model_id = model_id
        self.fallback_model_id = fallback_model_id
        self.consecutive_rate_limits_to_open = consecutive_rate_limits_to_open
        self.base_cooldown_sec = base_cooldown_sec
        self.max_cooldown_sec = max_cooldown_sec
        self.half_open_probe_limit = half_open_probe_limit
        self.jitter_factor = jitter_factor

        now = initial_time if initial_time is not None else time.time()
        self.token_bucket = QuotaBucket(
            capacity_tokens=max_tokens_per_minute,
            available_tokens=float(max_tokens_per_minute),
            replenish_rate_tokens_per_sec=max_tokens_per_minute / 60.0,
            last_replenish_time=now,
        )

        self.state: BreakerState = BreakerState.CLOSED
        self.metrics = CircuitMetrics()
        self._consecutive_rate_limits: int = 0
        self._consecutive_backoff_steps: int = 0
        self._opened_at: Optional[float] = None
        self._cooldown_duration: float = base_cooldown_sec
        self._probes_in_flight: int = 0
        self._successful_canary_count: int = 0

    def _replenish_quota(self, now: float) -> None:
        """Replenishes token quota according to elapsed wall time."""
        if self.token_bucket.last_replenish_time > now:
            # Handle simulated or backward timestamps
            self.token_bucket.last_replenish_time = now
            return

        elapsed = max(0.0, now - self.token_bucket.last_replenish_time)
        if elapsed > 0:
            added = elapsed * self.token_bucket.replenish_rate_tokens_per_sec
            self.token_bucket.available_tokens = min(
                float(self.token_bucket.capacity_tokens),
                self.token_bucket.available_tokens + added,
            )
            self.token_bucket.last_replenish_time = now

    def _calculate_cooldown(self) -> float:
        """Calculates exponential backoff cooldown with jitter."""
        mult = 2 ** min(self._consecutive_backoff_steps, 6)
        raw = min(self.max_cooldown_sec, self.base_cooldown_sec * mult)
        # Jitter: [raw * (1 - jitter), raw * (1 + jitter)]
        jitter_range = raw * self.jitter_factor
        return max(0.1, raw + random.uniform(-jitter_range, jitter_range))

    def evaluate_request(
        self,
        estimated_tokens: int,
        now: Optional[float] = None,
    ) -> Tuple[RouteDecision, Optional[str], Optional[float]]:
        """Evaluates whether to permit primary request, route to fallback, or shed.

        Returns:
            Tuple of (RouteDecision, target_model_id, retry_after_sec)
        """
        current_time = now if now is not None else time.time()
        self.metrics.total_requests += 1
        self._replenish_quota(current_time)

        # State transition: Check if OPEN cooldown has elapsed to enter HALF_OPEN
        if self.state == BreakerState.OPEN:
            assert self._opened_at is not None
            if current_time >= self._opened_at + self._cooldown_duration:
                self.state = BreakerState.HALF_OPEN
                self._probes_in_flight = 0
                self._successful_canary_count = 0

        # Branch according to circuit state
        if self.state == BreakerState.CLOSED:
            if self.token_bucket.available_tokens >= estimated_tokens:
                self.token_bucket.available_tokens -= estimated_tokens
                self.metrics.primary_allowed += 1
                return RouteDecision.ALLOW_PRIMARY, self.model_id, None
            else:
                # Quota exhausted in current window -> Route fallback or shed
                return self._fallback_or_shed(retry_wait=self._wait_time_for_tokens(estimated_tokens))

        elif self.state == BreakerState.HALF_OPEN:
            if self._probes_in_flight < self.half_open_probe_limit:
                self._probes_in_flight += 1
                self.metrics.primary_allowed += 1
                # Probe uses token quota if available, but allowed through as canary
                self.token_bucket.available_tokens = max(
                    0.0, self.token_bucket.available_tokens - estimated_tokens
                )
                return RouteDecision.ALLOW_PRIMARY, self.model_id, None
            else:
                # Canary slots full, route remaining to fallback
                return self._fallback_or_shed(retry_wait=5.0)

        else:  # BreakerState.OPEN
            retry_after = max(0.0, (self._opened_at + self._cooldown_duration) - current_time)
            return self._fallback_or_shed(retry_wait=retry_after)

    def _fallback_or_shed(self, retry_wait: float) -> Tuple[RouteDecision, Optional[str], Optional[float]]:
        if self.fallback_model_id is not None:
            self.metrics.fallback_routed += 1
            return RouteDecision.ROUTE_FALLBACK, self.fallback_model_id, retry_wait
        self.metrics.shed_requests += 1
        return RouteDecision.SHED_REQUEST, None, retry_wait

    def _wait_time_for_tokens(self, needed_tokens: int) -> float:
        deficit = needed_tokens - self.token_bucket.available_tokens
        if deficit <= 0:
            return 0.0
        return deficit / self.token_bucket.replenish_rate_tokens_per_sec

    def record_success(self, model_id: str) -> None:
        """Records a successful model invocation."""
        if model_id == self.model_id:
            self._consecutive_rate_limits = 0
            if self.state == BreakerState.HALF_OPEN:
                self._successful_canary_count += 1
                self.metrics.successful_probes += 1
                if self._successful_canary_count >= self.half_open_probe_limit:
                    # Circuit successfully recovers
                    self.state = BreakerState.CLOSED
                    self._consecutive_backoff_steps = 0
                    self._opened_at = None
                    self._probes_in_flight = 0

    def record_rate_limit(self, model_id: str, now: Optional[float] = None) -> None:
        """Records a 429 or rate-limit trip on the primary model."""
        if model_id != self.model_id:
            return

        current_time = now if now is not None else time.time()
        self.metrics.rate_limit_trips += 1
        self._consecutive_rate_limits += 1

        if self.state == BreakerState.HALF_OPEN:
            # Canary probe failed -> immediately re-trip to OPEN with escalated backoff
            self.metrics.failed_probes += 1
            self.state = BreakerState.OPEN
            self._consecutive_backoff_steps += 1
            self._opened_at = current_time
            self._cooldown_duration = self._calculate_cooldown()
            self._probes_in_flight = 0
        elif self.state == BreakerState.CLOSED:
            if self._consecutive_rate_limits >= self.consecutive_rate_limits_to_open:
                self.state = BreakerState.OPEN
                self._opened_at = current_time
                self._cooldown_duration = self._calculate_cooldown()


class MultiModelQuotaRegistry:
    """Registry coordinating rate-limit circuit breakers across heterogeneous model providers."""

    def __init__(self) -> None:
        self._breakers: Dict[str, ModelQuotaCircuitBreaker] = {}

    def register_breaker(self, breaker: ModelQuotaCircuitBreaker) -> None:
        self._breakers[breaker.model_id] = breaker

    def get_breaker(self, model_id: str) -> Optional[ModelQuotaCircuitBreaker]:
        return self._breakers.get(model_id)

    def route_request(
        self,
        primary_model_id: str,
        estimated_tokens: int,
        now: Optional[float] = None,
    ) -> Tuple[RouteDecision, Optional[str], Optional[float]]:
        breaker = self.get_breaker(primary_model_id)
        if breaker is None:
            return RouteDecision.ALLOW_PRIMARY, primary_model_id, None
        return breaker.evaluate_request(estimated_tokens, now=now)
