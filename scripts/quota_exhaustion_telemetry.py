"""
Multi-Tenant Token Quota Exhaustion & Rate-Limit Latency Rotary Telemetry Engine.

Tracks multi-tenant throttling events, quota exhaustion frequencies, backoff
latency distributions (p50/p90/p99 retry-after), and Prometheus-compatible metrics.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional, Set, Tuple


class RateLimitReason(str, Enum):
    TOKEN_CAP_EXCEEDED = "TOKEN_CAP_EXCEEDED"
    REQUEST_CAP_EXCEEDED = "REQUEST_CAP_EXCEEDED"
    CONCURRENCY_SPIKE = "CONCURRENCY_SPIKE"
    TIER_THROTTLED = "TIER_THROTTLED"


@dataclass(frozen=True)
class ThrottlingEvent:
    tenant_id: str
    timestamp_sec: float
    retry_after_ms: float
    requested_tokens: int
    available_tokens: int
    reason: RateLimitReason
    endpoint: str = "default"


@dataclass
class TenantQuotaTelemetrySnapshot:
    tenant_id: str
    total_throttles: int
    total_tokens_rejected: int
    last_throttled_sec: Optional[float]
    p50_retry_after_ms: float
    p90_retry_after_ms: float
    p99_retry_after_ms: float
    max_retry_after_ms: float
    mean_retry_after_ms: float
    throttles_per_minute: float
    affected_endpoints: List[str]
    dominant_reason: Optional[RateLimitReason]


@dataclass
class FleetThrottleSummary:
    total_fleet_throttles: int
    active_throttled_tenants: int
    top_throttled_tenants: List[Tuple[str, int]]
    fleet_p90_retry_after_ms: float
    total_tokens_shed: int


class QuotaExhaustionTelemetryRotary:
    """Thread-safe rotary telemetry buffer and latency aggregator for rate limits."""

    def __init__(self, ring_capacity: int = 2000, rate_window_sec: float = 60.0) -> None:
        if ring_capacity <= 0:
            raise ValueError("ring_capacity must be positive")
        self.ring_capacity = ring_capacity
        self.rate_window_sec = rate_window_sec
        self._lock = threading.Lock()
        self._events: Deque[ThrottlingEvent] = deque(maxlen=ring_capacity)
        self._tenant_events: Dict[str, Deque[ThrottlingEvent]] = {}
        self._tenant_total_throttles: Dict[str, int] = {}
        self._tenant_tokens_shed: Dict[str, int] = {}

    def record_throttle(
        self,
        tenant_id: str,
        retry_after_ms: float,
        requested_tokens: int,
        available_tokens: int = 0,
        reason: RateLimitReason = RateLimitReason.TOKEN_CAP_EXCEEDED,
        endpoint: str = "default",
        timestamp_sec: Optional[float] = None,
    ) -> ThrottlingEvent:
        """Records a tenant rate limit / quota exhaustion event in the rotary buffer."""
        if not tenant_id:
            raise ValueError("tenant_id must not be empty")
        if retry_after_ms < 0:
            raise ValueError("retry_after_ms must be non-negative")
        if requested_tokens < 0:
            raise ValueError("requested_tokens must be non-negative")

        ts = timestamp_sec if timestamp_sec is not None else time.time()
        event = ThrottlingEvent(
            tenant_id=tenant_id,
            timestamp_sec=ts,
            retry_after_ms=float(retry_after_ms),
            requested_tokens=requested_tokens,
            available_tokens=available_tokens,
            reason=reason,
            endpoint=endpoint,
        )

        with self._lock:
            self._events.append(event)
            if tenant_id not in self._tenant_events:
                self._tenant_events[tenant_id] = deque(maxlen=self.ring_capacity)
                self._tenant_total_throttles[tenant_id] = 0
                self._tenant_tokens_shed[tenant_id] = 0

            self._tenant_events[tenant_id].append(event)
            self._tenant_total_throttles[tenant_id] += 1
            self._tenant_tokens_shed[tenant_id] += requested_tokens

        return event

    def _compute_percentile(self, sorted_values: List[float], pct: float) -> float:
        if not sorted_values:
            return 0.0
        idx = int(math.ceil(pct * len(sorted_values))) - 1
        return sorted_values[max(0, min(idx, len(sorted_values) - 1))]

    def get_tenant_snapshot(self, tenant_id: str, now_sec: Optional[float] = None) -> Optional[TenantQuotaTelemetrySnapshot]:
        """Calculates latency quantiles and rate for a specific tenant."""
        now = now_sec if now_sec is not None else time.time()
        with self._lock:
            events_queue = self._tenant_events.get(tenant_id)
            if not events_queue:
                return None
            events = list(events_queue)
            total_throttles = self._tenant_total_throttles[tenant_id]
            total_shed = self._tenant_tokens_shed[tenant_id]

        latencies = sorted(e.retry_after_ms for e in events)
        p50 = self._compute_percentile(latencies, 0.50)
        p90 = self._compute_percentile(latencies, 0.90)
        p99 = self._compute_percentile(latencies, 0.99)
        max_lat = latencies[-1] if latencies else 0.0
        mean_lat = sum(latencies) / len(latencies) if latencies else 0.0

        # Calculate rate in recent window
        window_start = now - self.rate_window_sec
        recent_count = sum(1 for e in events if e.timestamp_sec >= window_start)
        rate_per_min = (recent_count / self.rate_window_sec) * 60.0

        endpoints = sorted(list({e.endpoint for e in events}))

        # Dominant reason
        reason_counts: Dict[RateLimitReason, int] = {}
        for e in events:
            reason_counts[e.reason] = reason_counts.get(e.reason, 0) + 1
        dominant_reason = max(reason_counts.items(), key=lambda x: x[1])[0] if reason_counts else None

        last_ts = events[-1].timestamp_sec if events else None

        return TenantQuotaTelemetrySnapshot(
            tenant_id=tenant_id,
            total_throttles=total_throttles,
            total_tokens_rejected=total_shed,
            last_throttled_sec=last_ts,
            p50_retry_after_ms=p50,
            p90_retry_after_ms=p90,
            p99_retry_after_ms=p99,
            max_retry_after_ms=max_lat,
            mean_retry_after_ms=mean_lat,
            throttles_per_minute=rate_per_min,
            affected_endpoints=endpoints,
            dominant_reason=dominant_reason,
        )

    def get_fleet_summary(self, top_k: int = 5) -> FleetThrottleSummary:
        """Generates an aggregated fleet-wide rate limit telemetry summary."""
        with self._lock:
            all_events = list(self._events)
            active_tenants = list(self._tenant_total_throttles.keys())
            total_shed = sum(self._tenant_tokens_shed.values())

            # Top tenants by total throttles
            sorted_tenants = sorted(
                self._tenant_total_throttles.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:top_k]

        latencies = sorted(e.retry_after_ms for e in all_events)
        p90 = self._compute_percentile(latencies, 0.90)

        return FleetThrottleSummary(
            total_fleet_throttles=len(all_events),
            active_throttled_tenants=len(active_tenants),
            top_throttled_tenants=sorted_tenants,
            fleet_p90_retry_after_ms=p90,
            total_tokens_shed=total_shed,
        )

    def export_prometheus_metrics(self) -> str:
        """Formats telemetry counters and gauge distributions in OpenMetrics/Prometheus syntax."""
        summary = self.get_fleet_summary()
        lines = [
            "# HELP token_tariff_fleet_throttles_total Total rate limiting occurrences in buffer",
            "# TYPE token_tariff_fleet_throttles_total counter",
            f"token_tariff_fleet_throttles_total {summary.total_fleet_throttles}",
            "# HELP token_tariff_fleet_tokens_shed_total Total tokens rejected across tenants",
            "# TYPE token_tariff_fleet_tokens_shed_total counter",
            f"token_tariff_fleet_tokens_shed_total {summary.total_tokens_shed}",
            "# HELP token_tariff_retry_after_p90_ms Fleet-wide 90th percentile backoff latency",
            "# TYPE token_tariff_retry_after_p90_ms gauge",
            f"token_tariff_retry_after_p90_ms {summary.fleet_p90_retry_after_ms:.2f}",
        ]

        with self._lock:
            for tenant, total in self._tenant_total_throttles.items():
                lines.append(f'token_tariff_tenant_throttles_total{{tenant="{tenant}"}} {total}')

        return "\n".join(lines) + "\n"
