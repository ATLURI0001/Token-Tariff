"""Distributed Cost Center Quota Rotary & Spend Velocity Telemetry."""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional, Tuple


class QuotaStatus(str, Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    EXHAUSTED = "EXHAUSTED"


@dataclass
class SpendEvent:
    timestamp_sec: float
    cost_usd: float
    token_count: int
    model_id: str


@dataclass
class QuotaDiagnosticReport:
    cost_center: str
    quota_limit_usd: float
    total_spent_usd: float
    remaining_quota_usd: float
    status: QuotaStatus
    velocity_usd_per_hour: float
    projected_hours_to_exhaustion: Optional[float]
    is_surge_anomaly: bool
    event_count: int


class SpendTelemetryEngine:
    """Thread-safe telemetry engine tracking cost consumption, burn velocity, and quota exhaustion."""

    def __init__(
        self,
        cost_center: str,
        quota_limit_usd: float,
        window_duration_sec: float = 3600.0,
        warning_threshold_pct: float = 0.75,
    ) -> None:
        if quota_limit_usd <= 0.0:
            raise ValueError("quota_limit_usd must be strictly positive")
        if window_duration_sec <= 0.0:
            raise ValueError("window_duration_sec must be strictly positive")
        if not (0.0 < warning_threshold_pct < 1.0):
            raise ValueError("warning_threshold_pct must be between 0.0 and 1.0")

        self.cost_center = cost_center
        self.quota_limit_usd = float(quota_limit_usd)
        self.window_duration_sec = float(window_duration_sec)
        self.warning_threshold_pct = float(warning_threshold_pct)

        self._lock = threading.Lock()
        self._total_spent_usd = 0.0
        self._total_tokens = 0
        self._events: Deque[SpendEvent] = deque()
        self._hourly_rates: Deque[float] = deque(maxlen=24)  # Track past 24 hourly averages

    def record_spend(
        self,
        cost_usd: float,
        token_count: int,
        model_id: str,
        timestamp_sec: float,
    ) -> None:
        """Records a consumption event and evicts stale records outside the sliding window."""
        if cost_usd < 0.0 or token_count < 0:
            raise ValueError("Cost and token count must be non-negative")

        with self._lock:
            self._total_spent_usd += cost_usd
            self._total_tokens += token_count
            event = SpendEvent(
                timestamp_sec=float(timestamp_sec),
                cost_usd=float(cost_usd),
                token_count=int(token_count),
                model_id=str(model_id),
            )
            self._events.append(event)
            self._evict_stale_locked(current_time=timestamp_sec)

    def _evict_stale_locked(self, current_time: float) -> None:
        cutoff = current_time - self.window_duration_sec
        while self._events and self._events[0].timestamp_sec < cutoff:
            self._events.popleft()

    def get_diagnostic_report(self, current_timestamp_sec: float) -> QuotaDiagnosticReport:
        """Evaluates spend velocity, remaining runway, surge anomaly, and quota status."""
        with self._lock:
            self._evict_stale_locked(current_time=current_timestamp_sec)

            remaining = max(0.0, self.quota_limit_usd - self._total_spent_usd)

            # Determine quota status
            if self._total_spent_usd >= self.quota_limit_usd:
                status = QuotaStatus.EXHAUSTED
            elif self._total_spent_usd >= self.quota_limit_usd * self.warning_threshold_pct:
                status = QuotaStatus.WARNING
            else:
                status = QuotaStatus.NORMAL

            # Calculate window spend velocity
            window_spend = sum(e.cost_usd for e in self._events)
            if self._events and len(self._events) > 1:
                window_span = max(1.0, self._events[-1].timestamp_sec - self._events[0].timestamp_sec)
                velocity_per_sec = window_spend / window_span
            elif self._events:
                velocity_per_sec = window_spend / max(1.0, self.window_duration_sec)
            else:
                velocity_per_sec = 0.0

            velocity_per_hour = velocity_per_sec * 3600.0

            # Time to exhaustion
            if velocity_per_hour > 1e-9 and remaining > 0.0:
                tte_hours = remaining / velocity_per_hour
            elif remaining <= 0.0:
                tte_hours = 0.0
            else:
                tte_hours = None

            # Anomaly Surge Detection (if historical baseline exists)
            is_surge = False
            if self._hourly_rates and len(self._hourly_rates) >= 3:
                mean_rate = sum(self._hourly_rates) / len(self._hourly_rates)
                variance = sum((r - mean_rate) ** 2 for r in self._hourly_rates) / len(self._hourly_rates)
                std_dev = math.sqrt(variance)
                if std_dev > 1e-4 and velocity_per_hour > (mean_rate + 2.5 * std_dev):
                    is_surge = True

            # Register current rate in history
            if velocity_per_hour > 0:
                self._hourly_rates.append(velocity_per_hour)

            return QuotaDiagnosticReport(
                cost_center=self.cost_center,
                quota_limit_usd=self.quota_limit_usd,
                total_spent_usd=round(self._total_spent_usd, 6),
                remaining_quota_usd=round(remaining, 6),
                status=status,
                velocity_usd_per_hour=round(velocity_per_hour, 6),
                projected_hours_to_exhaustion=round(tte_hours, 2) if tte_hours is not None else None,
                is_surge_anomaly=is_surge,
                event_count=len(self._events),
            )

    def export_prometheus_metrics(self, current_timestamp_sec: float) -> str:
        """Emits Prometheus metrics format for scraping."""
        report = self.get_diagnostic_report(current_timestamp_sec)
        status_num = 1 if report.status == QuotaStatus.NORMAL else (2 if report.status == QuotaStatus.WARNING else 3)

        lines = [
            f"# HELP token_tariff_spend_dollars_total Total cumulative cost consumed for cost center",
            f"# TYPE token_tariff_spend_dollars_total counter",
            f'token_tariff_spend_dollars_total{{cost_center="{self.cost_center}"}} {report.total_spent_usd:.6f}',
            f"# HELP token_tariff_quota_dollars Configured quota ceiling in USD",
            f"# TYPE token_tariff_quota_dollars gauge",
            f'token_tariff_quota_dollars{{cost_center="{self.cost_center}"}} {report.quota_limit_usd:.2f}',
            f"# HELP token_tariff_spend_velocity_dollars_per_hour Instantaneous burn velocity",
            f"# TYPE token_tariff_spend_velocity_dollars_per_hour gauge",
            f'token_tariff_spend_velocity_dollars_per_hour{{cost_center="{self.cost_center}"}} {report.velocity_usd_per_hour:.6f}',
            f"# HELP token_tariff_quota_status Status enum (1=NORMAL, 2=WARNING, 3=EXHAUSTED)",
            f"# TYPE token_tariff_quota_status gauge",
            f'token_tariff_quota_status{{cost_center="{self.cost_center}"}} {status_num}',
        ]
        return "\n".join(lines) + "\n"
