"""
Real-Time Token Consumption Velocity Profiler & Burn-Rate Degradation Diagnostic.

Tracks instantaneous tokens-per-minute (TPM) and cost-per-minute (CPM) velocity,
detects recursive generation loops, and provides automated burn-rate diagnostics.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class VelocityHealthStatus(str, Enum):
    HEALTHY = "HEALTHY"
    WARNING_ELEVATED_BURN_RATE = "WARNING_ELEVATED_BURN_RATE"
    WARNING_ASYMMETRIC_COMPLETION = "WARNING_ASYMMETRIC_COMPLETION"
    CRITICAL_RUNAWAY_CONSUMPTION = "CRITICAL_RUNAWAY_CONSUMPTION"


@dataclass
class TokenConsumptionRecord:
    tenant_id: str
    model_id: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class VelocityDiagnosticReport:
    total_events: int
    total_tokens: int
    total_cost_usd: float
    window_duration_sec: float
    tokens_per_minute: float
    cost_per_minute_usd: float
    prompt_completion_ratio: float
    status: VelocityHealthStatus
    recommendations: List[str]


@dataclass
class VelocityProfilerConfig:
    window_sec: float = 600.0                # 10 minute sliding window
    budget_cpm_usd: float = 5.0             # $5.00/min baseline ceiling
    asymmetric_ratio_threshold: float = 8.0 # completion > 8x prompt implies runaway loop
    max_history_entries: int = 5000


class TokenVelocityProfiler:
    """Measures real-time token consumption velocity and flags burn-rate anomalies."""

    def __init__(self, config: Optional[VelocityProfilerConfig] = None) -> None:
        self.config = config or VelocityProfilerConfig()
        self.history: List[TokenConsumptionRecord] = []

    def record_usage(
        self,
        tenant_id: str,
        model_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        timestamp: Optional[float] = None,
    ) -> TokenConsumptionRecord:
        """Records an individual inference consumption event."""
        if not tenant_id or not isinstance(tenant_id, str):
            raise ValueError("tenant_id must be a non-empty string")
        if not model_id or not isinstance(model_id, str):
            raise ValueError("model_id must be a non-empty string")
        if prompt_tokens < 0 or completion_tokens < 0:
            raise ValueError("Token counts cannot be negative")
        if cost_usd < 0.0 or math.isnan(cost_usd):
            raise ValueError("Cost cannot be negative or NaN")

        rec = TokenConsumptionRecord(
            tenant_id=tenant_id.strip(),
            model_id=model_id.strip(),
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
            cost_usd=round(float(cost_usd), 6),
            timestamp=float(timestamp if timestamp is not None else time.time()),
        )

        self.history.append(rec)
        if len(self.history) > self.config.max_history_entries:
            self.history.pop(0)

        return rec

    def diagnose(self, current_time: Optional[float] = None) -> VelocityDiagnosticReport:
        """Evaluates consumption velocity within the sliding window against configured SLAs."""
        now = current_time if current_time is not None else time.time()
        window_start = now - self.config.window_sec

        active_events = [r for r in self.history if r.timestamp >= window_start]

        if not active_events:
            return VelocityDiagnosticReport(
                total_events=0,
                total_tokens=0,
                total_cost_usd=0.0,
                window_duration_sec=self.config.window_sec,
                tokens_per_minute=0.0,
                cost_per_minute_usd=0.0,
                prompt_completion_ratio=1.0,
                status=VelocityHealthStatus.HEALTHY,
                recommendations=[],
            )

        total_prompt = sum(r.prompt_tokens for r in active_events)
        total_completion = sum(r.completion_tokens for r in active_events)
        total_tokens = total_prompt + total_completion
        total_cost = sum(r.cost_usd for r in active_events)

        # Elapsed span in minutes (at least 1 second to avoid div by zero)
        earliest_event = min(r.timestamp for r in active_events)
        elapsed_sec = max(1.0, min(self.config.window_sec, now - earliest_event))
        elapsed_min = elapsed_sec / 60.0

        tpm = round(total_tokens / elapsed_min, 2)
        cpm = round(total_cost / elapsed_min, 4)

        ratio = round(total_completion / max(1, total_prompt), 2)

        # Status & diagnostic determination
        status = VelocityHealthStatus.HEALTHY
        recommendations: List[str] = []

        if cpm >= self.config.budget_cpm_usd * 1.5:
            status = VelocityHealthStatus.CRITICAL_RUNAWAY_CONSUMPTION
            recommendations.append(
                f"Critical runaway burn rate detected (${cpm}/min vs budget ${self.config.budget_cpm_usd}/min). "
                f"Immediately engage global rate limiter or quota circuit breaker."
            )
        elif ratio >= self.config.asymmetric_ratio_threshold and total_completion > 5000:
            status = VelocityHealthStatus.WARNING_ASYMMETRIC_COMPLETION
            recommendations.append(
                f"Asymmetric completion-to-prompt ratio ({ratio}x) indicates repetitive or infinite generation loop. "
                f"Enforce strict max_tokens constraints on active model calls."
            )
        elif cpm >= self.config.budget_cpm_usd * 0.8:
            status = VelocityHealthStatus.WARNING_ELEVATED_BURN_RATE
            recommendations.append(
                f"Elevated spend velocity (${cpm}/min) is approaching configured budget ceiling (${self.config.budget_cpm_usd}/min). "
                f"Consider enabling token context compression or tiered model routing."
            )

        return VelocityDiagnosticReport(
            total_events=len(active_events),
            total_tokens=total_tokens,
            total_cost_usd=round(total_cost, 4),
            window_duration_sec=round(elapsed_sec, 2),
            tokens_per_minute=tpm,
            cost_per_minute_usd=cpm,
            prompt_completion_ratio=ratio,
            status=status,
            recommendations=recommendations,
        )

    def reset(self) -> None:
        """Flushes recorded consumption history."""
        self.history.clear()
