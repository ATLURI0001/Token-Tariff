"""
Provider Pricing Drift & Model Cost Disparity Telemetry Rotary Engine.

Monitors upstream LLM API pricing volatility, tracks price drift over rolling
windows, detects disparity anomalies across providers, and outputs
Prometheus/OpenTelemetry-formatted tariff telemetry.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional, Tuple


class DriftSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class DriftDirection(str, Enum):
    INFLATION = "INFLATION"
    DEFLATION = "DEFLATION"
    STABLE = "STABLE"


@dataclass(frozen=True)
class PriceObservation:
    provider: str
    model: str
    timestamp_sec: float
    input_cost_per_1m: float
    output_cost_per_1m: float
    cached_input_cost_per_1m: Optional[float] = None


@dataclass(frozen=True)
class DriftAlert:
    provider: str
    model: str
    dimension: str
    old_rate: float
    new_rate: float
    drift_pct: float
    severity: DriftSeverity
    direction: DriftDirection
    timestamp_sec: float


@dataclass
class ModelDriftSnapshot:
    provider: str
    model: str
    observation_count: int
    baseline_input_cost: float
    current_input_cost: float
    input_drift_pct: float
    baseline_output_cost: float
    current_output_cost: float
    output_drift_pct: float
    dominant_direction: DriftDirection
    max_drift_pct: float
    volatility_score: float
    active_alerts: List[DriftAlert] = field(default_factory=list)


class PriceDriftTelemetryRotary:
    """
    Thread-safe circular telemetry buffer tracking model pricing shifts
    and calculating real-time drift metrics and Prometheus metrics exports.
    """

    def __init__(
        self,
        buffer_capacity: int = 250,
        warning_drift_threshold_pct: float = 5.0,
        critical_drift_threshold_pct: float = 25.0,
    ) -> None:
        self.capacity = max(10, buffer_capacity)
        self.warning_threshold = warning_drift_threshold_pct
        self.critical_threshold = critical_drift_threshold_pct
        self._lock = threading.Lock()
        self._buffers: Dict[Tuple[str, str], Deque[PriceObservation]] = {}
        self._alerts: Deque[DriftAlert] = deque(maxlen=1000)

    def record_observation(
        self,
        provider: str,
        model: str,
        input_cost_per_1m: float,
        output_cost_per_1m: float,
        cached_input_cost_per_1m: Optional[float] = None,
        timestamp_sec: Optional[float] = None,
    ) -> Optional[DriftAlert]:
        if timestamp_sec is None:
            timestamp_sec = time.time()

        key = (provider.lower().strip(), model.lower().strip())
        obs = PriceObservation(
            provider=key[0],
            model=key[1],
            timestamp_sec=timestamp_sec,
            input_cost_per_1m=input_cost_per_1m,
            output_cost_per_1m=output_cost_per_1m,
            cached_input_cost_per_1m=cached_input_cost_per_1m,
        )

        with self._lock:
            if key not in self._buffers:
                self._buffers[key] = deque(maxlen=self.capacity)
            buf = self._buffers[key]

            alert: Optional[DriftAlert] = None
            if len(buf) > 0:
                prev = buf[-1]
                alert = self._detect_drift(prev, obs)
                if alert is not None:
                    self._alerts.append(alert)

            buf.append(obs)
            return alert

    def _detect_drift(
        self, prev: PriceObservation, curr: PriceObservation
    ) -> Optional[DriftAlert]:
        # Check input drift
        if prev.input_cost_per_1m > 0:
            diff = curr.input_cost_per_1m - prev.input_cost_per_1m
            pct = (diff / prev.input_cost_per_1m) * 100.0
            abs_pct = abs(pct)

            if abs_pct >= self.warning_threshold:
                sev = (
                    DriftSeverity.CRITICAL
                    if abs_pct >= self.critical_threshold
                    else DriftSeverity.WARNING
                )
                dirn = (
                    DriftDirection.INFLATION
                    if diff > 0
                    else DriftDirection.DEFLATION
                )
                return DriftAlert(
                    provider=curr.provider,
                    model=curr.model,
                    dimension="input",
                    old_rate=prev.input_cost_per_1m,
                    new_rate=curr.input_cost_per_1m,
                    drift_pct=round(pct, 2),
                    severity=sev,
                    direction=dirn,
                    timestamp_sec=curr.timestamp_sec,
                )

        # Check output drift
        if prev.output_cost_per_1m > 0:
            diff = curr.output_cost_per_1m - prev.output_cost_per_1m
            pct = (diff / prev.output_cost_per_1m) * 100.0
            abs_pct = abs(pct)

            if abs_pct >= self.warning_threshold:
                sev = (
                    DriftSeverity.CRITICAL
                    if abs_pct >= self.critical_threshold
                    else DriftSeverity.WARNING
                )
                dirn = (
                    DriftDirection.INFLATION
                    if diff > 0
                    else DriftDirection.DEFLATION
                )
                return DriftAlert(
                    provider=curr.provider,
                    model=curr.model,
                    dimension="output",
                    old_rate=prev.output_cost_per_1m,
                    new_rate=curr.output_cost_per_1m,
                    drift_pct=round(pct, 2),
                    severity=sev,
                    direction=dirn,
                    timestamp_sec=curr.timestamp_sec,
                )

        return None

    def get_model_snapshot(
        self, provider: str, model: str
    ) -> Optional[ModelDriftSnapshot]:
        key = (provider.lower().strip(), model.lower().strip())
        with self._lock:
            buf = self._buffers.get(key)
            if not buf:
                return None

            items = list(buf)
            first = items[0]
            latest = items[-1]

            in_drift = (
                ((latest.input_cost_per_1m - first.input_cost_per_1m) / first.input_cost_per_1m) * 100.0
                if first.input_cost_per_1m > 0
                else 0.0
            )
            out_drift = (
                ((latest.output_cost_per_1m - first.output_cost_per_1m) / first.output_cost_per_1m) * 100.0
                if first.output_cost_per_1m > 0
                else 0.0
            )

            max_drift = max(abs(in_drift), abs(out_drift))
            if max_drift < self.warning_threshold:
                dominant_dir = DriftDirection.STABLE
            elif in_drift > 0 or out_drift > 0:
                dominant_dir = DriftDirection.INFLATION
            else:
                dominant_dir = DriftDirection.DEFLATION

            # Simple volatility as standard deviation of input costs
            mean_in = sum(x.input_cost_per_1m for x in items) / len(items)
            var_in = sum((x.input_cost_per_1m - mean_in) ** 2 for x in items) / len(items)
            volatility = math.sqrt(var_in)

            alerts_for_model = [
                a for a in self._alerts if a.provider == key[0] and a.model == key[1]
            ]

            return ModelDriftSnapshot(
                provider=key[0],
                model=key[1],
                observation_count=len(items),
                baseline_input_cost=first.input_cost_per_1m,
                current_input_cost=latest.input_cost_per_1m,
                input_drift_pct=round(in_drift, 2),
                baseline_output_cost=first.output_cost_per_1m,
                current_output_cost=latest.output_cost_per_1m,
                output_drift_pct=round(out_drift, 2),
                dominant_direction=dominant_dir,
                max_drift_pct=round(max_drift, 2),
                volatility_score=round(volatility, 4),
                active_alerts=alerts_for_model[-5:],
            )

    def export_prometheus_metrics(self) -> str:
        lines: List[str] = [
            "# HELP tokentariff_model_input_cost_usd_per_million Current input cost per 1M tokens",
            "# TYPE tokentariff_model_input_cost_usd_per_million gauge",
        ]

        with self._lock:
            for (prov, mod), buf in self._buffers.items():
                if buf:
                    latest = buf[-1]
                    lines.append(
                        f'tokentariff_model_input_cost_usd_per_million{{provider="{prov}",model="{mod}"}} {latest.input_cost_per_1m:.4f}'
                    )

            lines.append(
                "# HELP tokentariff_model_output_cost_usd_per_million Current output cost per 1M tokens"
            )
            lines.append(
                "# TYPE tokentariff_model_output_cost_usd_per_million gauge"
            )
            for (prov, mod), buf in self._buffers.items():
                if buf:
                    latest = buf[-1]
                    lines.append(
                        f'tokentariff_model_output_cost_usd_per_million{{provider="{prov}",model="{mod}"}} {latest.output_cost_per_1m:.4f}'
                    )

            lines.append(
                "# HELP tokentariff_model_drift_ratio_pct Percentage price drift against rolling baseline"
            )
            lines.append(
                "# TYPE tokentariff_model_drift_ratio_pct gauge"
            )
            for (prov, mod), buf in self._buffers.items():
                if buf:
                    first = buf[0]
                    latest = buf[-1]
                    drift = (
                        ((latest.input_cost_per_1m - first.input_cost_per_1m) / first.input_cost_per_1m) * 100.0
                        if first.input_cost_per_1m > 0
                        else 0.0
                    )
                    lines.append(
                        f'tokentariff_model_drift_ratio_pct{{provider="{prov}",model="{mod}",dimension="input"}} {drift:.2f}'
                    )

            lines.append(
                "# HELP tokentariff_price_drift_alerts_total Total count of detected price drift anomalies"
            )
            lines.append(
                "# TYPE tokentariff_price_drift_alerts_total counter"
            )
            counts: Dict[Tuple[str, str, str, str], int] = {}
            for a in self._alerts:
                k = (a.provider, a.model, a.severity.value, a.direction.value)
                counts[k] = counts.get(k, 0) + 1

            for (prov, mod, sev, dirn), count in counts.items():
                lines.append(
                    f'tokentariff_price_drift_alerts_total{{provider="{prov}",model="{mod}",severity="{sev}",direction="{dirn}"}} {count}'
                )

        return "\n".join(lines) + "\n"
