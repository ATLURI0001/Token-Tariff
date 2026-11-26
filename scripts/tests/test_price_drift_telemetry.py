"""
Tests for Provider Pricing Drift & Model Cost Disparity Telemetry Rotary.
"""

import time
import pytest
from scripts.price_drift_telemetry import (
    DriftDirection,
    DriftSeverity,
    PriceDriftTelemetryRotary,
)


def test_record_observations_and_baseline() -> None:
    rotary = PriceDriftTelemetryRotary(buffer_capacity=10)
    # Baseline
    alert = rotary.record_observation(
        provider="anthropic",
        model="claude-3-5-sonnet",
        input_cost_per_1m=3.00,
        output_cost_per_1m=15.00,
    )
    assert alert is None

    snapshot = rotary.get_model_snapshot("anthropic", "claude-3-5-sonnet")
    assert snapshot is not None
    assert snapshot.observation_count == 1
    assert snapshot.baseline_input_cost == 3.00
    assert snapshot.current_input_cost == 3.00
    assert snapshot.input_drift_pct == 0.0
    assert snapshot.dominant_direction == DriftDirection.STABLE


def test_drift_warning_and_critical_detection() -> None:
    rotary = PriceDriftTelemetryRotary(
        buffer_capacity=50,
        warning_drift_threshold_pct=10.0,
        critical_drift_threshold_pct=30.0,
    )

    rotary.record_observation("openai", "gpt-4o", 5.00, 15.00)

    # 12% price increase -> WARNING INFLATION
    alert1 = rotary.record_observation("openai", "gpt-4o", 5.60, 15.00)
    assert alert1 is not None
    assert alert1.severity == DriftSeverity.WARNING
    assert alert1.direction == DriftDirection.INFLATION
    assert alert1.drift_pct == 12.0

    # 35% price drop -> CRITICAL DEFLATION
    alert2 = rotary.record_observation("openai", "gpt-4o", 3.64, 15.00)
    assert alert2 is not None
    assert alert2.severity == DriftSeverity.CRITICAL
    assert alert2.direction == DriftDirection.DEFLATION


def test_prometheus_metrics_export() -> None:
    rotary = PriceDriftTelemetryRotary(buffer_capacity=20)
    rotary.record_observation("google", "gemini-1.5-pro", 1.25, 5.00)
    rotary.record_observation("google", "gemini-1.5-pro", 1.50, 5.00)

    metrics = rotary.export_prometheus_metrics()
    assert "tokentariff_model_input_cost_usd_per_million" in metrics
    assert 'provider="google"' in metrics
    assert 'model="gemini-1.5-pro"' in metrics
    assert "tokentariff_model_drift_ratio_pct" in metrics
    assert "tokentariff_price_drift_alerts_total" in metrics


def test_ring_buffer_capacity_eviction() -> None:
    capacity = 10
    rotary = PriceDriftTelemetryRotary(buffer_capacity=capacity)
    for i in range(25):
        rotary.record_observation("cohere", "command-r", 0.50 + i * 0.01, 1.50)

    snapshot = rotary.get_model_snapshot("cohere", "command-r")
    assert snapshot is not None
    assert snapshot.observation_count == capacity
