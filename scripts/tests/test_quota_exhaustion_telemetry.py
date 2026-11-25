import pytest
import time
from scripts.quota_exhaustion_telemetry import (
    QuotaExhaustionTelemetryRotary,
    RateLimitReason,
    ThrottlingEvent,
)


def test_record_throttle_and_snapshot():
    rotary = QuotaExhaustionTelemetryRotary(ring_capacity=100, rate_window_sec=60.0)

    # Record 5 throttling events for tenant-alpha
    base_ts = 1000.0
    rotary.record_throttle("tenant-alpha", retry_after_ms=100.0, requested_tokens=1000, timestamp_sec=base_ts + 1)
    rotary.record_throttle("tenant-alpha", retry_after_ms=200.0, requested_tokens=2000, timestamp_sec=base_ts + 2)
    rotary.record_throttle("tenant-alpha", retry_after_ms=300.0, requested_tokens=1500, timestamp_sec=base_ts + 3)
    rotary.record_throttle("tenant-alpha", retry_after_ms=400.0, requested_tokens=500, timestamp_sec=base_ts + 4)
    rotary.record_throttle("tenant-alpha", retry_after_ms=500.0, requested_tokens=1000, timestamp_sec=base_ts + 5)

    snapshot = rotary.get_tenant_snapshot("tenant-alpha", now_sec=base_ts + 10)
    assert snapshot is not None
    assert snapshot.tenant_id == "tenant-alpha"
    assert snapshot.total_throttles == 5
    assert snapshot.total_tokens_rejected == 6000
    assert snapshot.p50_retry_after_ms == 300.0
    assert snapshot.p90_retry_after_ms == 500.0
    assert snapshot.max_retry_after_ms == 500.0
    assert snapshot.mean_retry_after_ms == 300.0
    assert snapshot.dominant_reason == RateLimitReason.TOKEN_CAP_EXCEEDED
    assert snapshot.throttles_per_minute == 5.0  # 5 events in 60s window


def test_fleet_summary_and_top_tenants():
    rotary = QuotaExhaustionTelemetryRotary(ring_capacity=50)

    rotary.record_throttle("tenant-A", retry_after_ms=50.0, requested_tokens=500)
    rotary.record_throttle("tenant-A", retry_after_ms=150.0, requested_tokens=500)
    rotary.record_throttle("tenant-A", retry_after_ms=250.0, requested_tokens=500)

    rotary.record_throttle("tenant-B", retry_after_ms=1000.0, requested_tokens=3000)

    summary = rotary.get_fleet_summary(top_k=2)
    assert summary.total_fleet_throttles == 4
    assert summary.active_throttled_tenants == 2
    assert summary.top_throttled_tenants[0] == ("tenant-A", 3)
    assert summary.top_throttled_tenants[1] == ("tenant-B", 1)
    assert summary.total_tokens_shed == 4500


def test_ring_buffer_capacity_eviction():
    rotary = QuotaExhaustionTelemetryRotary(ring_capacity=3)

    rotary.record_throttle("t1", retry_after_ms=10.0, requested_tokens=100)
    rotary.record_throttle("t1", retry_after_ms=20.0, requested_tokens=100)
    rotary.record_throttle("t1", retry_after_ms=30.0, requested_tokens=100)
    # Evicts 10.0
    rotary.record_throttle("t1", retry_after_ms=40.0, requested_tokens=100)

    snapshot = rotary.get_tenant_snapshot("t1")
    assert snapshot is not None
    assert snapshot.total_throttles == 4  # Cumulative total remains 4
    assert snapshot.p50_retry_after_ms == 30.0  # Active buffer: [20.0, 30.0, 40.0]


def test_prometheus_metrics_export():
    rotary = QuotaExhaustionTelemetryRotary(ring_capacity=100)
    rotary.record_throttle("tenant-prod", retry_after_ms=250.0, requested_tokens=4000)

    prom_text = rotary.export_prometheus_metrics()
    assert "token_tariff_fleet_throttles_total 1" in prom_text
    assert "token_tariff_fleet_tokens_shed_total 4000" in prom_text
    assert 'token_tariff_tenant_throttles_total{tenant="tenant-prod"} 1' in prom_text


def test_invalid_parameters():
    rotary = QuotaExhaustionTelemetryRotary(ring_capacity=10)

    with pytest.raises(ValueError):
        rotary.record_throttle("", retry_after_ms=10.0, requested_tokens=100)

    with pytest.raises(ValueError):
        rotary.record_throttle("t", retry_after_ms=-1.0, requested_tokens=100)

    with pytest.raises(ValueError):
        rotary.record_throttle("t", retry_after_ms=10.0, requested_tokens=-5)

    assert rotary.get_tenant_snapshot("non-existent") is None
