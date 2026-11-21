"""Unit tests for Distributed Cost Center Quota Rotary & Spend Velocity Telemetry."""

import threading
import pytest
from scripts.spend_telemetry import (
    SpendTelemetryEngine,
    QuotaStatus,
)


def test_spend_telemetry_initialization_and_validation():
    engine = SpendTelemetryEngine(
        cost_center="CC-DATA-INFRA",
        quota_limit_usd=1000.0,
        window_duration_sec=3600.0,
        warning_threshold_pct=0.75,
    )

    report = engine.get_diagnostic_report(current_timestamp_sec=1000.0)
    assert report.cost_center == "CC-DATA-INFRA"
    assert report.total_spent_usd == 0.0
    assert report.remaining_quota_usd == 1000.0
    assert report.status == QuotaStatus.NORMAL
    assert report.velocity_usd_per_hour == 0.0
    assert report.projected_hours_to_exhaustion is None

    # Invalid configurations
    with pytest.raises(ValueError):
        SpendTelemetryEngine(cost_center="CC", quota_limit_usd=-50.0)
    with pytest.raises(ValueError):
        SpendTelemetryEngine(cost_center="CC", quota_limit_usd=100.0, warning_threshold_pct=1.5)


def test_spend_telemetry_status_transitions():
    engine = SpendTelemetryEngine(
        cost_center="CC-GENAI-APP",
        quota_limit_usd=100.0,
        warning_threshold_pct=0.80,
    )

    # 1. 50 USD spend -> NORMAL
    engine.record_spend(cost_usd=50.0, token_count=100_000, model_id="gemini-1.5-pro", timestamp_sec=100.0)
    rep1 = engine.get_diagnostic_report(current_timestamp_sec=100.0)
    assert rep1.status == QuotaStatus.NORMAL
    assert rep1.remaining_quota_usd == 50.0

    # 2. Add 35 USD spend (total 85) -> WARNING (>= 80%)
    engine.record_spend(cost_usd=35.0, token_count=70_000, model_id="gemini-1.5-pro", timestamp_sec=200.0)
    rep2 = engine.get_diagnostic_report(current_timestamp_sec=200.0)
    assert rep2.status == QuotaStatus.WARNING
    assert rep2.remaining_quota_usd == 15.0

    # 3. Add 20 USD spend (total 105) -> EXHAUSTED (>= 100%)
    engine.record_spend(cost_usd=20.0, token_count=40_000, model_id="gemini-1.5-pro", timestamp_sec=300.0)
    rep3 = engine.get_diagnostic_report(current_timestamp_sec=300.0)
    assert rep3.status == QuotaStatus.EXHAUSTED
    assert rep3.remaining_quota_usd == 0.0
    assert rep3.projected_hours_to_exhaustion == 0.0


def test_spend_telemetry_velocity_and_sliding_window():
    # 1 hour window (3600 sec)
    engine = SpendTelemetryEngine(
        cost_center="CC-SEARCH-INDEX",
        quota_limit_usd=500.0,
        window_duration_sec=3600.0,
    )

    # Record $10 spend at t=1000 and $10 spend at t=2000 (span = 1000 sec, spend = $20)
    # Velocity = $20 / 1000s = $0.02/s = $72.00/hour
    engine.record_spend(cost_usd=10.0, token_count=20_000, model_id="model-a", timestamp_sec=1000.0)
    engine.record_spend(cost_usd=10.0, token_count=20_000, model_id="model-a", timestamp_sec=2000.0)

    rep = engine.get_diagnostic_report(current_timestamp_sec=2000.0)
    assert rep.velocity_usd_per_hour == 72.0
    # Remaining: $480. TTE = $480 / $72 = 6.67 hours
    assert rep.projected_hours_to_exhaustion == 6.67

    # Advance time to t=5000 (first event at t=1000 is now > 3600s old and evicted)
    rep_future = engine.get_diagnostic_report(current_timestamp_sec=5000.0)
    assert rep_future.event_count == 1  # Only t=2000 event retained
    assert rep_future.total_spent_usd == 20.0  # Cumulative spend never evicted


def test_spend_telemetry_surge_anomaly():
    engine = SpendTelemetryEngine(
        cost_center="CC-BURST-SERVICE",
        quota_limit_usd=10000.0,
        window_duration_sec=3600.0,
    )

    # Establish baseline hourly rate ~ $10/hour across multiple evaluations
    for t in range(5):
        engine.record_spend(cost_usd=2.5, token_count=5000, model_id="m1", timestamp_sec=100.0 + t * 900)
        engine.get_diagnostic_report(current_timestamp_sec=100.0 + t * 900)

    # Sudden 100x spike
    engine.record_spend(cost_usd=250.0, token_count=500_000, model_id="m1", timestamp_sec=4000.0)
    spike_rep = engine.get_diagnostic_report(current_timestamp_sec=4000.0)
    assert spike_rep.is_surge_anomaly is True


def test_spend_telemetry_prometheus_export():
    engine = SpendTelemetryEngine(cost_center="CC-ANALYTICS", quota_limit_usd=200.0)
    engine.record_spend(cost_usd=50.0, token_count=10000, model_id="m1", timestamp_sec=500.0)

    prom = engine.export_prometheus_metrics(current_timestamp_sec=500.0)
    assert 'token_tariff_spend_dollars_total{cost_center="CC-ANALYTICS"} 50.000000' in prom
    assert 'token_tariff_quota_dollars{cost_center="CC-ANALYTICS"} 200.00' in prom
    assert 'token_tariff_quota_status{cost_center="CC-ANALYTICS"} 1' in prom


def test_spend_telemetry_thread_safety():
    engine = SpendTelemetryEngine(cost_center="CC-MULTI-THREAD", quota_limit_usd=50000.0)
    threads = []
    num_threads = 6
    records_per_thread = 50

    def worker(worker_id: int):
        for i in range(records_per_thread):
            engine.record_spend(
                cost_usd=1.0,
                token_count=1000,
                model_id="m1",
                timestamp_sec=1000.0 + (worker_id * records_per_thread) + i,
            )
            _ = engine.get_diagnostic_report(current_timestamp_sec=2000.0)

    for w in range(num_threads):
        t = threading.Thread(target=worker, args=(w,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    final_report = engine.get_diagnostic_report(current_timestamp_sec=5000.0)
    assert final_report.total_spent_usd == float(num_threads * records_per_thread)
