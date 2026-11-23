import pytest
from scripts.token_velocity_telemetry import (
    TokenVelocityProfiler,
    VelocityProfilerConfig,
    VelocityHealthStatus,
)


def test_velocity_profiler_empty_state():
    profiler = TokenVelocityProfiler()
    diag = profiler.diagnose()
    assert diag.total_events == 0
    assert diag.total_tokens == 0
    assert diag.tokens_per_minute == 0.0
    assert diag.status == VelocityHealthStatus.HEALTHY
    assert len(diag.recommendations) == 0


def test_velocity_profiler_input_validation():
    profiler = TokenVelocityProfiler()

    with pytest.raises(ValueError):
        profiler.record_usage("", "gemini-1.5-pro", 100, 50, 0.001)

    with pytest.raises(ValueError):
        profiler.record_usage("tenant-1", "", 100, 50, 0.001)

    with pytest.raises(ValueError):
        profiler.record_usage("tenant-1", "gemini-1.5-pro", -10, 50, 0.001)

    with pytest.raises(ValueError):
        profiler.record_usage("tenant-1", "gemini-1.5-pro", 100, 50, -0.05)


def test_velocity_profiler_healthy_and_elevated_burn():
    cfg = VelocityProfilerConfig(window_sec=60.0, budget_cpm_usd=1.0)
    profiler = TokenVelocityProfiler(cfg)

    t0 = 1000.0
    # Ingest 3 events totaling $0.20 over 60s -> $0.20/min (< $0.80 budget) -> HEALTHY
    profiler.record_usage("t1", "m1", 500, 200, 0.05, timestamp=t0)
    profiler.record_usage("t1", "m1", 500, 200, 0.05, timestamp=t0 + 20.0)
    profiler.record_usage("t1", "m1", 500, 200, 0.10, timestamp=t0 + 50.0)

    diag = profiler.diagnose(current_time=t0 + 60.0)
    assert diag.status == VelocityHealthStatus.HEALTHY
    assert diag.total_tokens == 2100

    # Ingest large burst totaling $0.70 more -> total $0.90 / min (90% of $1.0 budget) -> WARNING_ELEVATED
    profiler.record_usage("t1", "m1", 1000, 500, 0.70, timestamp=t0 + 55.0)
    diag_elevated = profiler.diagnose(current_time=t0 + 60.0)
    assert diag_elevated.status == VelocityHealthStatus.WARNING_ELEVATED_BURN_RATE
    assert len(diag_elevated.recommendations) > 0


def test_velocity_profiler_asymmetric_completion_loop():
    cfg = VelocityProfilerConfig(window_sec=120.0, budget_cpm_usd=10.0, asymmetric_ratio_threshold=8.0)
    profiler = TokenVelocityProfiler(cfg)

    t0 = 2000.0
    # Small prompt (100 tokens), huge completion (8,000 tokens) -> 80x ratio
    profiler.record_usage("tenant-agent", "gpt-4o", 100, 8000, 0.12, timestamp=t0)

    diag = profiler.diagnose(current_time=t0 + 10.0)
    assert diag.status == VelocityHealthStatus.WARNING_ASYMMETRIC_COMPLETION
    assert diag.prompt_completion_ratio >= 8.0
    assert any("repetitive or infinite generation loop" in r for r in diag.recommendations)


def test_velocity_profiler_critical_runaway():
    cfg = VelocityProfilerConfig(window_sec=60.0, budget_cpm_usd=2.0)
    profiler = TokenVelocityProfiler(cfg)

    t0 = 3000.0
    # Spend $4.00 within 60s -> $4.00/min (200% of $2.00 budget ceiling) -> CRITICAL
    profiler.record_usage("tenant-rogue", "claude-3-opus", 50000, 25000, 4.00, timestamp=t0)

    diag = profiler.diagnose(current_time=t0 + 60.0)
    assert diag.status == VelocityHealthStatus.CRITICAL_RUNAWAY_CONSUMPTION
    assert any("Critical runaway burn rate" in r for r in diag.recommendations)


def test_velocity_profiler_window_sliding_and_reset():
    cfg = VelocityProfilerConfig(window_sec=30.0)
    profiler = TokenVelocityProfiler(cfg)

    t0 = 5000.0
    profiler.record_usage("t1", "m1", 100, 100, 0.01, timestamp=t0)

    # 40 seconds later, previous event is outside the 30s window
    diag = profiler.diagnose(current_time=t0 + 40.0)
    assert diag.total_events == 0

    # Test reset
    profiler.record_usage("t1", "m1", 100, 100, 0.01, timestamp=t0 + 45.0)
    assert len(profiler.history) == 2
    profiler.reset()
    assert len(profiler.history) == 0
