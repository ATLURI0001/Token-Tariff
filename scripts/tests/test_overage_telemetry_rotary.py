from scripts.overage_telemetry_rotary import OverageTelemetryRotary


def test_overage_telemetry_rotary():
    rotary = OverageTelemetryRotary(4)
    rotary.record("t1", 500, 1000)
    rotary.record("t2", 1500, 1000)
    rotary.record("t3", 800, 1000)
    rotary.record("t4", 2000, 1000)
    assert rotary.get_overage_rate_pct() == 50.0
