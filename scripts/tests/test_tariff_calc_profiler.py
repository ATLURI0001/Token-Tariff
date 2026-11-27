"""Tests for tariff_calc_profiler."""
import sys
sys.path.insert(0, "scripts")
from tariff_calc_profiler import TariffCalcProfiler


def test_tariff_calc_profiler():
    p = TariffCalcProfiler()
    p.record(4.0); p.record(6.0)
    assert p.avg_ms() == 5.0
