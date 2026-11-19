#!/usr/bin/env python3
"""
Unit tests for SlidingTariffMeter in Token-Tariff.
"""

import sys
import unittest
from pathlib import Path

# Ensure scripts directory is on sys.path
SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from sliding_tariff_meter import (
    DiagnosticSnapshot,
    ModelTierRate,
    QuotaExceededError,
    QuotaState,
    SlidingTariffMeter,
    UsageSample,
)


class TestSlidingTariffMeter(unittest.TestCase):
    def setUp(self):
        self.standard_rates = ModelTierRate(
            prompt_usd_per_mtok=3.0,
            completion_usd_per_mtok=15.0,
            cache_write_usd_per_mtok=3.75,
            cache_read_usd_per_mtok=0.30,
        )

    def test_nanodollar_rate_conversion(self):
        rates = self.standard_rates
        self.assertEqual(rates.prompt_nanodollars_per_token(), 3000)
        self.assertEqual(rates.completion_nanodollars_per_token(), 15000)
        self.assertEqual(rates.cache_write_nanodollars_per_token(), 3750)
        self.assertEqual(rates.cache_read_nanodollars_per_token(), 300)

    def test_exact_cost_calculation(self):
        meter = SlidingTariffMeter(window_seconds=60.0)
        # 1,000 prompt tokens: 1,000 * 3,000 = 3,000,000 nano-dollars ($0.003)
        # 500 completion tokens: 500 * 15,000 = 7,500,000 nano-dollars ($0.0075)
        # Total = 10,500,000 nano-dollars = $0.0105
        cost = meter.calculate_cost_nanodollars(
            prompt_tokens=1000,
            completion_tokens=500,
            cache_write_tokens=0,
            cache_read_tokens=0,
            rates=self.standard_rates,
        )
        self.assertEqual(cost, 10_500_000)
        self.assertAlmostEqual(cost / 1e9, 0.0105, places=6)

    def test_sliding_window_eviction(self):
        meter = SlidingTariffMeter(window_seconds=10.0)
        t0 = 1000.0

        # Sample 1 at t=1000
        meter.record(100, 50, rates=self.standard_rates, timestamp=t0)
        diag = meter.get_diagnostics(current_time=t0 + 2.0)
        self.assertEqual(diag.active_sample_count, 1)
        self.assertEqual(diag.window_total_tokens, 150)

        # Sample 2 at t=1005
        meter.record(200, 100, rates=self.standard_rates, timestamp=t0 + 5.0)
        diag = meter.get_diagnostics(current_time=t0 + 6.0)
        self.assertEqual(diag.active_sample_count, 2)
        self.assertEqual(diag.window_total_tokens, 450)

        # Check at t=1011 (Sample 1 at t=1000 has expired, t=1005 remains)
        diag = meter.get_diagnostics(current_time=t0 + 11.0)
        self.assertEqual(diag.active_sample_count, 1)
        self.assertEqual(diag.window_total_tokens, 300)

        # Check at t=1020 (Both expired)
        diag = meter.get_diagnostics(current_time=t0 + 20.0)
        self.assertEqual(diag.active_sample_count, 0)
        self.assertEqual(diag.window_total_tokens, 0)

    def test_soft_warning_and_hard_quota(self):
        # Soft quota $0.02, hard quota $0.03
        meter = SlidingTariffMeter(
            window_seconds=60.0,
            soft_quota_usd=0.02,
            hard_quota_usd=0.03,
        )
        t = 100.0

        # 1,000 completion tokens = 15,000,000 nano-dollars ($0.015) -> Healthy
        meter.record(0, 1000, rates=self.standard_rates, timestamp=t)
        diag = meter.get_diagnostics(current_time=t)
        self.assertEqual(diag.quota_state, QuotaState.HEALTHY)
        self.assertAlmostEqual(diag.quota_consumption_ratio, 0.5, places=2)

        # Add another 500 completion tokens ($0.0075) -> Total $0.0225 -> Soft Warning
        meter.record(0, 500, rates=self.standard_rates, timestamp=t + 1.0)
        diag = meter.get_diagnostics(current_time=t + 1.0)
        self.assertEqual(diag.quota_state, QuotaState.SOFT_WARNING)
        self.assertAlmostEqual(diag.quota_consumption_ratio, 0.75, places=2)

        # Attempt to record 1,000 more ($0.015) -> Total $0.0375 > $0.03 -> Hard Exceeded
        with self.assertRaises(QuotaExceededError) as ctx:
            meter.record(0, 1000, rates=self.standard_rates, timestamp=t + 2.0)
        self.assertIn("Hard quota exceeded", str(ctx.exception))

    def test_velocity_and_exhaustion_eta(self):
        meter = SlidingTariffMeter(
            window_seconds=10.0,
            hard_quota_usd=0.10,
        )
        t0 = 200.0
        # Record 1000 prompt tokens ($0.003)
        meter.record(1000, 0, rates=self.standard_rates, timestamp=t0)
        diag = meter.get_diagnostics(current_time=t0)

        # Velocity over 10s: 100 tokens/s, $0.0003/s
        self.assertEqual(diag.token_velocity_per_sec, 100.0)
        self.assertEqual(diag.cost_velocity_usd_per_sec, 0.0003)
        # Remaining budget = $0.097, ETA = 0.097 / 0.0003 = 323.33s
        self.assertIsNotNone(diag.exhaustion_eta_seconds)
        self.assertGreater(diag.exhaustion_eta_seconds, 300.0)

    def test_negative_tokens_rejected(self):
        meter = SlidingTariffMeter()
        with self.assertRaises(ValueError):
            meter.record(-10, 50)


if __name__ == "__main__":
    unittest.main()
