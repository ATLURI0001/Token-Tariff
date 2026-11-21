"""Unit tests for TokenSpendAnomalyDetector."""

import unittest
from scripts.anomaly_detector import TokenSpendAnomalyDetector


class TestTokenSpendAnomalyDetector(unittest.TestCase):
    def setUp(self):
        self.detector = TokenSpendAnomalyDetector(
            z_score_threshold=3.5,
            surge_ratio_threshold=3.0,
            min_samples=5,
            emergency_cost_ceiling_usd=50.0,
        )

    def test_normal_spending_within_baseline(self):
        history = [1.02, 0.98, 1.05, 1.00, 0.95, 1.03, 0.97]
        current = 1.08
        res = self.detector.evaluate(current, history)

        self.assertEqual(res["status"], "NORMAL")
        self.assertEqual(res["recommended_action"], "ALLOW")
        self.assertFalse(res["is_anomalous"])

    def test_critical_outlier_spike_detection(self):
        # Baseline ~ $1.00
        history = [1.00, 1.05, 0.95, 1.02, 0.98, 1.01]
        # Current spend explodes to $12.50 (12x surge, high Z)
        current = 12.50
        res = self.detector.evaluate(current, history)

        self.assertEqual(res["status"], "CRITICAL_SPIKE")
        self.assertEqual(res["recommended_action"], "THROTTLE_TIER_DOWNGRADE")
        self.assertTrue(res["is_anomalous"])
        self.assertGreater(res["modified_z_score"], 3.5)
        self.assertGreater(res["surge_ratio"], 3.0)

    def test_emergency_ceiling_breach(self):
        history = [0.10, 0.12, 0.11]
        current = 75.00  # Exceeds emergency ceiling $50.00
        res = self.detector.evaluate(current, history)

        self.assertEqual(res["status"], "CRITICAL_SPIKE")
        self.assertEqual(res["recommended_action"], "THROTTLE_TIER_DOWNGRADE")
        self.assertTrue(res["is_anomalous"])

    def test_insufficient_samples_graceful_handling(self):
        history = [1.0, 1.2]
        current = 1.5
        res = self.detector.evaluate(current, history)

        self.assertEqual(res["status"], "NORMAL")
        self.assertFalse(res["is_anomalous"])

    def test_degenerate_inputs_and_nans(self):
        history = [float("nan"), -5.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        res = self.detector.evaluate(-20.0, history)

        self.assertEqual(res["status"], "NORMAL")
        self.assertEqual(res["modified_z_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
