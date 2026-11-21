"""Robust Token Spend Anomaly Detector & Moving Baseline Surge Filter.

Detects runaway agent token spending and recursive loop cost spikes using
Modified Z-Scores (Median Absolute Deviation) and rolling baseline ratios.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional


class TokenSpendAnomalyDetector:
    """Detects spending anomalies and triggers cost backpressure policies."""

    def __init__(
        self,
        z_score_threshold: float = 3.5,
        surge_ratio_threshold: float = 3.0,
        min_samples: int = 5,
        emergency_cost_ceiling_usd: float = 50.0,
    ):
        self.z_score_threshold = z_score_threshold
        self.surge_ratio_threshold = surge_ratio_threshold
        self.min_samples = min_samples
        self.emergency_cost_ceiling_usd = emergency_cost_ceiling_usd

    def evaluate(self, current_cost_usd: float, history: List[float]) -> Dict[str, object]:
        """Evaluates current spend against historical baseline."""
        if not isinstance(current_cost_usd, (int, float)) or math.isnan(current_cost_usd) or current_cost_usd <= 0:
            return {
                "status": "NORMAL",
                "reason": "Zero or negligible spend",
                "modified_z_score": 0.0,
                "surge_ratio": 0.0,
                "recommended_action": "ALLOW",
                "is_anomalous": False,
            }

        clean_history = [
            float(x) for x in history
            if isinstance(x, (int, float)) and not math.isnan(x) and x >= 0
        ]

        # 1. Absolute Emergency Ceiling
        if current_cost_usd >= self.emergency_cost_ceiling_usd:
            return {
                "status": "CRITICAL_SPIKE",
                "reason": f"Expenditure ${current_cost_usd:.2f} exceeded emergency ceiling ${self.emergency_cost_ceiling_usd:.2f}",
                "modified_z_score": 99.0,
                "surge_ratio": 99.0,
                "recommended_action": "THROTTLE_TIER_DOWNGRADE",
                "is_anomalous": True,
            }

        # Insufficient history -> simple sanity check
        if len(clean_history) < self.min_samples:
            return {
                "status": "NORMAL",
                "reason": "Insufficient historical observations for statistical outlier testing",
                "modified_z_score": 0.0,
                "surge_ratio": 1.0,
                "recommended_action": "ALLOW",
                "is_anomalous": False,
            }

        # 2. Compute Median and Median Absolute Deviation (MAD)
        sorted_hist = sorted(clean_history)
        med = self._median(sorted_hist)
        deviations = [abs(x - med) for x in sorted_hist]
        mad = self._median(sorted(deviations))

        # Compute Modified Z-score: 0.6745 * (x - median) / MAD
        if mad > 1e-6:
            mod_z = 0.6745 * (current_cost_usd - med) / mad
        else:
            # Fallback to mean absolute deviation
            mean_val = sum(clean_history) / len(clean_history)
            mean_dev = sum(abs(x - mean_val) for x in clean_history) / len(clean_history)
            if mean_dev > 1e-6:
                mod_z = 0.6745 * (current_cost_usd - mean_val) / mean_dev
            else:
                mod_z = 0.0 if abs(current_cost_usd - med) < 1e-6 else 10.0

        mod_z = round(max(0.0, mod_z), 4)

        # 3. Compute Surge Ratio relative to baseline mean
        baseline_mean = max(1e-4, sum(clean_history) / len(clean_history))
        surge_ratio = round(current_cost_usd / baseline_mean, 4)

        is_z_anomaly = mod_z >= self.z_score_threshold
        is_surge_anomaly = surge_ratio >= self.surge_ratio_threshold

        if is_z_anomaly and is_surge_anomaly:
            status = "CRITICAL_SPIKE"
            action = "THROTTLE_TIER_DOWNGRADE"
            reason = f"Severe spend anomaly: Z={mod_z:.2f} >= {self.z_score_threshold}, surge={surge_ratio:.2f}x"
        elif is_z_anomaly or is_surge_anomaly:
            status = "ELEVATED_WATCH"
            action = "LOG_AND_NOTIFY"
            reason = f"Elevated spend: Z={mod_z:.2f}, surge={surge_ratio:.2f}x"
        else:
            status = "NORMAL"
            action = "ALLOW"
            reason = "Expenditure is consistent with historical statistical variance"

        return {
            "status": status,
            "reason": reason,
            "modified_z_score": mod_z,
            "surge_ratio": surge_ratio,
            "recommended_action": action,
            "is_anomalous": status in ("ELEVATED_WATCH", "CRITICAL_SPIKE"),
        }

    @staticmethod
    def _median(sorted_list: List[float]) -> float:
        n = len(sorted_list)
        if n == 0:
            return 0.0
        mid = n // 2
        if n % 2 == 1:
            return sorted_list[mid]
        return (sorted_list[mid - 1] + sorted_list[mid]) / 2.0
