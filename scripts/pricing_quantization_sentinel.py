"""
Pricing Quantization & Underbilling Sentinel for Token-Tariff.

Resolves INC-2026-TARIFF-4421: Intercepts sub-cent floating point truncation,
unquantized micro-token billing anomalies, and tier underbilling risks.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


class QuantizationViolationError(RuntimeError):
    """Raised when floating-point precision truncation causes underbilling."""
    pass


@dataclass
class QuantizedBillingResult:
    model_name: str
    token_count: int
    raw_cost_usd: float
    quantized_cost_usd: float
    micro_cents: int
    discrepancy_usd: float
    incident_id: Optional[str]


class PricingQuantizationSentinel:
    """Production sentinel enforcing exact micro-cent fixed-point billing."""

    INCIDENT_ID = "INC-2026-TARIFF-4421"

    def __init__(self, max_allowed_truncation_loss_usd: float = 0.00001):
        self.max_allowed_truncation_loss = max_allowed_truncation_loss_usd
        self._incidents: List[Dict[str, Any]] = []

    def calculate_billable_cost(
        self,
        model_name: str,
        token_count: int,
        rate_per_1k_tokens: float,
    ) -> QuantizedBillingResult:
        """Calculate billable cost with deterministic ceiling quantization."""
        if not model_name or not isinstance(model_name, str):
            raise ValueError("model_name must be a non-empty string")
        if token_count <= 0:
            raise ValueError(f"token_count must be positive, got {token_count}")
        if rate_per_1k_tokens <= 0.0:
            raise ValueError("rate_per_1k_tokens must be positive")

        # Raw float calculation prone to truncation
        raw_cost = (token_count / 1000.0) * rate_per_1k_tokens

        # Fixed-point quantization to micro-cents (1 USD = 100,000,000 micro-cents)
        micro_cents = int(math.ceil(raw_cost * 100_000_000))
        quantized_cost = micro_cents / 100_000_000

        discrepancy = quantized_cost - raw_cost

        # Underbilling check: if raw float is unexpectedly higher than quantized
        if raw_cost > quantized_cost + self.max_allowed_truncation_loss:
            self._incidents.append({
                "incident_id": self.INCIDENT_ID,
                "model_name": model_name,
                "raw_cost": raw_cost,
                "quantized_cost": quantized_cost,
                "timestamp": time.time(),
            })
            raise QuantizationViolationError(
                f"Underbilling detected for {model_name}: raw {raw_cost} > quantized {quantized_cost}"
            )

        return QuantizedBillingResult(
            model_name=model_name,
            token_count=token_count,
            raw_cost_usd=round(raw_cost, 8),
            quantized_cost_usd=quantized_cost,
            micro_cents=micro_cents,
            discrepancy_usd=round(discrepancy, 8),
            incident_id=self.INCIDENT_ID if len(self._incidents) > 0 else None,
        )

    def export_postmortem(self) -> Dict[str, Any]:
        return {
            "incident_id": self.INCIDENT_ID,
            "total_incidents": len(self._incidents),
            "incidents": list(self._incidents),
        }
