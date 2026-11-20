"""Real-time token budget circuit breaker with tiered load shedding and cryptographic audit seals.

Maintains fixed-point nanodollar financial accounting across LLM execution windows.
Prevents runaway recursive agent loops, bill shocks, and API cost spikes through
automated state transitions: CLOSED -> OPEN -> HALF_OPEN.
"""
from __future__ import annotations

import enum
import hashlib
import hmac
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

NANO_DOLLAR_SCALE = 1_000_000_000  # 1 USD = 10^9 nano-dollars


class CircuitState(enum.Enum):
    CLOSED = "CLOSED"        # Normal operations within budget
    OPEN = "OPEN"            # Budget breached, fail-fast non-essential traffic
    HALF_OPEN = "HALF_OPEN"  # Probe window allowing Canary executions


class BudgetTier(enum.Enum):
    GREEN = "GREEN"  # < 70% budget consumed: Unrestricted model routing
    AMBER = "AMBER"  # 70% - 90% budget: Restrict premium tier models
    RED = "RED"      # 90% - 100% budget: Force lowest-tariff fallback models
    EXHAUSTED = "EXHAUSTED"  # >= 100%: Fail-fast reject


@dataclass
class SpendAuditRecord:
    timestamp: float
    tenant_id: str
    tokens: int
    cost_nanodollars: int
    cumulative_spend_nanodollars: int
    circuit_state: CircuitState
    signature: str


class BudgetCircuitBreaker:
    """Production-grade Token Budget Circuit Breaker operating in integer nano-dollars."""

    def __init__(
        self,
        tenant_id: str,
        budget_limit_usd: float,
        cooldown_base_seconds: float = 30.0,
        max_cooldown_seconds: float = 600.0,
        half_open_probe_limit: int = 3,
        audit_secret: str = "token-tariff-audit-key",
    ):
        if budget_limit_usd <= 0:
            raise ValueError("budget_limit_usd must be positive")

        self.tenant_id = tenant_id
        self.budget_limit_nanodollars = int(round(budget_limit_usd * NANO_DOLLAR_SCALE))
        self.cooldown_base_seconds = max(0.001, float(cooldown_base_seconds))
        self.max_cooldown_seconds = max(self.cooldown_base_seconds, float(max_cooldown_seconds))
        self.half_open_probe_limit = max(1, int(half_open_probe_limit))
        self._audit_secret = audit_secret.encode("utf-8")

        self.spent_nanodollars = 0
        self.state = CircuitState.CLOSED
        self.consecutive_breaches = 0
        self.last_state_change = time.time()
        self.half_open_successes = 0

        self.audit_log: List[SpendAuditRecord] = []

    @property
    def budget_limit_usd(self) -> float:
        return self.budget_limit_nanodollars / NANO_DOLLAR_SCALE

    @property
    def spent_usd(self) -> float:
        return self.spent_nanodollars / NANO_DOLLAR_SCALE

    @property
    def consumption_ratio(self) -> float:
        if self.budget_limit_nanodollars == 0:
            return 1.0
        return self.spent_nanodollars / self.budget_limit_nanodollars

    def current_tier(self) -> BudgetTier:
        ratio = self.consumption_ratio
        if ratio >= 1.0:
            return BudgetTier.EXHAUSTED
        if ratio >= 0.90:
            return BudgetTier.RED
        if ratio >= 0.70:
            return BudgetTier.AMBER
        return BudgetTier.GREEN

    def _current_cooldown_seconds(self) -> float:
        backoff_exponent = max(0, min(self.consecutive_breaches - 1, 6))
        delay = self.cooldown_base_seconds * (2 ** backoff_exponent)
        return min(self.max_cooldown_seconds, delay)

    def _check_half_open_transition(self, now: float) -> None:
        if self.state == CircuitState.OPEN:
            cooldown = self._current_cooldown_seconds()
            if (now - self.last_state_change) >= cooldown:
                self.state = CircuitState.HALF_OPEN
                self.last_state_change = now
                self.half_open_successes = 0

    def can_execute(self, estimated_cost_usd: float = 0.0) -> Tuple[bool, str]:
        """Evaluates whether an inference request is permitted by budget and circuit state."""
        now = time.time()
        self._check_half_open_transition(now)

        cost_nanos = max(0, int(round(estimated_cost_usd * NANO_DOLLAR_SCALE)))

        if self.state == CircuitState.OPEN:
            remaining_cooldown = max(0.0, self._current_cooldown_seconds() - (now - self.last_state_change))
            return False, f"Circuit OPEN: budget breached. Retry in {remaining_cooldown:.1f}s"

        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_successes >= self.half_open_probe_limit:
                return False, "Circuit HALF_OPEN probe limit reached, awaiting evaluation"
            return True, "Circuit HALF_OPEN probe execution permitted"

        # CLOSED state checks
        if (self.spent_nanodollars + cost_nanos) > self.budget_limit_nanodollars:
            self._trip(now)
            return False, f"Budget exceeded: requires ${estimated_cost_usd:.4f}, only ${(self.budget_limit_nanodollars - self.spent_nanodollars)/NANO_DOLLAR_SCALE:.4f} remaining"

        return True, "Execution permitted"

    def is_model_allowed(self, model_tariff_per_million_usd: float) -> Tuple[bool, str]:
        """Tiered load shedding: sheds high-tariff models as budget approaches exhaustion."""
        tier = self.current_tier()
        if tier == BudgetTier.EXHAUSTED:
            return False, "All models rejected: budget exhausted"
        if tier == BudgetTier.RED:
            # Under RED tier (>90% spend), reject models exceeding $2.00 per MTok
            if model_tariff_per_million_usd > 2.00:
                return False, f"Model rejected by RED budget tier: tariff ${model_tariff_per_million_usd}/MTok exceeds $2.00 cap"
        elif tier == BudgetTier.AMBER:
            # Under AMBER tier (70-90% spend), reject models exceeding $10.00 per MTok
            if model_tariff_per_million_usd > 10.00:
                return False, f"Model rejected by AMBER budget tier: tariff ${model_tariff_per_million_usd}/MTok exceeds $10.00 cap"
        return True, "Model tier permitted"

    def record_spend(self, tokens: int, actual_cost_usd: float) -> SpendAuditRecord:
        """Records finalized token consumption and updates circuit state with HMAC audit signature."""
        now = time.time()
        cost_nanos = max(0, int(round(actual_cost_usd * NANO_DOLLAR_SCALE)))
        self.spent_nanodollars += cost_nanos

        if self.state == CircuitState.HALF_OPEN:
            self.half_open_successes += 1
            if self.half_open_successes >= self.half_open_probe_limit:
                # If budget was replenished or probe completed under budget limit
                if self.spent_nanodollars < self.budget_limit_nanodollars:
                    self.state = CircuitState.CLOSED
                    self.consecutive_breaches = 0
                    self.last_state_change = now
                else:
                    self._trip(now)

        elif self.state == CircuitState.CLOSED:
            if self.spent_nanodollars >= self.budget_limit_nanodollars:
                self._trip(now)

        sig_payload = f"{now}:{self.tenant_id}:{tokens}:{cost_nanos}:{self.spent_nanodollars}:{self.state.value}"
        signature = hmac.new(self._audit_secret, sig_payload.encode("utf-8"), hashlib.sha256).hexdigest()

        record = SpendAuditRecord(
            timestamp=now,
            tenant_id=self.tenant_id,
            tokens=tokens,
            cost_nanodollars=cost_nanos,
            cumulative_spend_nanodollars=self.spent_nanodollars,
            circuit_state=self.state,
            signature=signature,
        )
        self.audit_log.append(record)
        return record

    def _trip(self, now: float) -> None:
        self.state = CircuitState.OPEN
        self.consecutive_breaches += 1
        self.last_state_change = now

    def reset_budget(self, new_limit_usd: Optional[float] = None) -> None:
        """Replenishes or increases budget and resets circuit state to CLOSED."""
        if new_limit_usd is not None:
            if new_limit_usd <= 0:
                raise ValueError("new_limit_usd must be positive")
            self.budget_limit_nanodollars = int(round(new_limit_usd * NANO_DOLLAR_SCALE))

        self.spent_nanodollars = 0
        self.state = CircuitState.CLOSED
        self.consecutive_breaches = 0
        self.half_open_successes = 0
        self.last_state_change = time.time()

    def get_telemetry(self) -> Dict[str, Any]:
        """Provides a comprehensive telemetry snapshot for observability dashboards."""
        return {
            "tenantId": self.tenant_id,
            "state": self.state.value,
            "tier": self.current_tier().value,
            "budgetLimitUsd": self.budget_limit_usd,
            "spentUsd": self.spent_usd,
            "consumptionRatio": round(self.consumption_ratio, 4),
            "consecutiveBreaches": self.consecutive_breaches,
            "auditRecordsCount": len(self.audit_log),
        }
