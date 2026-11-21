"""
Token Budget Pre-Allocation Escrow & Multi-Tenant Quota Governor for Token-Tariff.
Prevents concurrent budget overruns by reserving in-flight quota with TTL expiration,
exact delta reconciliation upon inference completion, and automated leakage reclamation.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Dict, Optional


class QuotaException(Exception):
    """Base exception for quota escrow operations."""
    pass


class InsufficientQuotaError(QuotaException):
    """Raised when available balance cannot satisfy requested reservation."""
    pass


class ReservationNotFoundError(QuotaException):
    """Raised when reservation ID is unknown or already resolved."""
    pass


class ReservationExpiredError(QuotaException):
    """Raised when attempting to commit an expired reservation."""
    pass


@dataclass
class QuotaReservation:
    reservation_id: str
    tenant_id: str
    reserved_amount: float
    created_at: float
    expires_at: float
    status: str  # 'ACTIVE', 'COMMITTED', 'CANCELLED', 'EXPIRED'


@dataclass
class TenantQuotaStatus:
    tenant_id: str
    total_budget: float
    reserved_amount: float
    available_balance: float
    active_reservations: int


class QuotaEscrow:
    """
    Manages multi-tenant budget accounts and in-flight pre-allocation escrow.
    """

    def __init__(self, default_ttl_seconds: float = 30.0):
        self.default_ttl = max(1.0, float(default_ttl_seconds))
        # tenant_id -> total remaining budget balance
        self.balances: Dict[str, float] = {}
        # reservation_id -> QuotaReservation
        self.reservations: Dict[str, QuotaReservation] = {}
        self.seq = 0

    def set_budget(self, tenant_id: str, budget: float) -> None:
        """Sets or resets the absolute budget balance for a tenant."""
        if budget < 0.0:
            raise ValueError("Budget must be non-negative")
        self.balances[tenant_id] = float(budget)

    def replenish_budget(self, tenant_id: str, amount: float) -> float:
        """Adds funds/tokens to a tenant's budget."""
        if amount <= 0.0:
            raise ValueError("Replenishment amount must be positive")
        current = self.balances.get(tenant_id, 0.0)
        self.balances[tenant_id] = current + float(amount)
        return self.balances[tenant_id]

    def purge_expired_reservations(self, current_time: Optional[float] = None) -> int:
        """
        Reclaims reserved funds for all reservations that exceeded their TTL.
        """
        now = current_time if current_time is not None else time.time()
        purged = 0

        for res in self.reservations.values():
            if res.status == "ACTIVE" and now > res.expires_at:
                res.status = "EXPIRED"
                purged += 1

        return purged

    def get_tenant_status(self, tenant_id: str, current_time: Optional[float] = None) -> TenantQuotaStatus:
        """
        Calculates real-time tenant quota accounting.
        """
        now = current_time if current_time is not None else time.time()
        self.purge_expired_reservations(now)

        total = self.balances.get(tenant_id, 0.0)

        reserved = 0.0
        active_count = 0
        for res in self.reservations.values():
            if res.tenant_id == tenant_id and res.status == "ACTIVE":
                reserved += res.reserved_amount
                active_count += 1

        available = max(0.0, total - reserved)

        return TenantQuotaStatus(
            tenant_id=tenant_id,
            total_budget=round(total, 6),
            reserved_amount=round(reserved, 6),
            available_balance=round(available, 6),
            active_reservations=active_count,
        )

    def reserve_quota(
        self,
        tenant_id: str,
        estimated_amount: float,
        ttl_seconds: Optional[float] = None,
        current_time: Optional[float] = None,
    ) -> QuotaReservation:
        """
        Reserves estimated quota for an in-flight LLM call.
        Fails if available unreserved balance is insufficient.
        """
        if estimated_amount <= 0.0:
            raise ValueError("Estimated quota must be strictly positive")

        now = current_time if current_time is not None else time.time()
        ttl = float(ttl_seconds) if ttl_seconds is not None else self.default_ttl

        # Purge stale first
        self.purge_expired_reservations(now)

        status = self.get_tenant_status(tenant_id, now)
        if status.available_balance < estimated_amount:
            raise InsufficientQuotaError(
                f"Tenant '{tenant_id}' has available balance {status.available_balance:.4f}, "
                f"cannot reserve requested {estimated_amount:.4f}"
            )

        self.seq += 1
        res_id = f"escrow-{tenant_id}-{self.seq}"
        res = QuotaReservation(
            reservation_id=res_id,
            tenant_id=tenant_id,
            reserved_amount=float(estimated_amount),
            created_at=now,
            expires_at=now + ttl,
            status="ACTIVE",
        )
        self.reservations[res_id] = res
        return res

    def commit_reservation(
        self,
        reservation_id: str,
        actual_amount: float,
        current_time: Optional[float] = None,
    ) -> float:
        """
        Finalizes an in-flight reservation with actual observed spend.
        Deducts actual spend from the tenant's primary balance and releases the reservation.
        """
        if actual_amount < 0.0:
            raise ValueError("Actual amount cannot be negative")

        now = current_time if current_time is not None else time.time()
        res = self.reservations.get(reservation_id)
        if not res:
            raise ReservationNotFoundError(f"Reservation '{reservation_id}' not found")

        if res.status == "EXPIRED" or now > res.expires_at:
            res.status = "EXPIRED"
            raise ReservationExpiredError(f"Reservation '{reservation_id}' expired at {res.expires_at} (current {now})")

        if res.status != "ACTIVE":
            raise ReservationNotFoundError(f"Reservation '{reservation_id}' already closed with status '{res.status}'")

        # Deduct actual amount from primary tenant balance
        tenant_balance = self.balances.get(res.tenant_id, 0.0)
        new_balance = max(0.0, tenant_balance - float(actual_amount))
        self.balances[res.tenant_id] = new_balance

        res.status = "COMMITTED"
        return new_balance

    def cancel_reservation(self, reservation_id: str) -> None:
        """
        Releases an active reservation without deducting spend (e.g. prompt rejected, request aborted).
        """
        res = self.reservations.get(reservation_id)
        if not res or res.status != "ACTIVE":
            raise ReservationNotFoundError(f"Reservation '{reservation_id}' not found or already closed")

        res.status = "CANCELLED"
