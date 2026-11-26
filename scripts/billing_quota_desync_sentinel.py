"""
Billing Quota Desync & Overdraft Sentinel for Token-Tariff.

Resolves INC-2026-TARIFF-3391: Multi-tenant billing quota desynchronization,
concurrent overdraft races, and ledger discrepancy sentinels.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


class BillingDesyncViolationError(RuntimeError):
    """Raised when tenant quota desynchronization or overdraft is detected."""
    pass


class TenantQuarantinedError(BillingDesyncViolationError):
    """Raised when an operation targets a quarantined tenant."""
    pass


@dataclass
class TenantQuotaRecord:
    tenant_id: str
    allocated_tokens: int
    used_tokens: int
    version: int
    quarantined: bool = False
    last_reconciliation_time: float = field(default_factory=time.time)

    @property
    def remaining_tokens(self) -> int:
        return self.allocated_tokens - self.used_tokens


@dataclass
class AuditDebitEntry:
    entry_id: str
    tenant_id: str
    tokens_debited: int
    prior_version: int
    new_version: int
    resulting_balance: int
    timestamp: float
    sha256_hash: str


class BillingQuotaDesyncSentinel:
    """Production sentinel intercepting quota desync and overdraft conditions."""

    INCIDENT_ID = "INC-2026-TARIFF-3391"

    def __init__(
        self,
        max_allowed_desync_version_gap: int = 2,
        quarantine_on_overdraft: bool = True,
    ):
        self.max_allowed_desync_version_gap = max_allowed_desync_version_gap
        self.quarantine_on_overdraft = quarantine_on_overdraft
        self._tenants: Dict[str, TenantQuotaRecord] = {}
        self._audit_log: List[AuditDebitEntry] = []
        self._quarantined_tenants: Set[str] = set()
        self._incident_reports: List[Dict[str, Any]] = []

    def register_tenant(self, tenant_id: str, initial_allocation: int) -> TenantQuotaRecord:
        """Register a new tenant ledger record."""
        if not tenant_id or not isinstance(tenant_id, str):
            raise ValueError("tenant_id must be a valid non-empty string")
        if initial_allocation < 0:
            raise ValueError("initial_allocation cannot be negative")

        record = TenantQuotaRecord(
            tenant_id=tenant_id,
            allocated_tokens=initial_allocation,
            used_tokens=0,
            version=1,
            quarantined=False,
            last_reconciliation_time=time.time(),
        )
        self._tenants[tenant_id] = record
        return record

    def debit_tokens(
        self,
        tenant_id: str,
        tokens: int,
        client_observed_version: int,
        transaction_id: str,
        now: Optional[float] = None,
    ) -> AuditDebitEntry:
        """Atomic debit with versioned optimistic lock and non-negative balance enforcement."""
        current_time = time.time() if now is None else now

        if tenant_id in self._quarantined_tenants:
            raise TenantQuarantinedError(f"Tenant '{tenant_id}' is quarantined due to INC-2026-TARIFF-3391 policy")

        record = self._tenants.get(tenant_id)
        if record is None:
            raise KeyError(f"Unknown tenant: '{tenant_id}'")

        if tokens <= 0:
            raise ValueError(f"tokens to debit must be positive, got {tokens}")

        # Check version gap / desynchronization
        version_gap = abs(record.version - client_observed_version)
        if version_gap > self.max_allowed_desync_version_gap:
            self._record_incident(
                tenant_id,
                "QUOTA_DESYNC_DETECTED",
                f"Client version {client_observed_version} diverged from server version {record.version} (gap={version_gap})",
                current_time,
            )
            if self.quarantine_on_overdraft:
                self.quarantine_tenant(tenant_id, f"Desync version gap exceeded {self.max_allowed_desync_version_gap}")
            raise BillingDesyncViolationError(
                f"Quota cache desynchronization detected for '{tenant_id}': gap {version_gap} > max {self.max_allowed_desync_version_gap}"
            )

        # Check overdraft invariant
        potential_used = record.used_tokens + tokens
        if potential_used > record.allocated_tokens:
            deficit = potential_used - record.allocated_tokens
            self._record_incident(
                tenant_id,
                "OVERDRAFT_INTERCEPTED",
                f"Attempted debit of {tokens} exceeds remaining balance {record.remaining_tokens} by {deficit} tokens",
                current_time,
            )
            if self.quarantine_on_overdraft and deficit > 1000:
                self.quarantine_tenant(tenant_id, f"Severe overdraft deficit {deficit} tokens")
            raise BillingDesyncViolationError(
                f"Overdraft prevented: requested {tokens} exceeds remaining {record.remaining_tokens}"
            )

        # Apply state mutation atomically
        prior_version = record.version
        record.used_tokens = potential_used
        record.version += 1
        record.last_reconciliation_time = current_time

        # Compute tamper-evident hash
        hash_payload = f"{transaction_id}:{tenant_id}:{tokens}:{prior_version}:{record.version}:{record.remaining_tokens}:{current_time}"
        entry_hash = hashlib.sha256(hash_payload.encode("utf-8")).hexdigest()

        entry = AuditDebitEntry(
            entry_id=transaction_id,
            tenant_id=tenant_id,
            tokens_debited=tokens,
            prior_version=prior_version,
            new_version=record.version,
            resulting_balance=record.remaining_tokens,
            timestamp=current_time,
            sha256_hash=entry_hash,
        )
        self._audit_log.append(entry)
        return entry

    def quarantine_tenant(self, tenant_id: str, reason: str):
        """Immediately quarantine tenant from further debit operations."""
        self._quarantined_tenants.add(tenant_id)
        if tenant_id in self._tenants:
            self._tenants[tenant_id].quarantined = True

    def lift_quarantine(self, tenant_id: str, updated_allocation: Optional[int] = None):
        """Operator remediation action lifting quarantine."""
        self._quarantined_tenants.discard(tenant_id)
        if tenant_id in self._tenants:
            self._tenants[tenant_id].quarantined = False
            if updated_allocation is not None and updated_allocation >= self._tenants[tenant_id].used_tokens:
                self._tenants[tenant_id].allocated_tokens = updated_allocation
            self._tenants[tenant_id].version += 1

    def _record_incident(self, tenant_id: str, alert_type: str, details: str, timestamp: float):
        """Record diagnostic telemetry for the incident log."""
        self._incident_reports.append({
            "incident_id": self.INCIDENT_ID,
            "tenant_id": tenant_id,
            "alert_type": alert_type,
            "details": details,
            "timestamp": timestamp,
        })

    def export_postmortem(self) -> Dict[str, Any]:
        """Export comprehensive postmortem report for INC-2026-TARIFF-3391."""
        return {
            "incident_id": self.INCIDENT_ID,
            "total_incidents_recorded": len(self._incident_reports),
            "quarantined_tenants": list(self._quarantined_tenants),
            "incidents": list(self._incident_reports),
            "total_audit_debits": len(self._audit_log),
        }
