"""
Unit tests for Billing Quota Desync & Overdraft Sentinel (INC-2026-TARIFF-3391).
"""

import pytest

from scripts.billing_quota_desync_sentinel import (
    AuditDebitEntry,
    BillingDesyncViolationError,
    BillingQuotaDesyncSentinel,
    TenantQuarantinedError,
    TenantQuotaRecord,
)


def test_nominal_token_debit_flow():
    sentinel = BillingQuotaDesyncSentinel(max_allowed_desync_version_gap=2)
    sentinel.register_tenant("tenant-alpha", initial_allocation=10000)

    entry = sentinel.debit_tokens(
        tenant_id="tenant-alpha",
        tokens=2500,
        client_observed_version=1,
        transaction_id="tx-001",
        now=100.0,
    )

    assert entry.tokens_debited == 2500
    assert entry.prior_version == 1
    assert entry.new_version == 2
    assert entry.resulting_balance == 7500
    assert len(entry.sha256_hash) == 64


def test_overdraft_prevention_invariant():
    sentinel = BillingQuotaDesyncSentinel()
    sentinel.register_tenant("tenant-beta", initial_allocation=500)

    # Attempting to debit more than remaining balance
    with pytest.raises(BillingDesyncViolationError) as exc_info:
        sentinel.debit_tokens(
            tenant_id="tenant-beta",
            tokens=600,
            client_observed_version=1,
            transaction_id="tx-overdraft",
        )
    assert "Overdraft prevented" in str(exc_info.value)


def test_version_desync_gap_quarantine():
    sentinel = BillingQuotaDesyncSentinel(max_allowed_desync_version_gap=2, quarantine_on_overdraft=True)
    sentinel.register_tenant("tenant-gamma", initial_allocation=50000)

    # Client has version 1, but server version is already bumped to 5
    sentinel._tenants["tenant-gamma"].version = 5

    with pytest.raises(BillingDesyncViolationError) as exc_info:
        sentinel.debit_tokens(
            tenant_id="tenant-gamma",
            tokens=100,
            client_observed_version=1,
            transaction_id="tx-stale",
        )
    assert "Quota cache desynchronization detected" in str(exc_info.value)

    # Subsequent debits fail with TenantQuarantinedError
    with pytest.raises(TenantQuarantinedError):
        sentinel.debit_tokens(
            tenant_id="tenant-gamma",
            tokens=10,
            client_observed_version=5,
            transaction_id="tx-blocked",
        )


def test_remediation_and_quarantine_lift():
    sentinel = BillingQuotaDesyncSentinel()
    sentinel.register_tenant("tenant-delta", initial_allocation=1000)
    sentinel.quarantine_tenant("tenant-delta", "Operator precaution")

    with pytest.raises(TenantQuarantinedError):
        sentinel.debit_tokens("tenant-delta", 100, 1, "tx-fail")

    # Operator lifts quarantine and adds allocation
    sentinel.lift_quarantine("tenant-delta", updated_allocation=2000)
    entry = sentinel.debit_tokens("tenant-delta", 200, sentinel._tenants["tenant-delta"].version, "tx-success")
    assert entry.resulting_balance == 1800


def test_tamper_evident_audit_log_and_postmortem():
    sentinel = BillingQuotaDesyncSentinel()
    sentinel.register_tenant("tenant-audit", initial_allocation=5000)

    for i in range(1, 4):
        version = sentinel._tenants["tenant-audit"].version
        sentinel.debit_tokens("tenant-audit", 500, version, f"tx-seq-{i}")

    postmortem = sentinel.export_postmortem()
    assert postmortem["incident_id"] == "INC-2026-TARIFF-3391"
    assert postmortem["total_audit_debits"] == 3


def test_defensive_input_validation():
    sentinel = BillingQuotaDesyncSentinel()
    with pytest.raises(ValueError):
        sentinel.register_tenant("", 100)
    with pytest.raises(ValueError):
        sentinel.register_tenant("valid", -10)
    with pytest.raises(KeyError):
        sentinel.debit_tokens("non-existent", 100, 1, "tx")
