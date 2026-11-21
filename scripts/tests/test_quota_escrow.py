import pytest
from scripts.quota_escrow import (
    QuotaEscrow,
    InsufficientQuotaError,
    ReservationNotFoundError,
    ReservationExpiredError,
)

def test_quota_escrow_lifecycle_success():
    escrow = QuotaEscrow(default_ttl_seconds=10.0)
    escrow.set_budget("org-ai", 100.0)

    # Initial status
    status = escrow.get_tenant_status("org-ai")
    assert status.total_budget == 100.0
    assert status.reserved_amount == 0.0
    assert status.available_balance == 100.0
    assert status.active_reservations == 0

    # Reserve 25.0
    res = escrow.reserve_quota("org-ai", 25.0, ttl_seconds=5.0)
    assert res.status == "ACTIVE"
    assert res.reserved_amount == 25.0

    # Available balance should reflect reservation
    status2 = escrow.get_tenant_status("org-ai")
    assert status2.total_budget == 100.0
    assert status2.reserved_amount == 25.0
    assert status2.available_balance == 75.0
    assert status2.active_reservations == 1

    # Commit with actual spend of 20.0 (over-estimated by 5.0)
    new_bal = escrow.commit_reservation(res.reservation_id, 20.0)
    assert new_bal == 80.0

    # Post-commit status: reservation released, balance deducted
    status3 = escrow.get_tenant_status("org-ai")
    assert status3.total_budget == 80.0
    assert status3.reserved_amount == 0.0
    assert status3.available_balance == 80.0
    assert status3.active_reservations == 0


def test_quota_escrow_insufficient_quota_rejection():
    escrow = QuotaEscrow()
    escrow.set_budget("tenant-small", 10.0)

    # First reservation takes 8.0
    escrow.reserve_quota("tenant-small", 8.0)

    # Second reservation for 5.0 must fail (only 2.0 available)
    with pytest.raises(InsufficientQuotaError, match="cannot reserve requested 5.0000"):
        escrow.reserve_quota("tenant-small", 5.0)


def test_quota_escrow_cancellation():
    escrow = QuotaEscrow()
    escrow.set_budget("org-cancel", 50.0)

    res = escrow.reserve_quota("org-cancel", 30.0)
    assert escrow.get_tenant_status("org-cancel").available_balance == 20.0

    escrow.cancel_reservation(res.reservation_id)
    assert escrow.get_tenant_status("org-cancel").available_balance == 50.0

    # Re-cancelling raises error
    with pytest.raises(ReservationNotFoundError):
        escrow.cancel_reservation(res.reservation_id)


def test_quota_escrow_ttl_expiration_reclamation():
    escrow = QuotaEscrow()
    escrow.set_budget("org-ttl", 100.0)
    base_time = 1000.0

    res = escrow.reserve_quota("org-ttl", 40.0, ttl_seconds=5.0, current_time=base_time)
    assert escrow.get_tenant_status("org-ttl", current_time=base_time).available_balance == 60.0

    # Fast forward time past expiry (base_time + 6s)
    future_time = base_time + 6.0
    status = escrow.get_tenant_status("org-ttl", current_time=future_time)
    assert status.reserved_amount == 0.0
    assert status.available_balance == 100.0

    # Committing expired reservation fails
    with pytest.raises(ReservationExpiredError):
        escrow.commit_reservation(res.reservation_id, 35.0, current_time=future_time)
