import time
import pytest
from scripts.ledger_replay_sentinel import (
    LedgerAccount,
    LedgerReplayError,
    LedgerReplaySentinel,
    ReplayViolationCode,
    TransactionReceipt,
)


def test_successful_sequential_debits():
    sentinel = LedgerReplaySentinel()
    sentinel.register_account("acc-prod-1", initial_balance=500.0)

    now = time.time()
    r1 = TransactionReceipt("tx-1", "acc-prod-1", 100.0, 1, now)
    res1 = sentinel.execute_debit(r1, current_time=now)
    assert res1["status"] == "COMMITTED"
    assert res1["remaining_balance"] == 400.0
    assert res1["committed_nonce"] == 1

    r2 = TransactionReceipt("tx-2", "acc-prod-1", 150.0, 2, now + 1.0)
    res2 = sentinel.execute_debit(r2, current_time=now + 1.0)
    assert res2["status"] == "COMMITTED"
    assert res2["remaining_balance"] == 250.0
    assert res2["committed_nonce"] == 2


def test_deposit_and_unregistered_account():
    sentinel = LedgerReplaySentinel()
    with pytest.raises(LedgerReplayError) as exc_info:
        sentinel.deposit("unknown-acc", 100.0)
    assert exc_info.value.code == ReplayViolationCode.ACCOUNT_NOT_FOUND

    sentinel.register_account("acc-prod-2", 50.0)
    new_bal = sentinel.deposit("acc-prod-2", 100.0)
    assert new_bal == 150.0


def test_replay_nonce_regression_rejection():
    sentinel = LedgerReplaySentinel()
    sentinel.register_account("acc-prod-3", 1000.0)
    now = time.time()

    r1 = TransactionReceipt("tx-10", "acc-prod-3", 50.0, 1, now)
    sentinel.execute_debit(r1, current_time=now)

    # Attempt to replay nonce 1
    r_replay = TransactionReceipt("tx-11", "acc-prod-3", 50.0, 1, now + 1.0)
    with pytest.raises(LedgerReplayError) as exc_info:
        sentinel.execute_debit(r_replay, current_time=now + 1.0)
    assert exc_info.value.code == ReplayViolationCode.REPLAY_NONCE_REGRESSION


def test_nonce_sequence_gap_rejection():
    sentinel = LedgerReplaySentinel()
    sentinel.register_account("acc-prod-4", 1000.0)
    now = time.time()

    # Skip nonce 1 and submit nonce 2
    r_gap = TransactionReceipt("tx-20", "acc-prod-4", 50.0, 2, now)
    with pytest.raises(LedgerReplayError) as exc_info:
        sentinel.execute_debit(r_gap, current_time=now)
    assert exc_info.value.code == ReplayViolationCode.NONCE_GAP


def test_solvency_double_spend_rejection():
    sentinel = LedgerReplaySentinel()
    sentinel.register_account("acc-prod-5", 100.0)
    now = time.time()

    # Attempt to debit 150 from 100 balance
    r_overdraft = TransactionReceipt("tx-30", "acc-prod-5", 150.0, 1, now)
    with pytest.raises(LedgerReplayError) as exc_info:
        sentinel.execute_debit(r_overdraft, current_time=now)
    assert exc_info.value.code == ReplayViolationCode.INSUFFICIENT_FUNDS_DOUBLE_SPEND

    # Account balance and nonce must remain unchanged
    acc = sentinel.get_account("acc-prod-5")
    assert acc.balance == 100.0
    assert acc.current_nonce == 0


def test_timestamp_window_expiration_and_drift():
    sentinel = LedgerReplaySentinel(replay_ttl_seconds=300.0, max_future_drift_seconds=30.0)
    sentinel.register_account("acc-prod-6", 1000.0)
    now = time.time()

    # Expired receipt (>300s old)
    r_expired = TransactionReceipt("tx-old", "acc-prod-6", 10.0, 1, now - 350.0)
    with pytest.raises(LedgerReplayError) as exc_info1:
        sentinel.execute_debit(r_expired, current_time=now)
    assert exc_info1.value.code == ReplayViolationCode.EXPIRED_TIMESTAMP

    # Future drift (>30s in future)
    r_future = TransactionReceipt("tx-future", "acc-prod-6", 10.0, 1, now + 60.0)
    with pytest.raises(LedgerReplayError) as exc_info2:
        sentinel.execute_debit(r_future, current_time=now)
    assert exc_info2.value.code == ReplayViolationCode.FUTURE_TIMESTAMP_DRIFT


def test_audit_telemetry_accounting():
    sentinel = LedgerReplaySentinel()
    sentinel.register_account("acc-prod-7", 200.0)
    now = time.time()

    # Valid debit
    sentinel.execute_debit(TransactionReceipt("tx-ok", "acc-prod-7", 50.0, 1, now), current_time=now)

    # Invalid debits
    with pytest.raises(LedgerReplayError):
        sentinel.execute_debit(TransactionReceipt("tx-bad1", "acc-prod-7", 50.0, 1, now), current_time=now)
    with pytest.raises(LedgerReplayError):
        sentinel.execute_debit(TransactionReceipt("tx-bad2", "acc-prod-7", 0.0, 2, now), current_time=now)

    telemetry = sentinel.get_audit_telemetry()
    assert telemetry["active_accounts"] == 1
    assert telemetry["stats"]["total_submitted"] == 3
    assert telemetry["stats"]["total_committed"] == 1
    assert telemetry["stats"]["total_rejected"] == 2
    assert telemetry["stats"]["ERR_REPLAY_NONCE_REGRESSION"] == 1
    assert telemetry["stats"]["ERR_INVALID_TRANSACTION_AMOUNT"] == 1
