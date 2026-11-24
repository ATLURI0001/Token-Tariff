"""
Token Ledger Double-Spend & Transaction Replay Sentinel.

Enforces strict monotonic nonce sequencing, cryptographic receipt deduplication,
pre-deduction solvency invariants, and timestamp window validation to eliminate
double-spending and transaction replay in asynchronous token billing ledgers.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Set


class ReplayViolationCode(str, Enum):
    ACCOUNT_NOT_FOUND = "ERR_ACCOUNT_NOT_FOUND"
    EXPIRED_TIMESTAMP = "ERR_EXPIRED_TX_TIMESTAMP"
    FUTURE_TIMESTAMP_DRIFT = "ERR_FUTURE_TIMESTAMP_DRIFT"
    REPLAY_NONCE_REGRESSION = "ERR_REPLAY_NONCE_REGRESSION"
    NONCE_GAP = "ERR_NONCE_SEQUENCE_GAP"
    DUPLICATE_RECEIPT_DIGEST = "ERR_DUPLICATE_TX_RECEIPT"
    INSUFFICIENT_FUNDS_DOUBLE_SPEND = "ERR_INSUFFICIENT_FUNDS_DOUBLE_SPEND"
    INVALID_AMOUNT = "ERR_INVALID_TRANSACTION_AMOUNT"


class LedgerReplayError(Exception):
    """Raised when a token transaction violates double-spend or replay invariants."""

    def __init__(self, code: ReplayViolationCode, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(f"[{code.value}] {message}")
        self.code = code
        self.message = message
        self.details = details or {}


@dataclass
class TransactionReceipt:
    tx_id: str
    account_id: str
    amount: float
    nonce: int
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def compute_digest(self) -> str:
        """Computes deterministic canonical SHA-256 fingerprint for transaction receipt."""
        payload = f"{self.tx_id}:{self.account_id}:{self.amount:.6f}:{self.nonce}:{self.timestamp:.4f}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class LedgerAccount:
    account_id: str
    balance: float
    current_nonce: int = 0
    committed_hashes: Set[str] = field(default_factory=set)


class LedgerReplaySentinel:
    """
    Guards token accounting ledgers against double-spending and replay attacks.
    """

    def __init__(
        self,
        replay_ttl_seconds: float = 300.0,
        max_future_drift_seconds: float = 30.0,
    ) -> None:
        self.replay_ttl_seconds = replay_ttl_seconds
        self.max_future_drift_seconds = max_future_drift_seconds
        self._accounts: Dict[str, LedgerAccount] = {}
        self._stats: Dict[str, int] = {
            "total_submitted": 0,
            "total_committed": 0,
            "total_rejected": 0,
            ReplayViolationCode.ACCOUNT_NOT_FOUND.value: 0,
            ReplayViolationCode.EXPIRED_TIMESTAMP.value: 0,
            ReplayViolationCode.FUTURE_TIMESTAMP_DRIFT.value: 0,
            ReplayViolationCode.REPLAY_NONCE_REGRESSION.value: 0,
            ReplayViolationCode.NONCE_GAP.value: 0,
            ReplayViolationCode.DUPLICATE_RECEIPT_DIGEST.value: 0,
            ReplayViolationCode.INSUFFICIENT_FUNDS_DOUBLE_SPEND.value: 0,
            ReplayViolationCode.INVALID_AMOUNT.value: 0,
        }

    def register_account(self, account_id: str, initial_balance: float = 0.0) -> LedgerAccount:
        if not account_id or not isinstance(account_id, str):
            raise ValueError("account_id must be a non-empty string")
        if initial_balance < 0.0:
            raise ValueError("initial_balance cannot be negative")

        account = LedgerAccount(account_id=account_id.strip(), balance=float(initial_balance))
        self._accounts[account.account_id] = account
        return account

    def deposit(self, account_id: str, amount: float) -> float:
        if account_id not in self._accounts:
            raise LedgerReplayError(
                ReplayViolationCode.ACCOUNT_NOT_FOUND,
                f"Account '{account_id}' is not registered",
                {"account_id": account_id},
            )
        if amount <= 0.0:
            raise LedgerReplayError(
                ReplayViolationCode.INVALID_AMOUNT,
                f"Deposit amount must be strictly positive, got {amount}",
                {"amount": amount},
            )

        self._accounts[account_id].balance += amount
        return self._accounts[account_id].balance

    def execute_debit(self, receipt: TransactionReceipt, current_time: Optional[float] = None) -> Dict[str, Any]:
        """
        Atomically validates and executes a token ledger debit transaction.
        """
        self._stats["total_submitted"] += 1
        now = current_time if current_time is not None else time.time()

        # 1. Account existence check
        account = self._accounts.get(receipt.account_id)
        if not account:
            return self._reject(
                ReplayViolationCode.ACCOUNT_NOT_FOUND,
                f"Account '{receipt.account_id}' not found in ledger",
                {"account_id": receipt.account_id, "tx_id": receipt.tx_id},
            )

        # 2. Amount validity check
        if receipt.amount <= 0.0 or not isinstance(receipt.amount, (int, float)):
            return self._reject(
                ReplayViolationCode.INVALID_AMOUNT,
                f"Debit amount must be strictly positive, got {receipt.amount}",
                {"amount": receipt.amount, "tx_id": receipt.tx_id},
            )

        # 3. Timestamp expiration & future drift window
        age = now - receipt.timestamp
        if age > self.replay_ttl_seconds:
            return self._reject(
                ReplayViolationCode.EXPIRED_TIMESTAMP,
                f"Transaction receipt age ({age:.1f}s) exceeds replay TTL of {self.replay_ttl_seconds}s",
                {"age": age, "ttl": self.replay_ttl_seconds, "tx_id": receipt.tx_id},
            )
        if age < -self.max_future_drift_seconds:
            return self._reject(
                ReplayViolationCode.FUTURE_TIMESTAMP_DRIFT,
                f"Transaction receipt timestamp is {-age:.1f}s into the future (max drift: {self.max_future_drift_seconds}s)",
                {"drift": -age, "tx_id": receipt.tx_id},
            )

        # 4. Strict monotonic nonce validation
        expected_nonce = account.current_nonce + 1
        if receipt.nonce <= account.current_nonce:
            return self._reject(
                ReplayViolationCode.REPLAY_NONCE_REGRESSION,
                f"Transaction replay detected: nonce {receipt.nonce} <= current nonce {account.current_nonce}",
                {"submitted_nonce": receipt.nonce, "current_nonce": account.current_nonce, "tx_id": receipt.tx_id},
            )
        if receipt.nonce > expected_nonce:
            return self._reject(
                ReplayViolationCode.NONCE_GAP,
                f"Nonce sequence gap detected: expected {expected_nonce}, got {receipt.nonce}",
                {"expected_nonce": expected_nonce, "submitted_nonce": receipt.nonce, "tx_id": receipt.tx_id},
            )

        # 5. Cryptographic receipt fingerprint deduplication
        digest = receipt.compute_digest()
        if digest in account.committed_hashes:
            return self._reject(
                ReplayViolationCode.DUPLICATE_RECEIPT_DIGEST,
                f"Duplicate transaction receipt hash: '{digest}' already committed",
                {"digest": digest, "tx_id": receipt.tx_id},
            )

        # 6. Pre-deduction solvency check (double-spend mitigation)
        if account.balance < receipt.amount:
            return self._reject(
                ReplayViolationCode.INSUFFICIENT_FUNDS_DOUBLE_SPEND,
                f"Solvency breach: balance {account.balance:.4f} < requested debit {receipt.amount:.4f}",
                {"balance": account.balance, "amount": receipt.amount, "tx_id": receipt.tx_id},
            )

        # 7. Atomic commit
        account.balance -= receipt.amount
        account.current_nonce = receipt.nonce
        account.committed_hashes.add(digest)

        self._stats["total_committed"] += 1
        return {
            "status": "COMMITTED",
            "tx_id": receipt.tx_id,
            "account_id": receipt.account_id,
            "amount_debited": receipt.amount,
            "remaining_balance": account.balance,
            "committed_nonce": account.current_nonce,
            "digest": digest,
        }

    def _reject(self, code: ReplayViolationCode, message: str, details: Dict[str, Any]) -> None:
        self._stats["total_rejected"] += 1
        self._stats[code.value] += 1
        raise LedgerReplayError(code, message, details)

    def get_account(self, account_id: str) -> Optional[LedgerAccount]:
        return self._accounts.get(account_id)

    def get_audit_telemetry(self) -> Dict[str, Any]:
        return {
            "active_accounts": len(self._accounts),
            "stats": dict(self._stats),
        }
