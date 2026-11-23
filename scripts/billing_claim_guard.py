"""Token Billing Claim Hardener & Zero-Width Tokenizer Defense Guard.

Mitigates Denial-of-Wallet (DoW) attacks, strips invisible zero-width unicode
injection characters that desynchronize tokenizers, validates HMAC-SHA256
consumption claims, and bounds multi-tenant burst velocity.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

# Characters commonly used in tokenizer evasion and prompt smuggling
ZERO_WIDTH_PATTERN = re.compile(r"[\u200B\u200C\u200D\u200E\u200F\uFEFF\u00AD\u2060\u2061\u2062\u2063\u2064\u0000]")
TENANT_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{3,64}$")


@dataclass
class BillingClaim:
    claim_id: str
    tenant_id: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    estimated_cost_usd: float
    timestamp: float
    signature: str


@dataclass
class ValidationVerdict:
    valid: boolean
    sanitized_text: Optional[str]
    detected_evasion_chars: int
    errors: List[str]


class BillingClaimGuard:
    """Hardened gatekeeper against token manipulation, claim tampering, and burst overspend."""

    def __init__(
        self,
        signing_secret: str,
        max_burst_tokens_per_min: int = 500000,
        max_claim_age_sec: float = 300.0,
    ) -> None:
        if not signing_secret or len(signing_secret) < 16:
            raise ValueError("signing_secret must be a string of at least 16 characters")

        self.signing_secret = signing_secret.encode("utf-8")
        self.max_burst_tokens_per_min = max_burst_tokens_per_min
        self.max_claim_age_sec = max_claim_age_sec

        self._lock = threading.Lock()
        # tenant_id -> [(timestamp, token_count)]
        self._velocity_windows: Dict[str, List[Tuple[float, int]]] = {}
        self._seen_claims: Set[str] = set()

    def sanitize_tokenizer_input(self, text: str) -> Tuple[str, int]:
        """Detects and strips invisible unicode characters used in evasion attacks."""
        if not text or not isinstance(text, str):
            return "", 0

        matches = ZERO_WIDTH_PATTERN.findall(text)
        cleaned = ZERO_WIDTH_PATTERN.sub("", text)
        return cleaned, len(matches)

    def sign_claim(
        self,
        claim_id: str,
        tenant_id: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        estimated_cost_usd: float,
        timestamp: float,
    ) -> str:
        """Computes deterministic HMAC-SHA256 signature for a billing claim."""
        payload = f"{claim_id}:{tenant_id}:{model}:{prompt_tokens}:{completion_tokens}:{estimated_cost_usd:.6f}:{timestamp:.2f}"
        return hmac.new(self.signing_secret, payload.encode("utf-8"), hashlib.sha256).hexdigest()

    def verify_and_admit_claim(self, claim: BillingClaim, now: Optional[float] = None) -> Tuple[bool, List[str]]:
        """Verifies signature authenticity, prevents replay, and enforces burst limits."""
        t_now = now if now is not None else time.time()
        errors: List[str] = []

        if not claim or not isinstance(claim, BillingClaim):
            return False, ["Claim must be a valid BillingClaim instance"]

        # 1. Identifier format checks
        if not TENANT_ID_PATTERN.match(claim.tenant_id.strip()):
            errors.append(f"Invalid tenant_id format: {claim.tenant_id}")

        if not claim.claim_id or len(claim.claim_id) < 8:
            errors.append("claim_id must be at least 8 characters")

        # 2. Token sanity checks
        if claim.prompt_tokens < 0 or claim.completion_tokens < 0:
            errors.append("Token counts cannot be negative")

        if claim.estimated_cost_usd < 0.0:
            errors.append("estimated_cost_usd cannot be negative")

        # 3. Staleness check
        if abs(t_now - claim.timestamp) > self.max_claim_age_sec:
            errors.append(f"Claim timestamp expired or future-dated outside tolerance ({claim.timestamp})")

        # 4. Signature validation (constant-time)
        expected_sig = self.sign_claim(
            claim.claim_id,
            claim.tenant_id,
            claim.model,
            claim.prompt_tokens,
            claim.completion_tokens,
            claim.estimated_cost_usd,
            claim.timestamp,
        )

        if not hmac.compare_digest(expected_sig, claim.signature):
            errors.append("Cryptographic signature verification failed")

        if errors:
            return False, errors

        # 5. Thread-safe replay and burst velocity checks
        with self._lock:
            if claim.claim_id in self._seen_claims:
                return False, [f"Replay attack detected: claim_id {claim.claim_id} already consumed"]

            total_tokens = claim.prompt_tokens + claim.completion_tokens
            one_min_ago = t_now - 60.0

            window = self._velocity_windows.setdefault(claim.tenant_id, [])
            # Evict entries older than 60s
            valid_window = [(ts, cnt) for ts, cnt in window if ts >= one_min_ago]
            current_burst = sum(cnt for _, cnt in valid_window)

            if current_burst + total_tokens > self.max_burst_tokens_per_min:
                self._velocity_windows[claim.tenant_id] = valid_window
                return False, [
                    f"Burst rate ceiling exceeded for tenant {claim.tenant_id}: "
                    f"{current_burst + total_tokens} > {self.max_burst_tokens_per_min} tokens/min"
                ]

            valid_window.append((t_now, total_tokens))
            self._velocity_windows[claim.tenant_id] = valid_window
            self._seen_claims.add(claim.claim_id)

        return True, []
