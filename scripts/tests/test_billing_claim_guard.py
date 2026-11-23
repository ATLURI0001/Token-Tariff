"""Tests for Token Billing Claim Hardener & Zero-Width Tokenizer Defense."""

import pytest
from scripts.billing_claim_guard import BillingClaim, BillingClaimGuard


def test_sanitize_tokenizer_input():
    guard = BillingClaimGuard("super-secure-production-key-32bytes")

    # Injected with zero-width space (\u200B), joiner (\u200D), and byte order mark (\uFEFF)
    malicious_text = "Summarize\u200B this\u200D contract\uFEFF safely\u0000."
    cleaned, detected = guard.sanitize_tokenizer_input(malicious_text)

    assert cleaned == "Summarize this contract safely."
    assert detected == 4

    clean_text = "Clean prompt text with no evasion"
    c2, d2 = guard.sanitize_tokenizer_input(clean_text)
    assert c2 == clean_text
    assert d2 == 0


def test_sign_and_verify_valid_claim():
    guard = BillingClaimGuard("super-secure-production-key-32bytes")
    now = 1000.0

    sig = guard.sign_claim(
        claim_id="claim-999-alpha",
        tenant_id="tenant-corp-42",
        model="claude-3-5-sonnet-20241022",
        prompt_tokens=1500,
        completion_tokens=400,
        estimated_cost_usd=0.0105,
        timestamp=now,
    )

    claim = BillingClaim(
        claim_id="claim-999-alpha",
        tenant_id="tenant-corp-42",
        model="claude-3-5-sonnet-20241022",
        prompt_tokens=1500,
        completion_tokens=400,
        estimated_cost_usd=0.0105,
        timestamp=now,
        signature=sig,
    )

    valid, errors = guard.verify_and_admit_claim(claim, now=now)
    assert valid is True
    assert len(errors) == 0


def test_reject_signature_tampering_and_replay():
    guard = BillingClaimGuard("super-secure-production-key-32bytes")
    now = 1000.0

    sig = guard.sign_claim("claim-replay-1", "tenant-alpha", "gpt-4o", 100, 50, 0.001, now)
    claim = BillingClaim("claim-replay-1", "tenant-alpha", "gpt-4o", 100, 50, 0.001, now, sig)

    # 1. First admission succeeds
    v1, _ = guard.verify_and_admit_claim(claim, now=now)
    assert v1 is True

    # 2. Replay fails
    v2, errs2 = guard.verify_and_admit_claim(claim, now=now)
    assert v2 is False
    assert any("Replay attack detected" in e for e in errs2)

    # 3. Tampered payload fails signature
    tampered_sig = guard.sign_claim("claim-tamper-1", "tenant-alpha", "gpt-4o", 100, 50, 0.001, now)
    tampered_claim = BillingClaim("claim-tamper-1", "tenant-alpha", "gpt-4o", 9999, 50, 0.001, now, tampered_sig)
    v3, errs3 = guard.verify_and_admit_claim(tampered_claim, now=now)
    assert v3 is False
    assert any("signature verification failed" in e for e in errs3)


def test_reject_burst_rate_ceiling():
    guard = BillingClaimGuard("super-secure-production-key-32bytes", max_burst_tokens_per_min=10000)
    now = 1000.0

    # Claim 1: 8,000 tokens -> Admitted
    sig1 = guard.sign_claim("claim-burst-1", "tenant-burst", "gpt-4o", 5000, 3000, 0.05, now)
    c1 = BillingClaim("claim-burst-1", "tenant-burst", "gpt-4o", 5000, 3000, 0.05, now, sig1)
    v1, _ = guard.verify_and_admit_claim(c1, now=now)
    assert v1 is True

    # Claim 2: 5,000 tokens (8,000 + 5,000 = 13,000 > 10,000 limit) -> Rejected
    sig2 = guard.sign_claim("claim-burst-2", "tenant-burst", "gpt-4o", 3000, 2000, 0.03, now)
    c2 = BillingClaim("claim-burst-2", "tenant-burst", "gpt-4o", 3000, 2000, 0.03, now, sig2)
    v2, errs2 = guard.verify_and_admit_claim(c2, now=now)
    assert v2 is False
    assert any("Burst rate ceiling exceeded" in e for e in errs2)
