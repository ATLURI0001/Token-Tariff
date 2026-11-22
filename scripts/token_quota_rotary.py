"""
Token-Tariff - Multi-Provider Token Quota Rotary & Leaky Bucket Rate Limiter

Implements high-concurrency token and request bucket rate limiting across multiple AI
model providers (OpenAI, Anthropic, Google, Cohere, DeepSeek), featuring continuous
replenishment, automated failover routing, and reservation safety invariants.
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


class QuotaExceededError(Exception):
    """Raised when request or token quota is exhausted across all available providers."""
    pass


class InvalidQuotaConfigError(Exception):
    """Raised on invalid quota parameters."""
    pass


@dataclass
class ProviderQuotaConfig:
    provider: str
    tpm_limit: int  # Tokens per minute
    rpm_limit: int  # Requests per minute
    tier: str = "general"
    priority: int = 1  # 1 is highest priority

    def __post_init__(self):
        if not self.provider or not isinstance(self.provider, str):
            raise InvalidQuotaConfigError("Provider identifier must be a non-empty string")
        if self.tpm_limit <= 0:
            raise InvalidQuotaConfigError(f"tpm_limit must be > 0, got {self.tpm_limit}")
        if self.rpm_limit <= 0:
            raise InvalidQuotaConfigError(f"rpm_limit must be > 0, got {self.rpm_limit}")


@dataclass
class BucketState:
    available_tokens: float
    available_requests: float
    last_updated: float


class TokenQuotaRotary:
    """Manages multi-provider token rate limits with automated fallback rotary."""

    def __init__(self, configs: List[ProviderQuotaConfig]) -> None:
        if not configs:
            raise InvalidQuotaConfigError("Must provide at least one provider quota configuration")

        self._lock = threading.RLock()
        self._configs: Dict[str, ProviderQuotaConfig] = {}
        self._buckets: Dict[str, BucketState] = {}
        now = time.time()

        for c in configs:
            prov = c.provider.lower().strip()
            self._configs[prov] = c
            self._buckets[prov] = BucketState(
                available_tokens=float(c.tpm_limit),
                available_requests=float(c.rpm_limit),
                last_updated=now,
            )

        self._telemetry = {
            "total_reservations": 0,
            "successful_reservations": 0,
            "throttled_requests": 0,
            "failover_routes": 0,
            "total_tokens_allocated": 0,
        }

    def _replenish_bucket(self, provider: str, now: float) -> None:
        """Continuously replenishes tokens and requests based on elapsed time."""
        config = self._configs[provider]
        bucket = self._buckets[provider]

        elapsed = max(0.0, now - bucket.last_updated)
        if elapsed > 0:
            # Tokens replenish at tpm_limit / 60 per second
            token_refill = elapsed * (config.tpm_limit / 60.0)
            bucket.available_tokens = min(float(config.tpm_limit), bucket.available_tokens + token_refill)

            # Requests replenish at rpm_limit / 60 per second
            request_refill = elapsed * (config.rpm_limit / 60.0)
            bucket.available_requests = min(float(config.rpm_limit), bucket.available_requests + request_refill)

            bucket.last_updated = now

    def check_capacity(self, provider: str, requested_tokens: int) -> bool:
        """Non-mutating capacity inquiry."""
        prov = provider.lower().strip()
        with self._lock:
            if prov not in self._configs:
                return False
            now = time.time()
            self._replenish_bucket(prov, now)
            bucket = self._buckets[prov]
            return bucket.available_tokens >= requested_tokens and bucket.available_requests >= 1.0

    def reserve_quota(
        self,
        requested_tokens: int,
        preferred_provider: Optional[str] = None,
        tier: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Attempts to reserve token quota. If preferred provider lacks capacity,
        automatically falls back to same-tier alternative providers by priority.
        """
        if requested_tokens <= 0:
            raise ValueError(f"requested_tokens must be positive, got {requested_tokens}")

        with self._lock:
            self._telemetry["total_reservations"] += 1
            now = time.time()

            # Determine candidate evaluation order
            candidates = list(self._configs.values())
            if tier:
                candidates = [c for c in candidates if c.tier == tier]

            if not candidates:
                self._telemetry["throttled_requests"] += 1
                raise QuotaExceededError(f"No configured providers match tier '{tier}'")

            # Sort by preferred first, then priority ascending
            pref = preferred_provider.lower().strip() if preferred_provider else None
            candidates.sort(key=lambda c: (0 if c.provider == pref else 1, c.priority))

            selected_provider: Optional[str] = None
            is_failover = False

            for c in candidates:
                prov = c.provider
                self._replenish_bucket(prov, now)
                bucket = self._buckets[prov]

                if bucket.available_tokens >= requested_tokens and bucket.available_requests >= 1.0:
                    # Allocate quota
                    bucket.available_tokens -= requested_tokens
                    bucket.available_requests -= 1.0
                    selected_provider = prov
                    if pref and prov != pref:
                        is_failover = True
                    break

            if not selected_provider:
                self._telemetry["throttled_requests"] += 1
                raise QuotaExceededError(
                    f"Quota exhausted across all available providers for {requested_tokens} tokens"
                )

            if is_failover:
                self._telemetry["failover_routes"] += 1

            self._telemetry["successful_reservations"] += 1
            self._telemetry["total_tokens_allocated"] += requested_tokens

            info = {
                "provider": selected_provider,
                "allocated_tokens": requested_tokens,
                "is_failover": is_failover,
                "remaining_tokens": round(self._buckets[selected_provider].available_tokens, 1),
                "remaining_requests": round(self._buckets[selected_provider].available_requests, 1),
            }

            return selected_provider, info

    def release_unspent_tokens(self, provider: str, unspent_tokens: int) -> None:
        """Returns unused reserved tokens back to the provider bucket."""
        if unspent_tokens <= 0:
            return

        prov = provider.lower().strip()
        with self._lock:
            if prov not in self._configs:
                return
            config = self._configs[prov]
            bucket = self._buckets[prov]
            bucket.available_tokens = min(float(config.tpm_limit), bucket.available_tokens + unspent_tokens)

    def get_telemetry(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._telemetry)

    def reset(self) -> None:
        now = time.time()
        with self._lock:
            for prov, c in self._configs.items():
                self._buckets[prov] = BucketState(
                    available_tokens=float(c.tpm_limit),
                    available_requests=float(c.rpm_limit),
                    last_updated=now,
                )
            for k in self._telemetry:
                self._telemetry[k] = 0
