"""Multi-Tenant Token Usage Ingress Contract & Metering Invariant Guard.

Enforces strict boundary verification and mathematical invariants on LLM usage
events entering billing aggregation, tariff metering, and invoice generation.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

TENANT_ID_REGEX = re.compile(r"^[a-z0-9_\-\.]{3,64}$")
REQUEST_ID_REGEX = re.compile(r"^[a-zA-Z0-9_\-\.]{8,128}$")
MODEL_ID_REGEX = re.compile(r"^[a-z0-9_\-\.\/]{2,128}$")

ALLOWED_TARIFF_TIERS = {"standard", "batch", "priority", "reserved", "free"}
FORBIDDEN_METADATA_KEYS = {"__proto__", "constructor", "prototype"}


class UsageContractError(Exception):
    """Raised when usage records violate schema or mathematical invariants."""

    def __init__(self, code: str, message: str, field_name: str = ""):
        super().__init__(f"[{code}] {message}" + (f" (field: {field_name})" if field_name else ""))
        self.code = code
        self.message = message
        self.field_name = field_name


@dataclass(frozen=True)
class UsageContractLimits:
    """Configures strict boundary limits on usage metering."""

    max_tokens_per_request: int = 4_000_000
    max_batch_size: int = 1_000
    max_cost_cents_per_request: float = 100_000.0  # $1,000 max single request
    allow_zero_token_requests: bool = False


@dataclass
class SanitizedUsageRecord:
    """Defensively sanitized and normalized token usage record."""

    tenant_id: str
    request_id: str
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    cached_prompt_tokens: int
    total_tokens: int
    tariff_tier: str
    cost_cents: float
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchUsageValidationResult:
    """Summary of batch validation execution."""

    valid: bool
    record_count: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_cached_tokens: int
    total_cost_cents: float
    sanitized_records: List[SanitizedUsageRecord]


class UsageMeterContract:
    """Enforces mathematical invariants on LLM usage ingress."""

    def __init__(self, limits: Optional[UsageContractLimits] = None) -> None:
        self.limits = limits or UsageContractLimits()

    def validate_record(self, raw: Dict[str, Any]) -> SanitizedUsageRecord:
        """Validates a single usage event against schema and invariant rules."""
        if not isinstance(raw, dict):
            raise UsageContractError("INVALID_PAYLOAD_TYPE", "Usage payload must be a JSON object dictionary.")

        # 1. Tenant ID
        raw_tenant = str(raw.get("tenant_id") or "").strip().lower()
        if not TENANT_ID_REGEX.match(raw_tenant):
            raise UsageContractError(
                "INVALID_TENANT_ID",
                f"Tenant ID '{raw_tenant}' must be 3-64 lowercase alphanumeric chars with dashes/dots.",
                "tenant_id",
            )

        # 2. Request ID
        raw_req_id = str(raw.get("request_id") or "").strip()
        if not REQUEST_ID_REGEX.match(raw_req_id):
            raise UsageContractError(
                "INVALID_REQUEST_ID",
                f"Request ID '{raw_req_id}' must be 8-128 alphanumeric chars.",
                "request_id",
            )

        # 3. Model Name
        raw_model = str(raw.get("model_name") or "").strip().lower()
        if not MODEL_ID_REGEX.match(raw_model):
            raise UsageContractError(
                "INVALID_MODEL_NAME",
                f"Model identifier '{raw_model}' is malformed.",
                "model_name",
            )

        # 4. Token Invariants
        prompt_tokens = self._parse_positive_int(raw.get("prompt_tokens"), "prompt_tokens")
        completion_tokens = self._parse_positive_int(raw.get("completion_tokens"), "completion_tokens")
        cached_tokens = self._parse_positive_int(raw.get("cached_prompt_tokens", 0), "cached_prompt_tokens")

        total_tokens = prompt_tokens + completion_tokens
        if total_tokens == 0 and not self.limits.allow_zero_token_requests:
            raise UsageContractError(
                "ZERO_TOKEN_REQUEST",
                "Total tokens cannot be zero.",
                "total_tokens",
            )

        if total_tokens > self.limits.max_tokens_per_request:
            raise UsageContractError(
                "TOKEN_LIMIT_EXCEEDED",
                f"Total tokens ({total_tokens}) exceeds maximum allowed ({self.limits.max_tokens_per_request}).",
                "total_tokens",
            )

        # Critical Invariant: Cached prompt tokens cannot exceed prompt tokens
        if cached_tokens > prompt_tokens:
            raise UsageContractError(
                "CACHED_TOKENS_EXCEED_PROMPT",
                f"Cached prompt tokens ({cached_tokens}) cannot exceed total prompt tokens ({prompt_tokens}).",
                "cached_prompt_tokens",
            )

        # 5. Tariff Tier
        raw_tier = str(raw.get("tariff_tier") or "standard").strip().lower()
        if raw_tier not in ALLOWED_TARIFF_TIERS:
            raise UsageContractError(
                "UNSUPPORTED_TARIFF_TIER",
                f"Tariff tier '{raw_tier}' is not supported. Allowed: {sorted(ALLOWED_TARIFF_TIERS)}",
                "tariff_tier",
            )

        # 6. Cost Cents
        raw_cost = raw.get("cost_cents", 0.0)
        try:
            cost_float = float(raw_cost)
        except (TypeError, ValueError):
            raise UsageContractError("INVALID_COST", "cost_cents must be a valid numeric float.", "cost_cents")

        if not math.isfinite(cost_float) or cost_float < 0.0:
            raise UsageContractError("NEGATIVE_COST", "cost_cents cannot be negative or non-finite.", "cost_cents")

        if cost_float > self.limits.max_cost_cents_per_request:
            raise UsageContractError(
                "COST_CEILING_BREACH",
                f"Cost {cost_float} cents exceeds maximum single request limit {self.limits.max_cost_cents_per_request}.",
                "cost_cents",
            )

        # 7. Metadata Sanitization
        clean_metadata: Dict[str, Any] = {}
        raw_metadata = raw.get("metadata")
        if isinstance(raw_metadata, dict):
            for k, v in raw_metadata.items():
                clean_k = str(k).strip()
                if clean_k in FORBIDDEN_METADATA_KEYS or clean_k.startswith("__"):
                    continue
                if isinstance(v, (str, int, float, bool)):
                    clean_metadata[clean_k] = v

        timestamp_str = str(raw.get("timestamp") or datetime.now(timezone.utc).isoformat())

        return SanitizedUsageRecord(
            tenant_id=raw_tenant,
            request_id=raw_req_id,
            model_name=raw_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_prompt_tokens=cached_tokens,
            total_tokens=total_tokens,
            tariff_tier=raw_tier,
            cost_cents=round(cost_float, 4),
            timestamp=timestamp_str,
            metadata=clean_metadata,
        )

    def validate_batch(self, batch: List[Dict[str, Any]]) -> BatchUsageValidationResult:
        """Validates a batch of usage records, enforcing size limits and deduplicating request IDs."""
        if not isinstance(batch, list):
            raise UsageContractError("INVALID_BATCH_TYPE", "Batch payload must be a list of usage records.")

        if len(batch) == 0:
            raise UsageContractError("EMPTY_BATCH", "Batch cannot be empty.")

        if len(batch) > self.limits.max_batch_size:
            raise UsageContractError(
                "BATCH_SIZE_EXCEEDED",
                f"Batch size {len(batch)} exceeds maximum allowed {self.limits.max_batch_size}.",
            )

        sanitized_records: List[SanitizedUsageRecord] = []
        seen_requests: Set[str] = set()

        total_prompt = 0
        total_completion = 0
        total_cached = 0
        total_cost = 0.0

        for i, item in enumerate(batch):
            try:
                rec = self.validate_record(item)
            except UsageContractError as err:
                raise UsageContractError(
                    err.code,
                    f"Batch item at index {i} failed validation: {err.message}",
                    err.field_name,
                ) from err

            if rec.request_id in seen_requests:
                raise UsageContractError(
                    "DUPLICATE_REQUEST_ID_IN_BATCH",
                    f"Duplicate request_id '{rec.request_id}' encountered in single batch at index {i}.",
                    "request_id",
                )
            seen_requests.add(rec.request_id)

            total_prompt += rec.prompt_tokens
            total_completion += rec.completion_tokens
            total_cached += rec.cached_prompt_tokens
            total_cost += rec.cost_cents
            sanitized_records.append(rec)

        return BatchUsageValidationResult(
            valid=True,
            record_count=len(sanitized_records),
            total_prompt_tokens=total_prompt,
            total_completion_tokens=total_completion,
            total_cached_tokens=total_cached,
            total_cost_cents=round(total_cost, 4),
            sanitized_records=sanitized_records,
        )

    def _parse_positive_int(self, val: Any, field_name: str) -> int:
        try:
            num = int(val)
        except (TypeError, ValueError):
            raise UsageContractError("INVALID_INTEGER", f"Field '{field_name}' must be an integer.", field_name)

        if num < 0:
            raise UsageContractError("NEGATIVE_TOKENS", f"Field '{field_name}' cannot be negative ({num}).", field_name)

        return num
