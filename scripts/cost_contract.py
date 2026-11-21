"""
Token-Tariff: Ingress Cost Allocation Schema Contract & Tag Sanitizer.
Validates multi-tenant LLM token consumption records and sanitizes financial cost center tags.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

TENANT_ID_REGEX = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
COST_CENTER_REGEX = re.compile(r"^CC-[A-Z0-9]{3,16}$")
TAG_KEY_REGEX = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")

# Dangerous injection characters in tags
INJECTION_CHARS = re.compile(r"[\x00-\x1f\x7f\"'`;\\$]")


@dataclass
class ContractError:
    field: str
    code: str
    message: str


@dataclass
class ValidationResult:
    is_valid: bool
    sanitized: Optional[Dict[str, Any]] = None
    errors: List[ContractError] = field(default_factory=list)


def sanitize_tag_value(value: Any, max_len: int = 128) -> str:
    """Sanitizes tag values by stripping control/injection characters and trimming length."""
    if value is None:
        return ""
    val_str = str(value).strip()
    clean = INJECTION_CHARS.sub("", val_str)
    return clean[:max_len]


def sanitize_tag_key(key: Any) -> str:
    """Sanitizes tag keys to lower-case alphanumeric standards."""
    if not key or not isinstance(key, str):
        return ""
    clean = key.strip().lower()
    clean = INJECTION_CHARS.sub("", clean)
    clean = re.sub(r"\s+", "_", clean)
    return clean[:64]


def validate_cost_allocation_payload(
    raw: Any,
    max_tokens: int = 10**9,
    max_cost_usd: float = 1_000_000.0,
) -> ValidationResult:
    """
    Validates ingress token spend records against strict billing schema contracts.
    """
    errors: List[ContractError] = []

    if not isinstance(raw, dict):
        return ValidationResult(
            is_valid=False,
            errors=[ContractError("root", "INVALID_TYPE", "Payload must be a dictionary object")],
        )

    # 1. Tenant ID validation
    tenant_id = raw.get("tenant_id")
    if not tenant_id or not isinstance(tenant_id, str):
        errors.append(ContractError("tenant_id", "REQUIRED_FIELD", "tenant_id is required"))
    elif not TENANT_ID_REGEX.match(tenant_id):
        errors.append(
            ContractError(
                "tenant_id",
                "INVALID_FORMAT",
                f"tenant_id '{tenant_id}' must match pattern ^[a-zA-Z0-9_-]{{1,64}}$",
            )
        )

    # 2. Cost Center validation
    cost_center = raw.get("cost_center")
    if not cost_center or not isinstance(cost_center, str):
        errors.append(ContractError("cost_center", "REQUIRED_FIELD", "cost_center is required"))
    elif not COST_CENTER_REGEX.match(cost_center.upper()):
        errors.append(
            ContractError(
                "cost_center",
                "INVALID_FORMAT",
                f"cost_center '{cost_center}' must match format CC-<CODE> (e.g., CC-ENG101)",
            )
        )

    # 3. Model identifier validation
    model = raw.get("model")
    if not model or not isinstance(model, str):
        errors.append(ContractError("model", "REQUIRED_FIELD", "model identifier is required"))

    # 4. Token counts validation
    tokens_prompt = raw.get("tokens_prompt")
    if tokens_prompt is None or not isinstance(tokens_prompt, int) or tokens_prompt < 0:
        errors.append(
            ContractError(
                "tokens_prompt",
                "INVALID_TOKENS",
                "tokens_prompt must be a non-negative integer",
            )
        )
    elif tokens_prompt > max_tokens:
        errors.append(
            ContractError(
                "tokens_prompt",
                "TOKENS_EXCEEDED",
                f"tokens_prompt {tokens_prompt} exceeds safety limit {max_tokens}",
            )
        )

    tokens_completion = raw.get("tokens_completion")
    if tokens_completion is None or not isinstance(tokens_completion, int) or tokens_completion < 0:
        errors.append(
            ContractError(
                "tokens_completion",
                "INVALID_TOKENS",
                "tokens_completion must be a non-negative integer",
            )
        )
    elif tokens_completion > max_tokens:
        errors.append(
            ContractError(
                "tokens_completion",
                "TOKENS_EXCEEDED",
                f"tokens_completion {tokens_completion} exceeds safety limit {max_tokens}",
            )
        )

    # 5. Cost USD validation
    cost_usd = raw.get("cost_usd")
    if cost_usd is None or not isinstance(cost_usd, (int, float)) or cost_usd < 0:
        errors.append(
            ContractError("cost_usd", "INVALID_COST", "cost_usd must be a non-negative float")
        )
    elif cost_usd > max_cost_usd:
        errors.append(
            ContractError("cost_usd", "COST_EXCEEDED", f"cost_usd exceeds limit {max_cost_usd}")
        )

    # 6. Tags validation & sanitization
    sanitized_tags: Dict[str, str] = {}
    raw_tags = raw.get("tags", {})
    if not isinstance(raw_tags, dict):
        errors.append(ContractError("tags", "INVALID_TYPE", "tags must be a dictionary"))
    elif len(raw_tags) > 32:
        errors.append(ContractError("tags", "TOO_MANY_TAGS", "Maximum 32 tags allowed per payload"))
    else:
        for k, v in raw_tags.items():
            clean_k = sanitize_tag_key(k)
            if not clean_k or not TAG_KEY_REGEX.match(clean_k):
                errors.append(
                    ContractError(
                        f"tags[{k}]",
                        "INVALID_TAG_KEY",
                        f"Tag key '{k}' violates alphanumeric format",
                    )
                )
                continue
            sanitized_tags[clean_k] = sanitize_tag_value(v)

    if errors:
        return ValidationResult(is_valid=False, errors=errors)

    sanitized_record = {
        "tenant_id": str(tenant_id),
        "cost_center": str(cost_center).upper(),
        "model": str(model).strip(),
        "tokens_prompt": int(tokens_prompt),
        "tokens_completion": int(tokens_completion),
        "cost_usd": float(cost_usd),
        "tags": sanitized_tags,
    }

    return ValidationResult(is_valid=True, sanitized=sanitized_record)
