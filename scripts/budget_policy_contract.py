"""
Multi-Tenant LLM Budget Policy Ingress Contract & Schema Evolution Engine.

Validates declarative tenant spending limits, model routing constraints, and auto-evolves
legacy budget schemas to canonical v2 contract representations.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping


CURRENT_SCHEMA_VERSION = "2.0.0"
LEGACY_SCHEMA_VERSIONS = {"1.0.0", "1.1.0"}

TENANT_ID_REGEX = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")
MODEL_NAME_REGEX = re.compile(r"^[a-z0-9][a-z0-9\._-]{1,63}$")


class BudgetPeriod(str, Enum):
    HOURLY = "hourly"
    DAILY = "daily"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


class SoftLimitAction(str, Enum):
    WARN = "warn"
    THROTTLE = "throttle"
    DIVERT_FALLBACK = "divert_to_fallback_model"


class HardLimitAction(str, Enum):
    REJECT = "reject"
    DIVERT_FREE_TIER = "divert_to_free_tier"
    QUEUE = "queue"


@dataclass
class BudgetPolicyContract:
    schema_version: str
    tenant_id: str
    period: BudgetPeriod
    hard_limit_usd: float
    soft_limit_usd: float
    on_soft_limit: SoftLimitAction
    on_hard_limit: HardLimitAction
    allowed_models: list[str]
    fallback_model: str | None = None
    max_tokens_per_request: int = 16384
    cost_per_token_ceiling_usd: float | None = None
    policy_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["period"] = self.period.value
        data["on_soft_limit"] = self.on_soft_limit.value
        data["on_hard_limit"] = self.on_hard_limit.value
        return data


def compute_policy_hash(policy_dict: Mapping[str, Any]) -> str:
    """Compute deterministic SHA-256 fingerprint over normalized policy fields."""
    normalized = {
        "tenant_id": str(policy_dict.get("tenant_id", "")).strip().lower(),
        "period": str(policy_dict.get("period", "")).lower(),
        "hard_limit_usd": round(float(policy_dict.get("hard_limit_usd", 0.0)), 4),
        "soft_limit_usd": round(float(policy_dict.get("soft_limit_usd", 0.0)), 4),
        "on_soft_limit": str(policy_dict.get("on_soft_limit", "")).lower(),
        "on_hard_limit": str(policy_dict.get("on_hard_limit", "")).lower(),
        "allowed_models": sorted(str(m).strip().lower() for m in policy_dict.get("allowed_models", [])),
        "fallback_model": str(policy_dict.get("fallback_model", "")).strip().lower() if policy_dict.get("fallback_model") else None,
        "max_tokens_per_request": int(policy_dict.get("max_tokens_per_request", 16384)),
    }
    raw_json = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw_json.encode("utf-8")).hexdigest()


def parse_and_validate_budget_policy(raw: Mapping[str, Any]) -> BudgetPolicyContract:
    """Validate inbound policy dictionary and auto-evolve legacy schemas to v2."""
    if not isinstance(raw, dict):
        raise TypeError("Budget policy payload must be a JSON dictionary.")

    payload = dict(raw)

    # Schema evolution & migration
    version = str(payload.get("schema_version", "1.0.0")).strip()
    if version in LEGACY_SCHEMA_VERSIONS:
        # Legacy v1 migration: 'budget_usd' -> 'hard_limit_usd', set 'soft_limit_usd' = 0.8 * hard_limit
        if "hard_limit_usd" not in payload and "budget_usd" in payload:
            hard = float(payload.pop("budget_usd"))
            payload["hard_limit_usd"] = hard
            payload["soft_limit_usd"] = payload.get("soft_limit_usd", round(hard * 0.8, 2))
        if "on_soft_limit" not in payload:
            payload["on_soft_limit"] = SoftLimitAction.WARN.value
        if "on_hard_limit" not in payload:
            payload["on_hard_limit"] = HardLimitAction.REJECT.value
        version = CURRENT_SCHEMA_VERSION
    elif version != CURRENT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported budget policy schema_version '{version}'. Expected '{CURRENT_SCHEMA_VERSION}' or legacy {LEGACY_SCHEMA_VERSIONS}.")

    # Validate tenant_id
    tenant_id = str(payload.get("tenant_id", "")).strip().lower()
    if not tenant_id or not TENANT_ID_REGEX.match(tenant_id):
        raise ValueError(f"Invalid tenant_id format: '{tenant_id}'. Must be lowercase alphanumeric with hyphens.")
    if len(tenant_id) > 64:
        raise ValueError("tenant_id exceeds maximum allowed length of 64 characters.")

    # Validate period
    period_str = str(payload.get("period", "monthly")).strip().lower()
    try:
        period = BudgetPeriod(period_str)
    except ValueError:
        raise ValueError(f"Invalid budget period '{period_str}'. Must be one of {[p.value for p in BudgetPeriod]}.")

    # Validate limits
    try:
        hard_limit = float(payload.get("hard_limit_usd", 0.0))
        soft_limit = float(payload.get("soft_limit_usd", 0.0))
    except (ValueError, TypeError):
        raise ValueError("hard_limit_usd and soft_limit_usd must be valid numbers.")

    if hard_limit <= 0 or not (hard_limit < float("inf")):
        raise ValueError("hard_limit_usd must be a strictly positive finite number.")
    if soft_limit <= 0 or not (soft_limit < float("inf")):
        raise ValueError("soft_limit_usd must be a strictly positive finite number.")
    if soft_limit > hard_limit:
        raise ValueError(f"soft_limit_usd ({soft_limit}) cannot exceed hard_limit_usd ({hard_limit}).")

    # Validate actions
    soft_action_str = str(payload.get("on_soft_limit", SoftLimitAction.WARN.value)).strip().lower()
    try:
        on_soft = SoftLimitAction(soft_action_str)
    except ValueError:
        raise ValueError(f"Invalid on_soft_limit action '{soft_action_str}'. Must be one of {[a.value for a in SoftLimitAction]}.")

    hard_action_str = str(payload.get("on_hard_limit", HardLimitAction.REJECT.value)).strip().lower()
    try:
        on_hard = HardLimitAction(hard_action_str)
    except ValueError:
        raise ValueError(f"Invalid on_hard_limit action '{hard_action_str}'. Must be one of {[a.value for a in HardLimitAction]}.")

    # Validate models
    raw_models = payload.get("allowed_models", [])
    if not isinstance(raw_models, list) or len(raw_models) == 0:
        raise ValueError("allowed_models must be a non-empty list of model identifiers.")

    cleaned_models: list[str] = []
    for m in raw_models:
        model_str = str(m).strip().lower()
        if not MODEL_NAME_REGEX.match(model_str):
            raise ValueError(f"Invalid model identifier '{model_str}'. Must match pattern {MODEL_NAME_REGEX.pattern}.")
        if model_str not in cleaned_models:
            cleaned_models.append(model_str)

    # Validate fallback_model if required
    fallback_model: str | None = None
    if on_soft == SoftLimitAction.DIVERT_FALLBACK:
        raw_fb = payload.get("fallback_model")
        if not raw_fb:
            raise ValueError("fallback_model must be specified when on_soft_limit is 'divert_to_fallback_model'.")
        fallback_model = str(raw_fb).strip().lower()
        if not MODEL_NAME_REGEX.match(fallback_model):
            raise ValueError(f"Invalid fallback_model '{fallback_model}'.")
    elif payload.get("fallback_model"):
        fallback_model = str(payload["fallback_model"]).strip().lower()

    # Max tokens
    max_tokens = int(payload.get("max_tokens_per_request", 16384))
    if max_tokens <= 0 or max_tokens > 1000000:
        raise ValueError("max_tokens_per_request must be between 1 and 1,000,000.")

    # Cost ceiling optional check
    ceiling = payload.get("cost_per_token_ceiling_usd")
    cost_ceiling_usd = float(ceiling) if ceiling is not None else None
    if cost_ceiling_usd is not None and cost_ceiling_usd <= 0:
        raise ValueError("cost_per_token_ceiling_usd must be positive.")

    policy = BudgetPolicyContract(
        schema_version=version,
        tenant_id=tenant_id,
        period=period,
        hard_limit_usd=round(hard_limit, 4),
        soft_limit_usd=round(soft_limit, 4),
        on_soft_limit=on_soft,
        on_hard_limit=on_hard,
        allowed_models=cleaned_models,
        fallback_model=fallback_model,
        max_tokens_per_request=max_tokens,
        cost_per_token_ceiling_usd=cost_ceiling_usd,
    )

    policy.policy_hash = compute_policy_hash(policy.to_dict())
    return policy
