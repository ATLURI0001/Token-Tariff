"""Model Pricing Manifest Contract and Billing Unit Sanitizer.

Validates and sanitizes dynamic LLM pricing manifests, enforcing strict boundary
constraints on token unit rates, provider slugs, model slugs, and currency formats.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

VALID_PROVIDER_MODEL_REGEX = re.compile(r"^[a-z0-9_-]+/[a-z0-9_.-]+$")
VALID_CURRENCIES = {"USD", "EUR", "GBP", "CAD", "AUD", "JPY"}
FORBIDDEN_KEYS = {"__proto__", "constructor", "prototype"}


@dataclass
class ModelPricingEntry:
    """Sanitized pricing definition for an individual model."""

    model_id: str
    provider: str
    model_name: str
    currency: str
    input_cost_per_m: float
    output_cost_per_m: float
    cached_input_cost_per_m: Optional[float] = None
    effective_date: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ManifestValidationViolation:
    """Describes a specific schema or constraint violation."""

    field: str
    rule: str
    message: str


@dataclass
class ManifestValidationResult:
    """Contains the overall validation status and sanitized model entries."""

    valid: bool
    violations: List[ManifestValidationViolation] = field(default_factory=list)
    sanitized_entries: List[ModelPricingEntry] = field(default_factory=list)


def validate_and_sanitize_pricing_manifest(
    raw_manifest: Any,
) -> ManifestValidationResult:
    """Validates and sanitizes a complete pricing manifest dictionary or list.

    Args:
        raw_manifest: Raw parsed JSON object or array representing pricing models.

    Returns:
        ManifestValidationResult with sanitized data or violations.
    """
    violations: List[ManifestValidationViolation] = []

    if raw_manifest is None:
        return ManifestValidationResult(
            valid=False,
            violations=[
                ManifestValidationViolation("manifest", "required", "manifest cannot be null")
            ],
        )

    # Normalize entries into a list
    entries_to_validate: List[Dict[str, Any]] = []
    if isinstance(raw_manifest, dict):
        # Could be {"models": [...]} or dict of model_id -> spec
        if "models" in raw_manifest and isinstance(raw_manifest["models"], list):
            entries_to_validate = raw_manifest["models"]
        else:
            for k, v in raw_manifest.items():
                if isinstance(v, dict):
                    copied = dict(v)
                    if "model_id" not in copied:
                        copied["model_id"] = k
                    entries_to_validate.append(copied)
    elif isinstance(raw_manifest, list):
        entries_to_validate = [e for e in raw_manifest if isinstance(e, dict)]
    else:
        return ManifestValidationResult(
            valid=False,
            violations=[
                ManifestValidationViolation(
                    "manifest", "invalid_type", "manifest must be a list or dict of models"
                )
            ],
        )

    if not entries_to_validate:
        return ManifestValidationResult(
            valid=False,
            violations=[
                ManifestValidationViolation(
                    "models", "empty", "manifest must contain at least one valid model entry"
                )
            ],
        )

    if len(entries_to_validate) > 500:
        violations.append(
            ManifestValidationViolation(
                "models", "max_entries", f"manifest exceeds limit of 500 models ({len(entries_to_validate)})"
            )
        )

    sanitized_entries: List[ModelPricingEntry] = []

    for idx, entry in enumerate(entries_to_validate):
        prefix = f"models[{idx}]"

        # Check dangerous prototype keys
        for key in entry.keys():
            if str(key).lower() in FORBIDDEN_KEYS:
                violations.append(
                    ManifestValidationViolation(
                        f"{prefix}.{key}", "prototype_pollution", "forbidden prototype/constructor key detected"
                    )
                )

        # 1. model_id
        raw_id = str(entry.get("model_id", "")).strip().lower()
        if not raw_id:
            violations.append(
                ManifestValidationViolation(f"{prefix}.model_id", "required", "model_id is required")
            )
            continue

        if not VALID_PROVIDER_MODEL_REGEX.match(raw_id):
            violations.append(
                ManifestValidationViolation(
                    f"{prefix}.model_id",
                    "invalid_format",
                    f"model_id '{raw_id}' must match format 'provider/model-name'",
                )
            )
            continue

        provider, model_name = raw_id.split("/", 1)

        # 2. currency
        raw_curr = str(entry.get("currency", "USD")).strip().upper()
        if raw_curr not in VALID_CURRENCIES:
            violations.append(
                ManifestValidationViolation(
                    f"{prefix}.currency",
                    "invalid_currency",
                    f"currency '{raw_curr}' is not a supported ISO currency ({VALID_CURRENCIES})",
                )
            )

        # 3. input_cost_per_m
        input_cost = entry.get("input_cost_per_m")
        sanitized_input_cost = 0.0
        if input_cost is None:
            violations.append(
                ManifestValidationViolation(
                    f"{prefix}.input_cost_per_m", "required", "input_cost_per_m is required"
                )
            )
        else:
            try:
                sanitized_input_cost = float(input_cost)
                if math.isnan(sanitized_input_cost) or math.isinf(sanitized_input_cost):
                    violations.append(
                        ManifestValidationViolation(
                            f"{prefix}.input_cost_per_m", "invalid_numeric", "cost cannot be NaN or Infinite"
                        )
                    )
                elif sanitized_input_cost < 0.0 or sanitized_input_cost > 10_000.0:
                    violations.append(
                        ManifestValidationViolation(
                            f"{prefix}.input_cost_per_m", "bounds_error", "cost must be between 0.0 and 10,000.0"
                        )
                    )
            except (ValueError, TypeError):
                violations.append(
                    ManifestValidationViolation(
                        f"{prefix}.input_cost_per_m", "invalid_type", "input_cost_per_m must be numeric"
                    )
                )

        # 4. output_cost_per_m
        output_cost = entry.get("output_cost_per_m")
        sanitized_output_cost = 0.0
        if output_cost is None:
            violations.append(
                ManifestValidationViolation(
                    f"{prefix}.output_cost_per_m", "required", "output_cost_per_m is required"
                )
            )
        else:
            try:
                sanitized_output_cost = float(output_cost)
                if math.isnan(sanitized_output_cost) or math.isinf(sanitized_output_cost):
                    violations.append(
                        ManifestValidationViolation(
                            f"{prefix}.output_cost_per_m", "invalid_numeric", "cost cannot be NaN or Infinite"
                        )
                    )
                elif sanitized_output_cost < 0.0 or sanitized_output_cost > 10_000.0:
                    violations.append(
                        ManifestValidationViolation(
                            f"{prefix}.output_cost_per_m", "bounds_error", "cost must be between 0.0 and 10,000.0"
                        )
                    )
            except (ValueError, TypeError):
                violations.append(
                    ManifestValidationViolation(
                        f"{prefix}.output_cost_per_m", "invalid_type", "output_cost_per_m must be numeric"
                    )
                )

        # 5. cached_input_cost_per_m (optional)
        sanitized_cached_cost: Optional[float] = None
        if "cached_input_cost_per_m" in entry and entry["cached_input_cost_per_m"] is not None:
            try:
                c_cost = float(entry["cached_input_cost_per_m"])
                if math.isnan(c_cost) or math.isinf(c_cost) or c_cost < 0.0 or c_cost > 10_000.0:
                    violations.append(
                        ManifestValidationViolation(
                            f"{prefix}.cached_input_cost_per_m", "bounds_error", "cached cost invalid"
                        )
                    )
                else:
                    sanitized_cached_cost = c_cost
            except (ValueError, TypeError):
                violations.append(
                    ManifestValidationViolation(
                        f"{prefix}.cached_input_cost_per_m", "invalid_type", "cached cost must be numeric"
                    )
                )

        # 6. effective_date (YYYY-MM-DD)
        effective_date = str(entry.get("effective_date", "")).strip()
        if effective_date:
            try:
                datetime.strptime(effective_date, "%Y-%m-%d")
            except ValueError:
                violations.append(
                    ManifestValidationViolation(
                        f"{prefix}.effective_date",
                        "invalid_date_format",
                        f"effective_date '{effective_date}' must match ISO format YYYY-MM-DD",
                    )
                )

        # Build sanitized record if no violations for this entry
        sanitized_entries.append(
            ModelPricingEntry(
                model_id=raw_id,
                provider=provider,
                model_name=model_name,
                currency=raw_curr,
                input_cost_per_m=round(sanitized_input_cost, 6),
                output_cost_per_m=round(sanitized_output_cost, 6),
                cached_input_cost_per_m=round(sanitized_cached_cost, 6)
                if sanitized_cached_cost is not None
                else None,
                effective_date=effective_date,
            )
        )

    if violations:
        return ManifestValidationResult(valid=False, violations=violations)

    return ManifestValidationResult(valid=True, sanitized_entries=sanitized_entries)
