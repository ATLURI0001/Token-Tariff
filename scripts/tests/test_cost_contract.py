"""
Unit tests for Token-Tariff Ingress Cost Allocation Contract & Tag Sanitizer.
"""

import sys
from pathlib import Path
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from cost_contract import (
    validate_cost_allocation_payload,
    sanitize_tag_key,
    sanitize_tag_value,
)


def test_validate_cost_allocation_valid_payload():
    payload = {
        "tenant_id": "tenant-prod-42",
        "cost_center": "CC-AI99",
        "model": "claude-3-5-sonnet",
        "tokens_prompt": 1500,
        "tokens_completion": 450,
        "cost_usd": 0.0125,
        "tags": {
            "environment": "production",
            "feature_flag": "deep_rag",
            "user_id": "usr_9981",
        },
    }

    res = validate_cost_allocation_payload(payload)
    assert res.is_valid is True
    assert res.sanitized is not None
    assert res.sanitized["cost_center"] == "CC-AI99"
    assert res.sanitized["tags"]["environment"] == "production"


def test_validate_cost_allocation_invalid_fields():
    payload = {
        "tenant_id": "invalid tenant with spaces!",
        "cost_center": "INVALID-FORMAT",
        "model": "",
        "tokens_prompt": -10,
        "tokens_completion": "not_an_int",
        "cost_usd": -5.0,
    }

    res = validate_cost_allocation_payload(payload)
    assert res.is_valid is False
    assert len(res.errors) >= 5

    fields_with_errors = {e.field for e in res.errors}
    assert "tenant_id" in fields_with_errors
    assert "cost_center" in fields_with_errors
    assert "model" in fields_with_errors
    assert "tokens_prompt" in fields_with_errors
    assert "tokens_completion" in fields_with_errors
    assert "cost_usd" in fields_with_errors


def test_tag_sanitization_and_injection_defense():
    dirty_key = "USER-ROLE; DROP TABLE--"
    clean_k = sanitize_tag_key(dirty_key)
    assert ";" not in clean_k
    assert " " not in clean_k

    dirty_value = "admin' OR '1'='1\x00\n<script>"
    clean_v = sanitize_tag_value(dirty_value)
    assert "'" not in clean_v
    assert "\x00" not in clean_v
    assert "\n" not in clean_v


def test_tags_limit_and_non_dict_rejection():
    payload = {
        "tenant_id": "t1",
        "cost_center": "CC-FIN1",
        "model": "gpt-4o",
        "tokens_prompt": 100,
        "tokens_completion": 50,
        "cost_usd": 0.001,
        "tags": {f"tag_{i}": f"val_{i}" for i in range(40)},  # Exceeds max 32
    }

    res = validate_cost_allocation_payload(payload)
    assert res.is_valid is False
    assert any(e.code == "TOO_MANY_TAGS" for e in res.errors)
