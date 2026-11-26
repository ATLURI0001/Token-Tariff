"""Tests for quota_tier_contract."""
import pytest
from scripts.quota_tier_contract import validate_quota_tier


def test_quota_tier_contract():
    validate_quota_tier({"name": "pro", "monthly_limit": 100000})
    with pytest.raises(ValueError):
        validate_quota_tier({"name": "pro", "monthly_limit": 0})
