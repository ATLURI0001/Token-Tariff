"""Quota tier configuration contract validator."""


def validate_quota_tier(tier: dict) -> None:
    if not isinstance(tier, dict):
        raise ValueError("tier must be a dict")
    if not tier.get("name") or not isinstance(tier["name"], str):
        raise ValueError("tier requires non-empty string 'name'")
    if "monthly_limit" not in tier or not isinstance(tier["monthly_limit"], int) or tier["monthly_limit"] <= 0:
        raise ValueError("tier requires positive integer 'monthly_limit'")
