"""Multi-model token pricing normalizer and tiered volume tariff matrix for Token-Tariff."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


class UnknownModelError(KeyError):
    """Raised when an unmapped LLM model identifier is queried."""


class InvalidTokenCountError(ValueError):
    """Raised when token counts are negative or invalid."""


@dataclass(frozen=True)
class PricingTier:
    """Pricing rates in USD per 1 Million tokens for a specific cumulative usage tier."""

    min_cumulative_tokens: int
    prompt_per_m: float
    completion_per_m: float
    cache_read_per_m: float = 0.0
    cache_write_per_m: float = 0.0

    def __post_init__(self) -> None:
        if self.min_cumulative_tokens < 0:
            raise ValueError("min_cumulative_tokens must be >= 0")
        if any(r < 0 for r in (self.prompt_per_m, self.completion_per_m, self.cache_read_per_m, self.cache_write_per_m)):
            raise ValueError("Rates per 1M tokens must be >= 0.0")


@dataclass(frozen=True)
class CostBreakdown:
    """Detailed dollar breakdown for a calculated LLM completion request."""

    model_id: str
    prompt_tokens: int
    completion_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    prompt_cost: float
    completion_cost: float
    cache_read_cost: float
    cache_write_cost: float
    total_cost: float
    applied_tier_index: int

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "prompt_cost": round(self.prompt_cost, 6),
            "completion_cost": round(self.completion_cost, 6),
            "cache_read_cost": round(self.cache_read_cost, 6),
            "cache_write_cost": round(self.cache_write_cost, 6),
            "total_cost": round(self.total_cost, 6),
            "applied_tier_index": self.applied_tier_index,
        }


@dataclass
class ModelProfile:
    """Catalog entry describing a model's provider and tiered pricing schedule."""

    model_id: str
    provider: str
    tiers: List[PricingTier] = field(default_factory=list)

    def get_tier_for_usage(self, cumulative_tokens: int) -> tuple[int, PricingTier]:
        if not self.tiers:
            raise ValueError(f"No pricing tiers defined for model {self.model_id}")
        selected_idx = 0
        selected_tier = self.tiers[0]
        for idx, tier in enumerate(self.tiers):
            if cumulative_tokens >= tier.min_cumulative_tokens:
                selected_idx = idx
                selected_tier = tier
        return selected_idx, selected_tier


DEFAULT_CATALOG = [
    ModelProfile(
        model_id="gpt-4o",
        provider="openai",
        tiers=[
            PricingTier(min_cumulative_tokens=0, prompt_per_m=2.50, completion_per_m=10.00, cache_read_per_m=1.25),
            PricingTier(min_cumulative_tokens=10_000_000, prompt_per_m=2.25, completion_per_m=9.00, cache_read_per_m=1.12),
        ],
    ),
    ModelProfile(
        model_id="claude-3-5-sonnet",
        provider="anthropic",
        tiers=[
            PricingTier(min_cumulative_tokens=0, prompt_per_m=3.00, completion_per_m=15.00, cache_read_per_m=0.30, cache_write_per_m=3.75),
            PricingTier(min_cumulative_tokens=20_000_000, prompt_per_m=2.70, completion_per_m=13.50, cache_read_per_m=0.27, cache_write_per_m=3.37),
        ],
    ),
    ModelProfile(
        model_id="gemini-1-5-pro",
        provider="google",
        tiers=[
            PricingTier(min_cumulative_tokens=0, prompt_per_m=1.25, completion_per_m=5.00, cache_read_per_m=0.31),
        ],
    ),
]


class PriceNormalizer:
    """Normalizes token costs across heterogeneous LLM models and volume tiers."""

    def __init__(self, profiles: Optional[List[ModelProfile]] = None) -> None:
        self._profiles: Dict[str, ModelProfile] = {}
        for p in (profiles or DEFAULT_CATALOG):
            self.register_model(p)

    def register_model(self, profile: ModelProfile) -> None:
        """Register or update a model profile in the catalog."""
        sorted_tiers = sorted(profile.tiers, key=lambda t: t.min_cumulative_tokens)
        self._profiles[profile.model_id.lower().strip()] = ModelProfile(
            model_id=profile.model_id.lower().strip(),
            provider=profile.provider.lower().strip(),
            tiers=sorted_tiers,
        )

    def calculate_cost(
        self,
        model_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        cumulative_tokens: int = 0,
    ) -> CostBreakdown:
        """Calculate standardized USD cost for token consumption against applicable tier."""
        clean_id = model_id.lower().strip()
        if clean_id not in self._profiles:
            raise UnknownModelError(f"Model '{model_id}' is not registered in pricing catalog")

        for name, val in [
            ("prompt_tokens", prompt_tokens),
            ("completion_tokens", completion_tokens),
            ("cache_read_tokens", cache_read_tokens),
            ("cache_write_tokens", cache_write_tokens),
            ("cumulative_tokens", cumulative_tokens),
        ]:
            if not isinstance(val, int) or val < 0:
                raise InvalidTokenCountError(f"{name} must be a non-negative integer, got {val}")

        profile = self._profiles[clean_id]
        tier_idx, tier = profile.get_tier_for_usage(cumulative_tokens)

        prompt_cost = (prompt_tokens / 1_000_000.0) * tier.prompt_per_m
        completion_cost = (completion_tokens / 1_000_000.0) * tier.completion_per_m
        cache_read_cost = (cache_read_tokens / 1_000_000.0) * tier.cache_read_per_m
        cache_write_cost = (cache_write_tokens / 1_000_000.0) * tier.cache_write_per_m
        total_cost = prompt_cost + completion_cost + cache_read_cost + cache_write_cost

        return CostBreakdown(
            model_id=clean_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            prompt_cost=prompt_cost,
            completion_cost=completion_cost,
            cache_read_cost=cache_read_cost,
            cache_write_cost=cache_write_cost,
            total_cost=total_cost,
            applied_tier_index=tier_idx,
        )

    def compare_models(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        model_ids: Optional[List[str]] = None,
    ) -> List[dict]:
        """Rank queried models by total cost in ascending order."""
        targets = [m.lower().strip() for m in model_ids] if model_ids else list(self._profiles.keys())
        comparisons = []
        for mid in targets:
            if mid in self._profiles:
                breakdown = self.calculate_cost(
                    mid,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cache_read_tokens=cache_read_tokens,
                    cache_write_tokens=cache_write_tokens,
                )
                comparisons.append(breakdown.to_dict())

        comparisons.sort(key=lambda x: x["total_cost"])
        return comparisons
