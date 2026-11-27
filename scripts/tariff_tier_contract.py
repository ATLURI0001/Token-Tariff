"""tariff_tier_contract.py – validates tariff tier pricing contract."""


class TariffTierContract:
    def __init__(self, allowed_tiers: list[str]):
        self.allowed_tiers = allowed_tiers

    def validate(self, tier: str) -> bool:
        return tier in self.allowed_tiers
